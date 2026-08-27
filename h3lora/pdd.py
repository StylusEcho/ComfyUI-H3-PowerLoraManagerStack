"""Peel Acc/PDD output-head banks off H3 LoRAs and blend them at sample time.

alibaba-pai Acc LoRAs store a 32-interval bank as ``final_layer.video_out.set_weight``
of shape ``[N * 96, hidden]`` (and the audio twin). Stock Comfy ``copy_``s that
onto the native ``[96, hidden]`` head and crashes. Core PR 15908 teaches
``FinalLayer`` to blend those rows; this module does the same inside the Power
LoRA Stack so those files load without a core change.

The bank is kept beside the model (object patch + DIFFUSION_MODEL wrapper).
The native head is not resized, so a run without the bank is unchanged.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.minimax.model import _mod_row, time_shift_sigma
from comfy.patcher_extension import WrappersMP

LOG = logging.getLogger("h3.powerlorastack")

_VIDEO_WEIGHT = (
    "diffusion_model.final_layer.video_out.set_weight",
    "final_layer.video_out.set_weight",
    "proj_out.weight",
)
_VIDEO_BIAS = (
    "diffusion_model.final_layer.video_out.set_bias",
    "final_layer.video_out.set_bias",
    "proj_out.bias",
)
_AUDIO_WEIGHT = (
    "diffusion_model.final_layer.audio_out.set_weight",
    "final_layer.audio_out.set_weight",
    "audio_proj_out.weight",
)
_AUDIO_BIAS = (
    "diffusion_model.final_layer.audio_out.set_bias",
    "final_layer.audio_out.set_bias",
    "audio_proj_out.bias",
)


def _pop_first(sd, names):
    for name in names:
        if name in sd:
            return sd.pop(name), name
    return None, None


def _as_bank(tensor, out_features):
    """``[N, O, I]`` or ``[N*O, I]`` / ``[N*O]`` -> ``(N, flat)`` or None."""
    if tensor is None or out_features <= 0:
        return None
    t = tensor.detach().float().contiguous()
    if t.ndim == 3 and t.shape[1] == out_features:
        n = t.shape[0]
        return n, t.reshape(n * out_features, *t.shape[2:])
    if t.ndim == 2 and t.shape[1] == out_features:
        n = t.shape[0]
        return n, t.reshape(n * out_features)
    if t.ndim == 2 and t.shape[0] % out_features == 0:
        n = t.shape[0] // out_features
        return n, t
    if t.ndim == 1 and t.shape[0] % out_features == 0:
        n = t.shape[0] // out_features
        return n, t
    return None


def peel(sd, video_dim, audio_dim, hidden):
    """Pull a PDD bank out of a LoRA state dict. Returns ``(bank, note)``.

    ``bank`` is None when the file has no Acc heads. Matching keys that are
    not a whole number of native heads are dropped so stock ``set`` cannot crash.
    """
    vw, vw_key = _pop_first(sd, _VIDEO_WEIGHT)
    vb, vb_key = _pop_first(sd, _VIDEO_BIAS)
    aw, aw_key = _pop_first(sd, _AUDIO_WEIGHT)
    ab, ab_key = _pop_first(sd, _AUDIO_BIAS)
    if vw is None and aw is None:
        return None, ""

    def _restore():
        if vw is not None:
            sd[vw_key] = vw
        if vb is not None:
            sd[vb_key] = vb
        if aw is not None:
            sd[aw_key] = aw
        if ab is not None:
            sd[ab_key] = ab

    video = _as_bank(vw, video_dim) if vw is not None else None
    audio = _as_bank(aw, audio_dim) if aw is not None else None
    if video is None and vw is not None:
        LOG.warning("H3 PowerLoraStack: ignored %s shape %s (native video head is %d)",
                    vw_key, tuple(vw.shape), video_dim)
    if audio is None and aw is not None:
        LOG.warning("H3 PowerLoraStack: ignored %s shape %s (native audio head is %d)",
                    aw_key, tuple(aw.shape), audio_dim)
    if video is None and audio is None:
        return None, "PDD heads dropped (shape mismatch)"

    n = video[0] if video is not None else audio[0]
    if video is not None and video[0] != n:
        return None, f"PDD video n={video[0]} != audio n={n}"
    if audio is not None and audio[0] != n:
        return None, f"PDD audio n={audio[0]} != video n={n}"
    if n <= 1:
        _restore()
        return None, ""

    video_w = video[1] if video is not None else None
    audio_w = audio[1] if audio is not None else None
    if video_w is not None and video_w.shape[-1] != hidden:
        return None, f"PDD video in_features {video_w.shape[-1]} != {hidden}"
    if audio_w is not None and audio_w.shape[-1] != hidden:
        return None, f"PDD audio in_features {audio_w.shape[-1]} != {hidden}"

    video_b = None
    if vb is not None:
        packed = _as_bank(vb, video_dim)
        video_b = packed[1] if packed is not None and packed[0] == n else None
    audio_b = None
    if ab is not None:
        packed = _as_bank(ab, audio_dim)
        audio_b = packed[1] if packed is not None and packed[0] == n else None

    bank = {
        "n": n,
        "video_dim": video_dim,
        "audio_dim": audio_dim,
        "video_w": video_w,
        "video_b": video_b,
        "audio_w": audio_w,
        "audio_b": audio_b,
    }
    return bank, f"PDD bank n={n}"


def _blend(weight, bias, h, n, start, stop, flow_shift):
    grid = torch.linspace(1.0, 0.0, n + 1, dtype=torch.float64)
    dt = (1.0 - flow_shift * grid / (1.0 + (flow_shift - 1.0) * grid)).diff()[start:stop]
    w = (dt / dt.sum()).to(device=h.device, dtype=h.dtype)
    rows = weight.to(device=h.device, dtype=h.dtype).reshape(n, -1, weight.shape[-1])[start:stop]
    out_w = torch.einsum("n,noi->oi", w, rows)
    out_b = None
    if bias is not None:
        b_rows = bias.to(device=h.device, dtype=h.dtype).reshape(n, -1)[start:stop]
        out_b = torch.einsum("n,no->o", w, b_rows)
    return F.linear(h, out_w, out_b)


def interval(sigma, sigma_next, n, video_shift):
    start, stop = (round(float(1.0 - time_shift_sigma(s, video_shift, 1.0)) * n)
                   for s in (sigma, sigma_next))
    return start, max(stop, start + 1)


class PddFinalLayer(nn.Module):
    def __init__(self, inner, bank, ctx):
        super().__init__()
        self.inner = inner
        self.bank = bank
        self.ctx = ctx
        self.norm = inner.norm
        self.adaln_proj = inner.adaln_proj
        self.video_out = inner.video_out
        self.audio_out = inner.audio_out

    def forward(self, x, t_emb, video_seg, audio_seg):
        inner = self.inner
        bank = self.bank
        ctx = self.ctx
        n = bank["n"]
        sample_sigmas = ctx.get("sample_sigmas")
        sigma = ctx.get("sigma")
        if sample_sigmas is None or sigma is None:
            raise ValueError("MiniMax H3 PDD heads need the sampler's sigma schedule")

        shift, scale = inner.adaln_proj(t_emb)

        def mod(seg):
            a, b, row = seg
            return (inner.norm(x[a:b]) * (1.0 + _mod_row(scale, row, scale.dtype))
                    + _mod_row(shift, row, shift.dtype)).to(torch.float32)

        i = int((sample_sigmas - sigma).abs().argmin())
        sigma_next = sample_sigmas[min(i + 1, sample_sigmas.shape[0] - 1)]
        shifts = ctx.get("shifts") or (12.0, 3.0)
        start, stop = interval(sigma, sigma_next, n, shifts[0])
        vh = mod(video_seg)
        ah = mod(audio_seg)
        video = _blend(bank["video_w"], bank["video_b"], vh, n, start, stop, shifts[0])
        if bank["audio_w"] is None:
            audio = inner.audio_out(ah)
        else:
            audio = _blend(bank["audio_w"], bank["audio_b"], ah, n, start, stop, shifts[1])
        return video, audio


def attach(patcher, bank):
    fl = patcher.get_model_object("diffusion_model.final_layer")
    ctx = {"sigma": None, "sample_sigmas": None, "shifts": (12.0, 3.0)}
    wrapped = PddFinalLayer(fl, bank, ctx)
    patcher.add_object_patch("diffusion_model.final_layer", wrapped)

    def wrapper(executor, x, timestep, context, transformer_options=None, **kwargs):
        transformer_options = transformer_options or {}
        sigma = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        ctx["sigma"] = sigma
        ctx["sample_sigmas"] = transformer_options.get("sample_sigmas")
        ctx["shifts"] = (
            float(transformer_options.get("minimax_h3_sigma_shift_video", 12.0)),
            float(transformer_options.get("minimax_h3_sigma_shift_audio", 3.0)),
        )
        return executor(x, timestep, context, transformer_options, **kwargs)

    n = 0
    while patcher.get_wrappers(WrappersMP.DIFFUSION_MODEL, f"h3_power_lora_pdd_{n}"):
        n += 1
    patcher.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, f"h3_power_lora_pdd_{n}", wrapper)
    return n
