"""Capture MiniMax H3 adaLN sidecar tensors without touching native ComfyUI.

Curve H3 checkpoints often bake ``adaln_basis`` + ``adaln_mean`` (or a
``silu_t_emb_grid``) next to the DiT weights.  ComfyUI does not register those
keys on the UNet, so ``load_model_weights`` logs them as unexpected and drops
them.  The Power LoRA Stack needs them to port dense-trained adaLN pairs onto
the loaded curve basis.

We never read missing attributes off Comfy's model-config object: its
``__getattr__`` warns ``Please fix your code`` instead of raising.  Sidecars are
peeled from the checkpoint state dict (via a ``BaseModel.load_model_weights``
wrapper installed by this pack) and stashed on the H3 model instance.
"""

from __future__ import annotations

import logging

import torch

LOG = logging.getLogger("h3.powerlorastack")

SIDECAR_NAMES = ("adaln_basis", "adaln_mean", "silu_t_emb_grid")
STASH_ATTR = "_h3_adaln_sidecars"
PATCHER_ATTACHMENT = "h3_adaln_sidecars"


def _as_mapping(value):
    return value if isinstance(value, dict) else None


def _instance_dict(obj):
    try:
        return vars(obj)
    except TypeError:
        return None


def unet_config_of(model):
    """``unet_config`` from a BaseModel, never via model-config ``__getattr__``."""
    cfg = getattr(model, "model_config", None)
    data = _instance_dict(cfg) if cfg is not None else None
    if not data:
        return None
    unet = data.get("unet_config")
    return unet if isinstance(unet, dict) else None


def is_h3_base(model) -> bool:
    if model is None:
        return False
    if type(model).__name__ == "MiniMaxH3":
        return True
    diffusion = getattr(model, "diffusion_model", None)
    if diffusion is not None and type(diffusion).__name__ == "MiniMaxH3Model":
        return True
    unet = unet_config_of(model)
    return bool(unet) and unet.get("image_model") == "minimax_h3"


def _key_for(sd, prefix: str, name: str):
    prefix = prefix or ""
    for key in (prefix + name if prefix else None, name,
                None if prefix else "diffusion_model." + name):
        if key and key in sd:
            return key
    return None


def peel(sd, prefix: str = "") -> dict:
    """Pop H3 adaLN sidecar tensors out of a checkpoint state dict.

    Mutates ``sd``.  Non-tensor values are ignored and left in place.
    """
    if not isinstance(sd, dict) or not sd:
        return {}
    found = {}
    for name in SIDECAR_NAMES:
        key = _key_for(sd, prefix, name)
        if key is None:
            continue
        value = sd[key]
        if torch.is_tensor(value):
            found[name] = sd.pop(key)
    return found


def attach(model, sidecars: dict):
    """Stash sidecars on the H3 BaseModel and its DiT.  Survives ModelPatcher.clone."""
    if not sidecars or model is None:
        return
    existing = _as_mapping(getattr(model, STASH_ATTR, None)) or {}
    merged = {**existing, **sidecars}
    try:
        object.__setattr__(model, STASH_ATTR, merged)
    except Exception:
        pass
    diffusion = getattr(model, "diffusion_model", None)
    if diffusion is not None and diffusion is not model:
        try:
            object.__setattr__(diffusion, STASH_ATTR, merged)
        except Exception:
            pass
    patcher = getattr(model, "current_patcher", None)
    setter = getattr(patcher, "set_attachments", None) if patcher is not None else None
    if callable(setter):
        try:
            setter(PATCHER_ATTACHMENT, merged)
        except Exception:
            pass
    LOG.info("H3 PowerLoraStack: captured checkpoint adaLN sidecars %s",
             ", ".join(sorted(merged)))


def _take_named_tensors(obj, found: dict):
    extra = _as_mapping(getattr(obj, STASH_ATTR, None))
    if extra:
        found.update(extra)
    for name in SIDECAR_NAMES:
        try:
            value = object.__getattribute__(obj, name)
        except AttributeError:
            continue
        if torch.is_tensor(value):
            found.setdefault(name, value)


def collect(patcher=None, diffusion_model=None) -> dict | None:
    """Gather baked adaLN sidecars without touching model-config ``__getattr__``."""
    found: dict = {}
    objects = []
    model = None
    if patcher is not None:
        objects.append(patcher)
        getter = getattr(patcher, "get_attachment", None)
        if callable(getter):
            extra = _as_mapping(getter(PATCHER_ATTACHMENT))
            if extra:
                found.update(extra)
        model = getattr(patcher, "model", None)
    if model is not None:
        objects.append(model)
        cfg = getattr(model, "model_config", None)
        data = _instance_dict(cfg) if cfg is not None else None
        if data:
            extra = _as_mapping(data.get("adaln_sidecars"))
            if extra:
                found.update(extra)
        if diffusion_model is None:
            diffusion_model = getattr(model, "diffusion_model", None)
    if diffusion_model is not None:
        objects.append(diffusion_model)
    for obj in objects:
        if obj is None:
            continue
        _take_named_tensors(obj, found)
    return found or None


def wrap_load_model_weights(cls) -> bool:
    """Wrap ``cls.load_model_weights`` so H3 sidecar keys never become unexpected."""
    orig = getattr(cls, "load_model_weights", None)
    if not callable(orig) or getattr(orig, "_h3_adaln_hook", False):
        return bool(getattr(orig, "_h3_adaln_hook", False))

    def wrapped(self, sd, *args, **kwargs):
        prefix = kwargs["unet_prefix"] if "unet_prefix" in kwargs else (
            args[0] if args else "")
        if isinstance(sd, dict) and is_h3_base(self):
            found = peel(sd, prefix)
            if found:
                attach(self, found)
        return orig(self, sd, *args, **kwargs)

    wrapped._h3_adaln_hook = True
    wrapped._h3_adaln_orig = orig
    cls.load_model_weights = wrapped
    return True


def ensure_load_hook() -> bool:
    """Install the loader wrap on Comfy's BaseModel.  No-op if Comfy is absent.

    Also wraps ``MiniMaxH3`` when that subclass defines its own
    ``load_model_weights``, so a native override cannot bypass the peel.
    """
    try:
        from comfy import model_base
    except Exception:
        return False
    installed = False
    for name in ("BaseModel", "MiniMaxH3"):
        cls = getattr(model_base, name, None)
        if cls is None:
            continue
        if name != "BaseModel" and "load_model_weights" not in vars(cls):
            continue
        try:
            installed = wrap_load_model_weights(cls) or installed
        except Exception:
            LOG.debug("H3 PowerLoraStack: could not wrap %s.load_model_weights",
                      name, exc_info=True)
    return installed
