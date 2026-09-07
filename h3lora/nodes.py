"""The one node this pack provides: H3 Power LoRA Stack (LoRA Manager).

An alternative build of the MiniMax H3 Power LoRA Stack that draws its rows from
a `ComfyUI-Lora-Manager <https://github.com/willmiao/ComfyUI-Lora-Manager>`_
library instead of a bare folder listing.  It registers under its own node id so
it can sit alongside the original pack in the same ComfyUI install.
"""

from __future__ import annotations

import logging
import math
import os

import folder_paths

from . import apply as apply_mod
from . import loramanager

LOG = logging.getLogger("h3.powerloramanagerstack")
CATEGORY = "MiniMax-H3/lora"


class AnyType(str):
    def __ne__(self, other):
        return False


ANY = AnyType("*")


class FlexibleOptionalInputType(dict):
    """Accepts the arbitrary ``lora_N`` inputs the frontend adds at runtime.

    ComfyUI validates a prompt against INPUT_TYPES; a stack whose row count is
    decided in the browser has no fixed schema, so this dict reports that it
    contains every key and hands back a permissive type for the ones it does
    not know about.
    """

    def __init__(self, type, data=None):
        super().__init__()
        self.type = type
        self.data = data or {}
        self.update(self.data)

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        return (self.type,)

    def __contains__(self, key):
        return True


def _from_filename_list(name: str):
    """Match a stored name against ``models/loras``, tolerating separator drift."""
    path = folder_paths.get_full_path("loras", name)
    if path:
        return path
    wanted = name.replace("\\", "/").lower()
    matches = []
    for candidate in folder_paths.get_filename_list("loras"):
        normalized = candidate.replace("\\", "/").lower()
        if normalized == wanted:
            matches.append(candidate)
        elif "/" not in wanted and normalized.endswith("/" + wanted):
            matches.append(candidate)
    if len(matches) == 1:
        return folder_paths.get_full_path("loras", matches[0])
    if len(matches) > 1:
        LOG.warning("H3 PowerLoraManagerStack: ambiguous lora name %r (%d matches)",
                    name, len(matches))
    return None


def _resolve_lora(name: str, manager_path: str | None = None):
    """Map a stored row onto a real file.

    A row picked in the search bar carries the absolute path the LoRA Manager
    reported for it.  ``models/loras`` is still tried first, so a workflow keeps
    resolving through ComfyUI's own listing when the manager is not running;
    the manager's path is the fallback, which is what lets a library configured
    under ``extra_loras_roots`` -- outside ``models/loras`` entirely -- load at
    all.
    """
    if name and name != "None":
        path = _from_filename_list(name)
        if path:
            return path
    if manager_path:
        candidate = str(manager_path).replace("\\", os.sep).replace("/", os.sep)
        if os.path.isfile(candidate):
            return candidate
        # The library moved, or the workflow came from another machine: the
        # basename is still worth one pass through the local listing.
        base = os.path.basename(candidate)
        if base and base != name:
            path = _from_filename_list(base)
            if path:
                return path
    return None


def _row_strength(value, order, issues):
    try:
        strength = float(value.get("strength", 1.0))
    except (TypeError, ValueError):
        message = f"row {order}: strength is not numeric"
        LOG.warning("H3 PowerLoraManagerStack: %s", message)
        issues.append(message)
        return None
    if not math.isfinite(strength):
        message = f"row {order}: strength must be finite"
        LOG.warning("H3 PowerLoraManagerStack: %s", message)
        issues.append(message)
        return None
    return strength


def _collect(kwargs, issues=None):
    """Pull enabled rows out of the dynamic ``lora_N`` inputs, in UI order."""
    if issues is None:
        issues = []
    rows = []
    for key, value in kwargs.items():
        if not key.lower().startswith("lora_") or not isinstance(value, dict):
            continue
        if "lora" not in value:
            continue
        try:
            order = int(key.split("_", 1)[1])
        except (IndexError, ValueError):
            order = 1 << 30
        rows.append((order, value))
    rows.sort(key=lambda item: item[0])

    entries = []
    for order, value in rows:
        if not value.get("on", True):
            continue
        strength = _row_strength(value, order, issues)
        if strength is None or strength == 0.0:
            continue
        name = value.get("lora")
        if not name or name == "None":
            continue        # a row the user added but has not filled in yet
        manager_path = value.get("lmPath") or None
        path = _resolve_lora(name, manager_path)
        if path is None:
            message = f"row {order}: could not resolve lora {name!r}"
            LOG.warning("H3 PowerLoraManagerStack: %s", message)
            issues.append(message)
            continue
        words = value.get("triggerWords")
        entries.append({
            "name": name,
            "path": path,
            "strength": strength,
            "row": order,
            "trigger_words": [w for w in words if isinstance(w, str)]
            if isinstance(words, list) else None,
        })
    return entries


def _collect_lora_stack(lora_stack, issues):
    """Adopt a ``LORA_STACK`` produced upstream, e.g. by Lora Stacker (LoraManager).

    Entries are ``(name, model_strength, clip_strength)``; H3 has no CLIP tower
    in this path, so the clip strength is ignored rather than silently averaged
    into the model strength.
    """
    entries = []
    if not lora_stack:
        return entries
    for offset, item in enumerate(lora_stack):
        try:
            name, model_strength = item[0], float(item[1])
        except (TypeError, ValueError, IndexError):
            message = f"lora_stack entry {offset + 1}: unreadable"
            LOG.warning("H3 PowerLoraManagerStack: %s", message)
            issues.append(message)
            continue
        if not math.isfinite(model_strength) or model_strength == 0.0:
            continue
        path = _resolve_lora(str(name), str(name))
        if path is None:
            message = f"lora_stack entry {offset + 1}: could not resolve lora {name!r}"
            LOG.warning("H3 PowerLoraManagerStack: %s", message)
            issues.append(message)
            continue
        # Zero and down: an H3 LoRA Schedule selector only parses one-based row
        # numbers, so these can be swept up by ``all`` but never picked out by a
        # number that means one of this node's own visible rows.
        entries.append({
            "name": os.path.basename(str(name)),
            "path": path,
            "strength": model_strength,
            "row": -offset,
            "trigger_words": None,
        })
    return entries


def _trigger_words(entries) -> str:
    """Trigger words for the applied rows, in LoRA Manager's own wire format.

    Words picked up in the search bar ride along on the row; anything else --
    a row restored from a saved workflow, or one that arrived over
    ``lora_stack`` -- is read back out of the manager's ``.metadata.json``
    sidecar.  ``",, "`` is the separator Lora Stacker (LoraManager) emits, so
    the string drops straight into TriggerWord Toggle (LoraManager).
    """
    words = []
    for entry in entries:
        found = entry.get("trigger_words")
        if not found:
            found = loramanager.trigger_words(entry["path"])
        for word in found or ():
            word = word.strip()
            if word and word not in words:
                words.append(word)
    return ",, ".join(words)


class H3PowerLoraManagerStack:
    """Stacked multi-LoRA loader for MiniMax H3, fed from a LoRA Manager library.

    Handles the four things generic LoRA loaders get wrong on H3:

    * quantized bases -- w4a8/int4 weights are patched with an exact runtime
      branch instead of a lossy dequantize/requantize merge;
    * adaLN basis mismatch -- LoRAs trained against a dense checkpoint are
      rebased onto a pruned checkpoint's curve, instead of being skipped;
    * Acc/PDD head banks -- ``video_out.set_weight`` of shape ``[N*96, hidden]``
      is peeled off before the stock ``set`` path and blended per sampler step;
    * key conventions -- ai-toolkit, kohya, lycoris, peft and bare-prefix
      layouts all resolve against the model's own key set.

    Rows are added through a search bar that queries ComfyUI-Lora-Manager, so
    the library's names, previews, base models and trigger words are what you
    pick from.  Without the manager installed the search bar falls back to
    ``models/loras`` and the node behaves like an ordinary stack.

    The ``adaln_modality`` and ``schedule`` inputs take the MiniMax H3 adaLN
    Modality and H3 LoRA Schedule nodes from the original Power LoRA Stack pack,
    which this node is meant to be installed beside.
    """

    @classmethod
    def INPUT_TYPES(cls):
        modality = {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}
        return {
            "required": {},
            "optional": FlexibleOptionalInputType(ANY, {
                "model": ("MODEL",),
                "quantized_layers": (["auto", "branch", "merge"], {
                    "default": "auto",
                    "tooltip": "auto: runtime branch for quantized layers, merge for the rest. "
                               "branch: never modify a weight. merge: stock behaviour "
                               "(destroys LoRAs on w4a8/int4).",
                }),
                "adaln_port": (["auto", "strip", "off"], {
                    "default": "auto",
                    "tooltip": "auto: rebase adaLN pairs on this stack and any already on MODEL "
                               "between dense (2688) and curve (8), including curve-to-curve "
                               "when the LoRA ships a table. strip: drop mismatched pairs. "
                               "off: leave them.",
                }),
                "adaln_video": ("FLOAT", dict(modality, tooltip=(
                    "Scale every stacked LoRA's adaLN modulation for the video modality "
                    "(tag 0). All three at 1.0 is a no-op; 0.0 removes that modality's "
                    "share of every adapter."))),
                "adaln_text": ("FLOAT", dict(modality, tooltip=(
                    "Scale every stacked LoRA's adaLN modulation for the "
                    "text/conditioning modality (tag 1)."))),
                "adaln_audio": ("FLOAT", dict(modality, tooltip=(
                    "Scale every stacked LoRA's adaLN modulation for the audio modality "
                    "(tag 2)."))),
                "adaln_modality": ("H3_MODALITY", {
                    "tooltip": "Optional. A MiniMax H3 adaLN Modality node -- from the "
                               "original Power LoRA Stack pack -- wired here scales every "
                               "stacked LoRA's adaLN modulation per modality. It takes "
                               "precedence over the adaln_video/text/audio widgets above.",
                }),
                "schedule": ("H3_SCHEDULE", {
                    "tooltip": "Optional. An H3 LoRA Schedule chain -- from the original "
                               "Power LoRA Stack pack -- varies selected row strengths over "
                               "the denoising trajectory. Rows are this node's own, numbered "
                               "top to bottom; lora_stack entries are only reached by 'all'.",
                }),
                "lora_stack": ("LORA_STACK", {
                    "tooltip": "Optional. A LORA_STACK from Lora Stacker (LoraManager) or any "
                               "other stacker; its entries are applied ahead of this node's "
                               "own rows. CLIP strengths are ignored -- H3 has no CLIP tower "
                               "on this path.",
                }),
            }),
            "hidden": {},
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "report", "trigger_words")
    OUTPUT_TOOLTIPS = (
        "Patched model",
        "Per-LoRA account of what was applied",
        "Trigger words the LoRA Manager holds for the applied LoRAs, ',, '-separated",
    )
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = __doc__

    def apply(self, model=None, quantized_layers="auto", adaln_port="auto",
              adaln_video=1.0, adaln_text=1.0, adaln_audio=1.0,
              adaln_modality=None, schedule=None, lora_stack=None, **kwargs):
        if model is None:
            raise ValueError("H3 Power LoRA Stack (LoRA Manager): no model connected")

        issues = []
        rows = _collect(kwargs, issues)
        # Upstream stack entries land ahead of this node's rows, matching how a
        # chain of stackers reads on the canvas.
        entries = _collect_lora_stack(lora_stack, issues) + rows
        widgets = {
            "video": float(adaln_video),
            "text": float(adaln_text),
            "audio": float(adaln_audio),
        }
        # One wired node beats three widgets -- but never silently: a setting on
        # the widgets that the wire discards is called out in the report.
        modality = widgets if adaln_modality is None else adaln_modality
        overridden = adaln_modality is not None and any(
            value != 1.0 for value in widgets.values())
        try:
            patcher, report = apply_mod.apply_stack(
                model, entries, mode=quantized_layers, adaln_mode=adaln_port,
                modality=modality, schedule=schedule,
            )
        except Exception as exc:
            LOG.exception("H3 Power LoRA Stack (LoRA Manager) failed; "
                          "passing the model through")
            extra = "\n".join(issues)
            return (model, f"stack failed ({exc}); model unchanged"
                    + ("\n" + extra if extra else ""), "")
        text = report.text()
        if not entries and not text.strip():
            text = "no LoRAs enabled"
        if overridden:
            text += ("\n  ! adaln_modality is wired, so the node's own "
                     "adaln_video/text/audio widgets were ignored")
        if issues:
            text = text + ("\n" if text else "") + "\n".join(issues)
            # A row that will not resolve is the one moment the optional
            # dependency is worth naming: without the manager installed, a
            # library outside models/loras has no way to be found.
            if any("could not resolve" in issue for issue in issues) \
                    and not loramanager.installed():
                text += ("\n  ! ComfyUI-Lora-Manager is not installed, so rows can "
                         "only resolve through models/loras")
        words = _trigger_words(entries)
        LOG.info("H3 Power LoRA Stack (LoRA Manager):\n%s", text)
        return (patcher, text, words)


NODE_CLASS_MAPPINGS = {
    "H3PowerLoraManagerStack": H3PowerLoraManagerStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PowerLoraManagerStack": "H3 Power LoRA Stack (LoRA Manager)",
}
