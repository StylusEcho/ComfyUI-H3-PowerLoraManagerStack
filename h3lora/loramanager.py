"""Bridge to willmiao/ComfyUI-Lora-Manager.

The manager's Python package is deliberately never imported.  Its module layout
(``py.services``, ``py.utils``) moves between releases and importing another
pack's internals would tie this node to a version of it.  Everything needed is
reachable two ways that do not:

* **In the browser** -- the manager registers its HTTP API on ComfyUI's own
  aiohttp app, so the search bar queries ``/api/lm/loras/list`` from the same
  origin.  That is where results, previews, base models and trigger words come
  from while a stack is being built.  See ``web/h3_power_lora_manager_stack.js``.

* **In Python** -- the manager writes a ``<model>.metadata.json`` sidecar next
  to every model it has scanned.  This module reads that file, which is what
  recovers trigger words and the saved strength preset for a row that arrived
  from a saved workflow rather than from this session's search bar.

Both paths degrade to nothing when the manager is not installed: the node keeps
working against ``models/loras`` alone.
"""

from __future__ import annotations

import json
import logging
import os

LOG = logging.getLogger("h3.powerloramanagerstack")

# Written by the manager next to each model file, e.g. ``style.safetensors``
# alongside ``style.metadata.json``.
SIDECAR_SUFFIX = ".metadata.json"

# Files that identify a ComfyUI-Lora-Manager checkout, relative to its root.
_MARKERS = ("py/nodes/lora_stacker.py", "py/services/lora_service.py")

_cache: dict[str, tuple[tuple, dict]] = {}


def sidecar_path(model_path: str) -> str:
    """``/loras/style.safetensors`` -> ``/loras/style.metadata.json``."""
    return os.path.splitext(model_path)[0] + SIDECAR_SUFFIX


def read_sidecar(model_path: str) -> dict | None:
    """The manager's metadata for one LoRA file, or ``None``.

    Cached on the sidecar's own (mtime, size) so a stack of ten rows costs ten
    stats rather than ten JSON parses on every prompt.
    """
    if not model_path:
        return None
    path = sidecar_path(model_path)
    try:
        stat = os.stat(path)
    except OSError:
        _cache.pop(path, None)
        return None
    stamp = (stat.st_mtime_ns, stat.st_size)
    hit = _cache.get(path)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        LOG.debug("H3 PowerLoraManagerStack: unreadable LoRA Manager sidecar %s (%s)",
                  path, exc)
        return None
    if not isinstance(data, dict):
        return None
    _cache[path] = (stamp, data)
    return data


def trigger_words(model_path: str) -> list[str]:
    """Civitai trained words the manager recorded for this file."""
    data = read_sidecar(model_path)
    if not data:
        return []
    civitai = data.get("civitai")
    words = civitai.get("trainedWords") if isinstance(civitai, dict) else None
    if not isinstance(words, list):
        words = data.get("trainedWords")
    if not isinstance(words, list):
        return []
    return [word for word in words if isinstance(word, str) and word.strip()]


def usage_tips(model_path: str) -> dict:
    """The manager's saved presets for this file (``strength`` and friends).

    ``usage_tips`` is stored as a JSON *string* inside the sidecar, so it is
    parsed here rather than by the caller.
    """
    data = read_sidecar(model_path)
    raw = data.get("usage_tips") if data else None
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def preferred_strength(model_path: str) -> float | None:
    """The strength preset saved in the manager, if there is a usable one."""
    value = usage_tips(model_path).get("strength")
    try:
        strength = float(value)
    except (TypeError, ValueError):
        return None
    return strength if strength == strength and abs(strength) != float("inf") else None


def describe(model_path: str) -> dict:
    """The handful of manager fields the node's report can use."""
    data = read_sidecar(model_path)
    if not data:
        return {}
    civitai = data.get("civitai") if isinstance(data.get("civitai"), dict) else {}
    return {
        "model_name": data.get("model_name") or "",
        "base_model": data.get("base_model") or "",
        "version": civitai.get("name") or "",
        "favorite": bool(data.get("favorite", False)),
    }


def root() -> str | None:
    """Path of the installed ComfyUI-Lora-Manager, or ``None``.

    Used only to say so in the node's report; nothing is imported from it.
    """
    try:
        import folder_paths
    except Exception:
        return None
    getter = getattr(folder_paths, "get_folder_paths", None)
    roots = []
    if callable(getter):
        try:
            roots = list(getter("custom_nodes"))
        except Exception:
            roots = []
    if not roots:
        base = getattr(folder_paths, "base_path", None)
        if base:
            roots = [os.path.join(base, "custom_nodes")]
    for parent in roots:
        try:
            entries = sorted(os.listdir(parent))
        except OSError:
            continue
        for entry in entries:
            candidate = os.path.join(parent, entry)
            if all(os.path.isfile(os.path.join(candidate, m)) for m in _MARKERS):
                return candidate
    return None


def installed() -> bool:
    return root() is not None
