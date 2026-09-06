"""ComfyUI-H3-PowerLoraStack: stacked multi-LoRA loading for MiniMax H3."""

from .h3lora import server as _server
from .h3lora.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .h3lora.sidecars import ensure_load_hook

WEB_DIRECTORY = "./web"

_server.register()      # /h3_power_lora_stack/balance, for the auto-balance button
ensure_load_hook()      # peel adaln_basis/adaln_mean before Comfy drops them

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
