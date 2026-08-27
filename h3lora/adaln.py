"""AdaLN basis conversion between dense and curve MiniMax H3 checkpoints.

H3 ships in two forms.  A *dense* checkpoint feeds ``silu(time_embedder(t))``,
a 2688-dim vector, into every ``adaln_proj.linear``.  A *curve* (pruned)
checkpoint drops the time embedder and instead stores a precomputed
``adaln_t_table`` of shape ``[grid, k]`` (k = 8 locally); the linears then take
a k-dim input.  ComfyUI's tables are mean-centered, i.e. the two spaces are
related by

    S(t) = c + V @ table(t)          V: [2688, k],  c: [2688]

A LoRA trained on one form has an ``adaln_proj.linear.lora_A`` of the wrong
width for the other, which is why ComfyUI logs

    shape '[96768, 8]' is invalid for input of size 260112384

and silently skips the layer.  Stripping those pairs is not an acceptable fix:
on the turbo distillation LoRAs the constant term alone is ~100% of the
magnitude of ``dW @ S(t)``, so dropping adaLN discards essentially the whole
adapter.  Instead we change basis, which preserves rank:

    dense -> curve   A' = A @ V,       plus bias delta  B @ (A @ c)
    curve -> dense   A' = A @ pinv(V), plus bias delta -B @ (A' @ c)

The reverse direction needs the pseudo-inverse rather than the transpose: the
least-squares ``V`` spans the right subspace but its columns are not
orthonormal, so ``V.T`` would not satisfy ``M @ V == I``.

The bias delta is emitted as ``.diff_b``; ``comfy.lora.load_lora`` turns that
into a ``("diff",)`` patch on the sibling ``.bias``.  Without it the port is
nearly worthless.

``V`` and ``c`` are recovered by least squares of ``[1 | table]`` against the
silu grid.  The fit must use the *target model's own* table: comfy's tables are
mean-centered while some third-party bakes are not, so tables from different
checkpoints are not interchangeable even though each is self-consistent with
its own weights.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import struct

import torch
import torch.nn.functional as F

LOG = logging.getLogger(__name__)

GRID_FILENAME = "h3_silu_temb_grid.safetensors"
_GRID_TENSOR = "silu_t_emb_grid"
BASIS_SUBDIR = "h3_adaln"
TE_IN_W = "time_embedder.proj_in.weight"
TE_IN_B = "time_embedder.proj_in.bias"
TE_OUT_W = "time_embedder.proj_out.weight"
TE_OUT_B = "time_embedder.proj_out.bias"
MAX_RESIDUAL = 5e-3
FREQ_DIM = 256

_grid_cache: dict[tuple[str, int, int], torch.Tensor] = {}
_basis_cache: dict[tuple, tuple] = {}


def find_silu_grid(extra_dirs=()) -> str:
    """Locate ``h3_silu_temb_grid.safetensors`` in the usual places."""
    candidates = list(extra_dirs)
    try:
        import folder_paths
        candidates.append(os.path.join(folder_paths.models_dir, "h3_adaln"))
        candidates.append(folder_paths.get_folder_paths("loras")[0])
        candidates.append(os.path.join(folder_paths.models_dir, "diffusion_models"))
        candidates.append(folder_paths.get_folder_paths("custom_nodes")[0]
                          if "custom_nodes" in getattr(folder_paths, "folder_names_and_paths", {})
                          else os.path.join(os.path.dirname(folder_paths.models_dir), "custom_nodes"))
    except Exception:
        pass
    for base in candidates:
        if not base or not os.path.isdir(base):
            continue
        direct = os.path.join(base, GRID_FILENAME)
        if os.path.isfile(direct):
            return direct
        # one level down covers custom_nodes/<pack>/h3_silu_temb_grid.safetensors
        try:
            for entry in os.listdir(base):
                nested = os.path.join(base, entry, GRID_FILENAME)
                if os.path.isfile(nested):
                    return nested
        except OSError:
            continue
    return ""


def load_silu_grid(path: str) -> torch.Tensor:
    """Load the ``[grid, 2688]`` silu(t_emb) grid, cached by file version."""
    stat = os.stat(path)
    cache_key = (path, stat.st_mtime_ns, stat.st_size)
    cached = _grid_cache.get(cache_key)
    if cached is not None:
        return cached
    from safetensors.torch import load_file
    sd = load_file(path)
    if _GRID_TENSOR not in sd:
        raise ValueError(f"{path} does not contain '{_GRID_TENSOR}'")
    grid = sd[_GRID_TENSOR].to(torch.float32)
    for old_key in tuple(_grid_cache):
        if old_key[0] == path and old_key != cache_key:
            _grid_cache.pop(old_key, None)
    _grid_cache[cache_key] = grid
    return grid


def _cpu_f32(t):
    return t.detach().to(device="cpu", dtype=torch.float32).contiguous()


def tensor_hash(tensor):
    t = _cpu_f32(tensor)
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


def silu_temb_grid(proj_in_w, proj_in_b, proj_out_w, proj_out_b,
                   rows=1025, freq_dim=FREQ_DIM):
    device = proj_in_w.device
    t = torch.arange(rows, device=device, dtype=torch.float32) / float(rows - 1)
    half = freq_dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(
        half, device=device, dtype=torch.float32) / half)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    hidden = F.silu(emb @ proj_in_w.float().T + proj_in_b.float())
    temb = hidden @ proj_out_w.float().T + proj_out_b.float()
    return F.silu(temb)


def grid_from_time_embedder(time_embedder, rows=1025):
    try:
        proj_in, proj_out = time_embedder.proj_in, time_embedder.proj_out
        tensors = (proj_in.weight, proj_in.bias, proj_out.weight, proj_out.bias)
        if any(t is None or t.device.type == "meta" for t in tensors):
            return None
        freq_dim = int(getattr(time_embedder, "freq_dim", FREQ_DIM))
        return silu_temb_grid(*tensors, rows=rows, freq_dim=freq_dim).clone()
    except Exception:
        return None


def _safetensors_header(path):
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            return None
        size = struct.unpack("<Q", raw)[0]
        if size <= 0 or size > (64 << 20):
            return None
        return json.loads(fh.read(size))


def _time_embedder_prefix(header):
    if not header:
        return None
    if any(k.endswith("adaln_t_table") for k in header):
        return None
    for key in header:
        if key.endswith(TE_OUT_W):
            prefix = key[:-len(TE_OUT_W)]
            if all(prefix + n in header for n in (TE_IN_W, TE_IN_B, TE_OUT_B)):
                return prefix
    return None


def _load_named(path, names):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        return [_cpu_f32(f.get_tensor(n)) for n in names]


def _basis_dir():
    try:
        import folder_paths
        return os.path.join(folder_paths.models_dir, BASIS_SUBDIR)
    except Exception:
        return ""


def _cache_path(key):
    d = _basis_dir()
    return os.path.join(d, "basis_%s.safetensors" % key) if d else ""


def _read_basis_cache(key):
    path = _cache_path(key)
    if not path or not os.path.isfile(path):
        return None
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
            return (_cpu_f32(f.get_tensor("V")), _cpu_f32(f.get_tensor("c")),
                    float(meta.get("residual", "nan")),
                    meta.get("source", "cache"))
    except Exception:
        return None


def _write_basis_cache(key, v, c, residual, source):
    path = _cache_path(key)
    if not path:
        return
    try:
        from safetensors.torch import save_file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_file({"V": v.contiguous(), "c": c.contiguous()}, path,
                  metadata={"residual": repr(residual), "source": source, "table_hash": key})
    except Exception:
        pass


def _iter_diffusion_safetensors():
    try:
        import folder_paths
    except Exception:
        return
    for name in folder_paths.get_filename_list("diffusion_models"):
        if not name.lower().endswith(".safetensors"):
            continue
        path = folder_paths.get_full_path("diffusion_models", name)
        if path:
            yield os.path.basename(name), path


def _grid_from_checkpoint(path, rows):
    try:
        prefix = _time_embedder_prefix(_safetensors_header(path))
        if prefix is None:
            return None
        tensors = _load_named(path, [prefix + n for n in (TE_IN_W, TE_IN_B, TE_OUT_W, TE_OUT_B)])
        return silu_temb_grid(*tensors, rows=rows)
    except Exception:
        return None


def _table_from_checkpoint(path):
    try:
        header = _safetensors_header(path)
        key = next((k for k in (header or ()) if k.endswith("adaln_t_table")), None)
        if key is None:
            return None
        return _load_named(path, [key])[0]
    except Exception:
        return None


def fit_basis(table: torch.Tensor, grid: torch.Tensor):
    """Least-squares fit of ``grid ~= 1 c^T + table V^T``.

    ``table`` is the target model's ``adaln_t_table`` ``[G, k]``; ``grid`` is
    ``[G, 2688]``.  Returns ``(V [2688, k], c [2688])``.
    """
    if table.shape[0] != grid.shape[0]:
        raise ValueError(
            f"adaLN table grid ({table.shape[0]}) and silu grid ({grid.shape[0]}) "
            "have different resolutions; they must come from the same bake"
        )
    table_cpu = _cpu_f32(table)
    grid_cpu = _cpu_f32(grid)
    key = (tensor_hash(table_cpu), tensor_hash(grid_cpu))
    cached = _basis_cache.get(key)
    if cached is not None:
        return cached

    t = table_cpu.to(torch.float64)
    s = grid_cpu.to(torch.float64)
    design = torch.cat([torch.ones(t.shape[0], 1, dtype=torch.float64), t], dim=1)  # [G, 1+k]
    solution = torch.linalg.lstsq(design, s).solution                                # [1+k, 2688]
    c = solution[0].contiguous()                                                     # [2688]
    v = solution[1:].transpose(0, 1).contiguous()                                    # [2688, k]

    residual = float((design @ solution - s).norm() / s.norm().clamp(min=1e-12))
    out = (v.to(torch.float32), c.to(torch.float32), residual)
    _basis_cache[key] = out
    return out


def sidecar_basis(sidecars, table=None):
    if not sidecars:
        return None
    v, c = sidecars.get("adaln_basis"), sidecars.get("adaln_mean")
    if torch.is_tensor(v) and torch.is_tensor(c):
        v, c = _cpu_f32(v), _cpu_f32(c)
        if v.ndim == 2 and c.ndim == 1 and v.shape[0] == c.shape[0]:
            if table is None or v.shape[1] == table.shape[1]:
                return v, c, 0.0, "checkpoint sidecars"
    grid = sidecars.get("silu_t_emb_grid")
    if torch.is_tensor(grid) and table is not None:
        v, c, residual = fit_basis(table, grid)
        return v, c, residual, "checkpoint silu_t_emb_grid"
    return None


def _accept(residual):
    return residual == residual and residual <= MAX_RESIDUAL


def fit_table_to_table(source_table: torch.Tensor, target_table: torch.Tensor):
    """Map a source curve basis onto a target curve basis.

    Used when a LoRA ships its own ``adaln_t_table``: the two bakes span nearly
    the same subspace but with different rotations and centering, so a LoRA
    trained against one is wrong on the other even though the shapes match.
    Fitting ``source ~= a + target M`` needs no silu grid at all.

    Returns ``(M [k_src, k_tgt], a [k_src], residual)``.
    """
    src = _cpu_f32(source_table).to(torch.float64)
    tgt = _cpu_f32(target_table).to(torch.float64)
    if src.shape[0] != tgt.shape[0]:
        raise ValueError(
            f"adaLN tables have different grid resolutions ({src.shape[0]} vs {tgt.shape[0]})"
        )
    design = torch.cat([torch.ones(tgt.shape[0], 1, dtype=torch.float64), tgt], dim=1)
    solution = torch.linalg.lstsq(design, src).solution      # [1+k_tgt, k_src]
    residual = float((design @ solution - src).norm() / src.norm().clamp(min=1e-12))
    a = solution[0].contiguous()                             # [k_src]
    m = solution[1:].transpose(0, 1).contiguous()            # [k_src, k_tgt]
    return m.to(torch.float32), a.to(torch.float32), residual


class AdalnContext:
    """Everything needed to move adaLN LoRA pairs between the two bases."""

    def __init__(self, target_dim: int, table=None, grid_path: str = "",
                 sidecars=None, time_embedder=None):
        self.target_dim = int(target_dim) if target_dim is not None else 0
        self.table = table
        self.grid_path = grid_path
        self.sidecars = sidecars or {}
        self.time_embedder = time_embedder
        self.residual = None
        self.source = ""
        self._basis = None
        self._failed = False
        self._curve_cache = {}

    @property
    def is_curve(self) -> bool:
        return self.table is not None

    def _record(self, v, c, residual, source):
        if not _accept(residual):
            LOG.warning(
                "H3 PowerLoraStack: adaLN basis from %s residual %.2e exceeds %.0e",
                source, residual, MAX_RESIDUAL,
            )
            return None
        LOG.info("H3 PowerLoraStack: adaLN basis from %s, residual %.2e", source, residual)
        self._basis = (v, c, residual)
        self.residual = residual
        self.source = source
        return self._basis

    def _fit_grid(self, table, grid, source):
        v, c, residual = fit_basis(table.detach().to("cpu"), grid)
        return self._record(v, c, residual, source)

    def _scan_grids(self, table):
        rows = int(table.shape[0])
        table_key = tensor_hash(table)
        best = None
        for label, path in _iter_diffusion_safetensors():
            grid = _grid_from_checkpoint(path, rows)
            if grid is None:
                continue
            key = "d%s%s" % (table_key[:8], tensor_hash(grid)[:8])
            cached = _read_basis_cache(key)
            if cached is not None:
                v, c, residual, _src = cached
            else:
                v, c, residual = fit_basis(table.detach().to("cpu"), grid)
                _write_basis_cache(key, v, c, residual, label)
            if best is None or residual < best[2]:
                best = (v, c, residual, label)
        return self._record(*best) if best is not None else None

    def _scan_tables(self, grid):
        grid_key = tensor_hash(grid)
        best = None
        for label, path in _iter_diffusion_safetensors():
            table = _table_from_checkpoint(path)
            if table is None or table.shape[0] != grid.shape[0]:
                continue
            key = "d%s%s" % (tensor_hash(table)[:8], grid_key[:8])
            cached = _read_basis_cache(key)
            if cached is not None:
                v, c, residual, _src = cached
            else:
                v, c, residual = fit_basis(table, grid)
                _write_basis_cache(key, v, c, residual, label)
            if best is None or residual < best[2]:
                best = (v, c, residual, label)
        if best is None:
            return None
        return self._record(*best)

    def basis(self, curve_table=None):
        """``(V, c, residual)`` relating the dense space to a curve basis."""
        table = self.table if self.table is not None else curve_table
        cache_key = None if table is None else tensor_hash(table)
        if table is not self.table and cache_key in self._curve_cache:
            return self._curve_cache[cache_key]
        if self.table is not None and (self._basis is not None or self._failed):
            return self._basis
        if self._failed and table is None:
            return None

        got = None
        if table is not None and table is self.table:
            got = sidecar_basis(self.sidecars, table)
            if got is not None:
                got = self._record(*got)

        if got is None and table is not None:
            live = grid_from_time_embedder(self.time_embedder, rows=int(table.shape[0])) if self.time_embedder is not None else None
            if live is not None:
                got = self._fit_grid(table, live, "live time_embedder")

        if got is None and table is not None and self.grid_path:
            try:
                got = self._fit_grid(table, load_silu_grid(self.grid_path), self.grid_path)
            except Exception as exc:
                LOG.warning("H3 PowerLoraStack: could not load silu grid (%s)", exc)

        if got is None and table is not None:
            got = self._scan_grids(table)

        if got is None and table is None:
            live = grid_from_time_embedder(self.time_embedder) if self.time_embedder is not None else None
            if live is not None:
                got = sidecar_basis(self.sidecars, None)
                if got is not None:
                    got = self._record(*got)
                if got is None:
                    got = self._scan_tables(live)

        if got is None:
            if not self._failed:
                LOG.warning(
                    "H3 PowerLoraStack: adaLN porting unavailable - no baked basis, "
                    "silu grid, live time embedder, or matching diffusion_models bake"
                )
                self._failed = True
            if table is not self.table and table is not None:
                self._curve_cache[cache_key] = None
            return None

        if table is not self.table and table is not None:
            self._curve_cache[cache_key] = got
        return got


def port_adaln_pairs(sd: dict, ctx: AdalnContext, source_table=None):
    """Rebase every ``adaln_proj.linear`` LoRA pair onto the target's basis.

    ``source_table`` is the LoRA's own ``adaln_t_table`` if it shipped one.
    That is the better route when available: two curve bakes can have the same
    width but different rotations and centering, so a LoRA whose adaLN width
    already matches may still be wrong on this checkpoint, and only the table
    reveals it.  It also needs no silu grid.

    Mutates nothing; returns ``(new_sd, stats)``.
    """
    stats = {"ported": 0, "skipped": 0, "ok": 0, "rebased": 0, "residual": None}
    modules: dict[str, dict] = {}
    for key in sd:
        if ".adaln_proj.linear." not in key:
            continue
        body, _, suffix = key.rpartition(".adaln_proj.linear.")
        modules.setdefault(body + ".adaln_proj.linear", {})[suffix] = key

    if not modules:
        return sd, stats

    # curve -> curve via the LoRA's own table, when it carries one
    table_map = None
    table_map_failed = source_table is not None and ctx.table is not None
    if source_table is not None and ctx.table is not None:
        try:
            m, a_const, residual = fit_table_to_table(source_table, ctx.table)
            if _accept(residual):
                table_map = (m, a_const)
                stats["residual"] = residual
                table_map_failed = False
                LOG.info("H3 PowerLoraStack: adaLN table-to-table fit residual %.2e", residual)
            else:
                LOG.warning("H3 PowerLoraStack: adaLN table-to-table fit residual %.2e exceeds %.0e",
                            residual, MAX_RESIDUAL)
        except Exception as exc:
            LOG.warning("H3 PowerLoraStack: adaLN table rebase unavailable (%s)", exc)

    out = dict(sd)
    basis = None
    basis_pinv = None
    for module, parts in modules.items():
        a_key = next((parts.get(suffix) for suffix in (
            "lora_A.weight", "lora_A.default.weight", "lora_A",
            "lora_down.weight", "_lora.down.weight", "lora.down.weight",
            "lora_linear_layer.down.weight",
        ) if parts.get(suffix)), None)
        b_key = next((parts.get(suffix) for suffix in (
            "lora_B.weight", "lora_B.default.weight", "lora_B",
            "lora_up.weight", "_lora.up.weight", "lora.up.weight",
            "lora_linear_layer.up.weight",
        ) if parts.get(suffix)), None)
        if a_key is None or b_key is None:
            continue
        a = sd[a_key]
        b = sd[b_key]
        if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
            LOG.warning("H3 PowerLoraStack: malformed adaLN pair %s, skipped", module)
            for key in parts.values():
                out.pop(key, None)
            stats["skipped"] += 1
            continue
        source_dim = a.shape[1]
        a32 = a.to(torch.float32)

        if table_map is not None and source_dim == table_map[0].shape[0]:
            m, a_const = (x.to(device=a32.device) for x in table_map)
            a_new = a32 @ m                            # [r, k_tgt]
            const = a32 @ a_const                      # [r]
            sign = 1.0
            kind = "rebased"
        elif source_dim == ctx.target_dim and not table_map_failed:
            stats["ok"] += 1
            continue
        else:
            if basis is None:
                basis = ctx.basis(curve_table=source_table)
            if basis is None:
                for key in parts.values():
                    out.pop(key, None)
                stats["skipped"] += 1
                continue
            v, c, residual = basis                     # V: [2688, k], c: [2688]
            v = v.to(device=a32.device)
            c = c.to(device=a32.device)
            stats["residual"] = residual
            if source_dim == v.shape[0] and ctx.target_dim == v.shape[1]:
                a_new = a32 @ v                        # [r, k]
                const = a32 @ c                        # [r]
                sign = 1.0
            elif source_dim == v.shape[1] and ctx.target_dim == v.shape[0]:
                if basis_pinv is None:
                    basis_pinv = torch.linalg.pinv(v)
                a_new = a32 @ basis_pinv                # [r, 2688]
                const = a_new @ c                      # [r]
                sign = -1.0
            else:
                LOG.warning(
                    "H3 PowerLoraStack: adaLN pair %s has width %d, expected %d or %d - skipped",
                    module, source_dim, v.shape[0], v.shape[1],
                )
                for key in parts.values():
                    out.pop(key, None)
                stats["skipped"] += 1
                continue
            kind = "ported"

        # comfy scales the A/B pair by alpha/rank inside the adapter but applies
        # a diff_b patch verbatim, so the factor has to be folded in here
        alpha_key = parts.get("alpha")
        scale = 1.0
        if alpha_key is not None:
            try:
                alpha = sd[alpha_key]
                scale = float(alpha.item() if hasattr(alpha, "item") else alpha) / a_new.shape[0]
            except Exception:
                scale = 1.0
        bias_delta = sign * scale * (b.to(torch.float32) @ const)

        out.pop(a_key, None)
        out.pop(b_key, None)
        out[module + ".lora_A.weight"] = a_new.to(a.dtype)
        out[module + ".lora_B.weight"] = b
        if alpha_key is not None and alpha_key != module + ".alpha":
            out.pop(alpha_key, None)
            out[module + ".alpha"] = sd[alpha_key]
        out[module + ".diff_b"] = bias_delta.to(b.dtype)
        stats[kind] += 1

    return out, stats


def read_target(diffusion_model):
    """Inspect a loaded H3 DiT for its adaLN input width and curve table."""
    table = getattr(diffusion_model, "adaln_t_table", None)
    dim = None
    try:
        dim = diffusion_model.blocks[0].adaln_proj.linear.weight.shape[1]
    except Exception:
        if table is not None:
            dim = int(table.shape[1])
    return dim, table
