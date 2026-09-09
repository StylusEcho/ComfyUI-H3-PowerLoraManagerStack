# ComfyUI-H3-PowerLoraManagerStack

One node: **H3 Power LoRA Stack (LoRA Manager)** — stacked multi-LoRA loading
for **MiniMax H3**, with a search bar that pulls its rows straight out of a
[ComfyUI-Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager)
library.

Quantized bases keep an exact runtime branch, AdaLN pairs are rebased between
dense and curve checkpoints, and Acc/PDD head banks are blended per sampler step
instead of crashing the native head.

This is an alternative build of the
[MiniMax H3 Power LoRA Stack](https://github.com/cicalooo/ComfyUI-H3-PowerLoraStack).
It registers under its own node id (`H3PowerLoraManagerStack`), its own web
extension and its own HTTP route, so it can sit next to that pack in the same
ComfyUI install without either one shadowing the other — and it takes that
pack's **adaLN Modality** and **LoRA Schedule** nodes on its own inputs, so
nothing that needed them is lost by there being one node here.

## The node

| Input | |
| --- | --- |
| `model` | The H3 `MODEL` to patch |
| `quantized_layers` | `auto` / `branch` / `merge` |
| `adaln_port` | `auto` / `strip` / `off` |
| `adaln_modality` | Optional `H3_MODALITY`, from **MiniMax H3 adaLN Modality** |
| `schedule` | Optional `H3_SCHEDULE` chain, from **MiniMax H3 LoRA Schedule** |
| `lora_stack` | Optional `LORA_STACK`, e.g. from **Lora Stacker (LoraManager)** |

| Output | |
| --- | --- |
| `MODEL` | Patched model |
| `report` | Per-LoRA account of what was applied |
| `trigger_words` | The library's trained words for the applied LoRAs, `,, `-separated — drops straight into **TriggerWord Toggle (LoraManager)** |

Rows themselves are added in the browser: **🔍 Add LoRA** opens the search bar,
each row gets a toggle, a strength and a remove button, and there is no limit on
how many.

Drag a row by the grip on its left to reorder it — or right-click the row for
**Move up / down / to top / to bottom**. Order is not cosmetic: it is the order
the LoRAs are applied, and it is what an H3 LoRA Schedule's `1,3` and `2-4` row
selectors address. A row keeps its strength and its auto-balance measurement
when it moves.

`adaln_modality` and `schedule` are the outputs of two nodes in the original
pack. ComfyUI matches links by type name, so installing that pack next to this
one is all the wiring there is. Left unconnected they simply do nothing: adaLN
runs at full strength for every modality and rows run at their static strength.

## The LoRA Manager connection

With ComfyUI-Lora-Manager installed, the search bar queries the library live:

- **Search** across file name, the creator's model name and tags, fuzzy, so
  `turbo` finds a LoRA whose file is named after its hash.
- **Filter** by base model, by folder, or to your favourites. A folder includes
  everything under it, so `styles` also finds `styles/anime`. The three stick
  between openings — they are remembered in the browser rather than in the
  workflow, so a workflow you share does not carry your library's folder names
  to someone whose library has no such folder.
- Results carry the manager's **preview, model name, version and base model**,
  and picking one adopts the **strength preset** saved on that model.
- **Hover** a result — or a row on the node — for the full-size preview.
- The row keeps the **trigger words** the manager holds, which is what feeds the
  `trigger_words` output.
- The footer links through to the manager's own library page.

Nothing imports the manager's Python package — its module layout moves between
releases. The browser talks to its HTTP API (`/api/lm/…`) on ComfyUI's own
aiohttp app, and the Python side reads the `<model>.metadata.json` sidecar the
manager writes next to each file, which is what recovers trigger words for a row
restored from a saved workflow.

Without the manager the search bar falls back to filtering ComfyUI's own
`models/loras` listing and the node works as an ordinary stack; the footer
button switches between the two sources by hand.

A row picked from the library also stores the absolute path the manager reported
for it. `models/loras` is still tried first, so a workflow keeps resolving
through ComfyUI's listing when the manager is not running — the stored path is
the fallback, and is what lets a library configured under `extra_loras_roots`,
outside `models/loras` entirely, load at all.

## Good to know

- **AdaLN port — 8 Aug 2026, commit `28ac439`.** This stack rebases dense↔curve
  AdaLN LoRA pairs so ComfyUI does not skip them with
  `ERROR lora ... adaln_proj.linear.weight shape '[96768, 8]' is invalid for input of size 260112384`.
  The port keeps rank and restores the DC term as a bias delta. A separate
  AdaLN-fix node on the same `MODEL` is not needed; if one is already attached
  the report tells you to disable it (`adaln_port=off` only if you mean to keep
  that other node).
- **`quantized_layers=auto`** branches quantized weights and merges the rest.
  Stock dequantize/requantize merge destroys small H3 LoRAs.
- **Strength 1.0 is not a unit.** Auto-balance only ever trims outliers; it
  never boosts a quiet (e.g. turbo) LoRA.
- Empty stack rows still run the incoming AdaLN pass, so the node can sit after
  another loader as a fix.
- Wire the `report` output to a show-text node (it is also logged).

<details>
<summary>Quantized weights</summary>

ComfyUI's stock path is dequantize → add delta →
`requantize_from_float(scale="recalculate")`. That round trip is not
idempotent: re-fitting the codebook and re-rounding to int4 injects ~1.5%
relative weight noise, while a typical H3 LoRA delta is 0.01–0.08% of the
weight. On a w4a8 checkpoint the stock merge recovers under a third of the
LoRA magnitude (cos 0.12–0.14) and takes 128 s versus 6.5 s.

This node routes quantized layers through an exact runtime low-rank branch
(`y = W_q(x) + B @ A @ x`), keeping the quantized kernel (~1.5% extra FLOPs at
rank 64).

`quantized_layers`:

- `auto` (default) — branch quantized layers, merge unquantized ones
- `branch` — never modify a weight, even in bf16; adapters without a runtime
  branch are reported as rejected
- `merge` — stock behaviour; only useful for comparison

`mlp.fc2` under `TensorWiseINT8Layout` is reached through
`comfy.ops.linear_input_act`, which never calls `fc2.forward`. A branch there
would be dropped, so those layers merge even in `branch` mode.

</details>

<details>
<summary>AdaLN basis (dense ↔ curve)</summary>

H3 ships in two forms. A *dense* checkpoint feeds `silu(time_embedder(t))`
(2688-wide) into every `adaln_proj.linear`. A *curve* (pruned) checkpoint
stores `adaln_t_table` of shape `[grid, 8]` and drops the time embedder.

A LoRA trained on one form has the wrong `lora_A` width for the other. ComfyUI
logs the shape error above and skips the layer. Dropping those pairs is not an
acceptable fix: on turbo distillation LoRAs the constant term alone is ~100% of
`dW @ S(t)`.

This node changes basis and preserves rank:

```
dense -> curve   A' = A @ V,       bias delta  B @ (A @ c)
curve -> dense   A' = A @ pinv(V), bias delta -B @ (A' @ c)
```

`S(t) = c + V @ table(t)` is recovered by least squares of `[1 | table]`
against the silu grid. The bias is emitted as `.diff_b`. End-to-end, the
ported adapter matches the dense contribution at **cos 0.999998**.

The fit uses the *target checkpoint's own* table. Bakes differ: one local bake
is uncentered (column norms 22.98, 2.67, 1.66, …), another is mean-centered
(7.08, 2.09, 0.75, …).

Two curve bakes can share AdaLN width, so a LoRA trained on one loads on the
other without complaint and is simply wrong (measured **cos −0.375**). When a
LoRA ships `adaln_t_table`, this node rebases table-to-table (cos 1.000000, no
grid). When it does not, prefer LoRAs trained against the checkpoint you run.

Incoming patches on `MODEL` are scanned by shape and rebased the same way,
still as a rank decomposition plus `.diff_b`. `adaln_port`: `auto` (default),
`strip` (drop mismatched pairs), or `off`.

**Grid sources.** Prefer a basis baked into the checkpoint (`adaln_basis` +
`adaln_mean`, or `silu_t_emb_grid`). Else a live `time_embedder` on a dense
base, then `h3_silu_temb_grid.safetensors` (`models/h3_adaln/`, `models/loras/`,
`models/diffusion_models/`, one level under `custom_nodes/`), then a scored
scan of other H3 checkpoints. Table-to-table needs no grid. A grid from the
wrong build bottoms out at ~1.7e-3; fits worse than 5e-3 are rejected.

</details>

<details>
<summary>Key conventions</summary>

Keys resolve against the model's own state dict, not by guessing underscore
splits (`qkv_proj` is never two tokens):

| Convention | Example |
| --- | --- |
| ai-toolkit / diffusers | `diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight` |
| bare (no prefix) | `blocks.0.attn.qkv_proj.lora_A.weight` |
| kohya / musubi | `lora_unet_blocks_0_attn_qkv_proj.lora_down.weight` + `.alpha` |
| lycoris | `lycoris_blocks_0_...` |
| peft / diffusers trainer | `base_model.model.blocks.0...`, `transformer.blocks.0...` |

Unmatched keys are reported per row.

</details>

<details>
<summary>Acc / PDD output heads</summary>

alibaba-pai Acc LoRAs ship a 32-interval output-head bank as
`final_layer.video_out.set_weight` of shape `[3072, 5376]` (`32 × 96`) plus the
audio twin. Stock Comfy `copy_`s that onto the native `[96, 5376]` head and
crashes. This node peels those tensors before the stock `set` path and blends
the heads the sampler step spans (same rule as Comfy PR 15908). Row strength
interpolates between native and blended heads. Use `simple` at 8 steps with
shifts 12/3 so steps land on the trained grid. Later stack rows that also carry
a bank replace the earlier one.

</details>

<details>
<summary>Stacking</summary>

Multiple LoRAs on the same layer fuse into one pair by concatenating along the
rank axis:

```
sum_i s_i * B_i @ A_i @ x  ==  [s_1 B_1 | ... | s_N B_N] @ [A_1; ...; A_N] @ x
```

A ten-LoRA stack costs one extra matmul pair per layer, not ten. Factors live
in a bank registered via `set_additional_models` so VRAM is accounted for.

Entries arriving on `lora_stack` are applied ahead of this node's own rows, so a
**Lora Stacker (LoraManager)** chain can feed straight in. CLIP strengths on
those entries are ignored rather than folded into the model strength — H3 has no
CLIP tower on this path.

</details>

<details>
<summary>Denoising schedules</summary>

Wire the original pack's **MiniMax H3 LoRA Schedule** into `schedule`. Select
rows with `all`, `1,3`, or `2-4`, then a linear, cosine, smoothstep, power, step
or explicit curve. `start_percent` / `end_percent` limit the transition. Chain
schedule nodes; the later node wins where selectors overlap.

`steps` follows model-call indices. `sigma` follows the scheduler's normalized
noise. The stack reads `sample_sigmas` from ComfyUI; plain KSampler works, no
SIGMAS wire.

Row numbers are this node's own rows, counted top to bottom. Entries arriving
on `lora_stack` are not rows you can see or number, so they are reachable by
`all` and by nothing else.

Scheduled rows always use the live branch path, including on unquantized
bases. Anything that cannot branch is merged at the row's static strength and
called out in the report. Ported AdaLN bias deltas follow their LoRA rather
than staying fixed.

Schedule links cross the pack boundary by shape, not by class: ComfyUI imports
each pack under its own name, so the other pack's `Schedule` is a different
class built from the same source. Links are recognised by carrying `matches`,
which is what lets that node drive these rows.

</details>

<details>
<summary>Auto-balance</summary>

In the reference non-distillation H3 corpus, perturbation at strength 1.0 spans
**65×** (0.054% of base weights vs 5.24%). Rank and file size do not predict
it. `⚖ Auto-balance strengths` measures each active LoRA and puts them on one
scale:

```
rel = sqrt( sum_l ||dW_l||_F^2 / sum_l ||W_l||_F^2 )
```

The factor multiplies the strength you already chose — or the preset the LoRA
Manager holds for that model — so relative intent between rows survives. It is
clamped to ≤ 1 (trim only). Distillation adapters are quiet on purpose and stay
at ×1.00.

`dW` is never formed. `||B A||_F^2 = tr((B^T B)(A A^T))` uses r×r matrices.
Results cache on (path, mtime, size). LoKr: `||W1 ⊗ W2||_F = ||W1||_F · ||W2||_F`.

The reference is the collection *median*. AdaLN is excluded from the
measurement (basis is checkpoint-dependent; distillation keeps the schedule
change there). `↺ Restore manual strengths` puts every row back; editing a
strength by hand is not clobbered by a later recompute.

Frobenius energy is not perceptual strength, and it says nothing about
contention. Distinct LoRAs are near-orthogonal in weight space
(|cos| ≤ 0.03) yet overlap 3–15× above chance in feature subspaces. Stack
energy adds in quadrature (`1/√N`, not `1/N`); auto-balance does **not** apply
that — adding a row never silently weakens the others.

</details>

<details>
<summary>AdaLN modality control</summary>

H3 packs audio and video through the same 50 blocks. The only modality-specific
weights are four tensors (`video_patch_proj`, `audio_patch_proj`,
`final_layer.video_out`, `final_layer.audio_out`), and typical H3 LoRAs touch
none of them.

AdaLN does split: `AdalnProj` emits three contiguous blocks of 32256 rows
(`{video: 0, text: 1, audio: 2}`). Scaling a slice of `lora_B` scales that
modality's modulation with no runtime hook.

Wire the original pack's **MiniMax H3 adaLN Modality** node into
`adaln_modality` and it scales every stacked LoRA at once. All three at 1.0 is a
no-op; 0.0 removes that modality's share. Scaling runs *before* AdaLN porting so
`.diff_b` inherits it. Geometry is read off the loaded `AdalnProj`;
`final_layer.adaln_proj` is one-modality and is left alone.

Where AdaLN is present it is not a marginal knob: 89–99.7% of weight-space
perturbation for content LoRAs (median ~96%), 16–23% for curve8 turbo
adapters. Read that as where most of the weight change lives, not as 96% of
what you see.

</details>

<details>
<summary>Report output</summary>

```
base: ConvRotW4A4 x300, INT8 x50
adaLN: curve (input dim 8)
turbo dense-adaLN @ 1: 26 merged, 275 branched, adaLN ported x51 (basis fit 1.7e-03)
  rel dW 0.047%
style_lora curve-adaLN @ 0.8: 0 merged, 258 branched, adaLN modality video/text/audio = 1/1/0.25 x50
  rel dW 0.054%
motion_lora @ 1: 0 merged, 104 branched, adaLN modality 1/1/0.25 INACTIVE (LoRA has no adaLN pairs)
  rel dW 0.316%
branch bank: 300 layers, 412 MB
```

`rel dW` is reported for every LoRA whether or not auto-balance ran. When the
applied strength is far from the calibrated one, the line says what
auto-balance would have used.

</details>

<details>
<summary>Limitations</summary>

- A curve-trained LoRA on a **dense** checkpoint can only be ported if the LoRA
  carries its own `adaln_t_table`; otherwise those AdaLN pairs are dropped with
  a warning.
- Runtime branches apply to `MODEL` from ComfyUI's native H3 loader. The
  streaming loader in `minimaxh3chinkloader` uses its own handle and LoRA path.
- DoRA, LoHa, LoKr and locon merge in `auto`; `branch` rejects them (only plain
  rank decompositions have the runtime branch path).
- `adaln_modality` and `schedule` have nothing to connect to unless the original
  Power LoRA Stack pack is installed, so per-modality adaLN scaling and per-row
  schedules need that pack; the sockets are simply unused until then.

</details>

<details>
<summary>Development checks</summary>

From the ComfyUI root:

```powershell
$env:PYTHONPATH = "custom_nodes/ComfyUI-H3-PowerLoraManagerStack;."
python -m pytest custom_nodes/ComfyUI-H3-PowerLoraManagerStack/tests --import-mode=importlib -q
```

`python -m unittest discover -s tests -t tests` works too, from the pack root.
CPU paths are covered; the mixed-device check skips without CUDA.

The frontend has a syntax gate and a small suite for the logic that does not
need a canvas — where a dragged row lands, and that a picker filter survives a
round trip through browser storage:

```
node --check web/h3_power_lora_manager_stack.js
node --test tests/js/*.test.mjs
```

Both run on plain Node with no dependencies. Drawing, dragging and hovering
need a real browser and are not covered.

Requires Python 3.10+ and a ComfyUI revision with MiniMax H3,
`QuantizedTensor`, weight adapters, and patcher wrappers. ComfyUI-Lora-Manager
is optional.

</details>

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
