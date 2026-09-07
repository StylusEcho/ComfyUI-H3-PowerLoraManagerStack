import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "H3PowerLoraManagerStack";
const ROW_TYPE = "H3LM_LORA";
const ROW_HEIGHT = 26;
const MARGIN = 12;

const BALANCE_ROUTE = "/h3_power_lora_manager_stack/balance";
const BALANCE_LABEL = "⚖ Auto-balance strengths";
const RESTORE_LABEL = "↺ Restore manual strengths";

/* ------------------------------------------------------- LoRA Manager API -- */

// willmiao/ComfyUI-Lora-Manager registers these on ComfyUI's own aiohttp app,
// so they are same-origin and need no configuration.  Nothing here imports the
// manager: the HTTP API is its stable surface, its Python internals are not.
//
// The paths are written without the `/api` the manager documents, because
// `api.fetchApi` prepends it -- along with ComfyUI's `api_base`, which is what
// keeps this working when ComfyUI is served under a sub-path.
const LM = {
  health: "/lm/health-check",
  list: "/lm/loras/list",
  baseModels: "/lm/loras/base-models",
  page: "/loras",
};

/** A non-API URL (a page, or a preview the manager handed us) under api_base. */
function siteUrl(path) {
  return typeof api.fileURL === "function" ? api.fileURL(path) : path;
}

// Availability is re-probed rather than cached forever: the manager finishes
// its initial library scan some seconds after ComfyUI is up, and a user can
// install it without reloading the graph.
const HEALTH_TTL_OK = 300_000;
const HEALTH_TTL_FAIL = 20_000;
let healthState = { ok: false, at: 0 };

async function managerAvailable() {
  const ttl = healthState.ok ? HEALTH_TTL_OK : HEALTH_TTL_FAIL;
  if (healthState.at && Date.now() - healthState.at < ttl) return healthState.ok;
  let ok = false;
  try {
    const res = await api.fetchApi(LM.health);
    ok = res.ok && (await res.json())?.status === "ok";
  } catch (err) {
    ok = false;
  }
  healthState = { ok, at: Date.now() };
  return ok;
}

/**
 * Search the LoRA Manager library.
 *
 * `fuzzy_search` is what makes a partial word match the way the manager's own
 * search box does; the three `search_*` flags widen a query across the file
 * name, the creator's model name and the tags, so "turbo" finds a LoRA whose
 * file is named after its hash.
 */
async function managerSearch(query, { baseModel = "", favorites = false, limit = 60,
                                      signal } = {}) {
  const params = new URLSearchParams({
    page: "1",
    page_size: String(limit),
    sort_by: "name",
    fuzzy_search: "true",
    search_filename: "true",
    search_modelname: "true",
    search_tags: "true",
  });
  if (query) params.set("search", query);
  if (baseModel) params.set("base_model", baseModel);
  if (favorites) params.set("favorites_only", "true");
  const res = await api.fetchApi(`${LM.list}?${params}`, { signal });
  if (!res.ok) throw new Error(`LoRA Manager search failed (${res.status})`);
  const data = await res.json();
  return { items: data?.items ?? [], total: data?.total ?? 0 };
}

async function managerBaseModels() {
  try {
    const res = await api.fetchApi(`${LM.baseModels}?limit=100`);
    if (!res.ok) return [];
    const data = await res.json();
    // The manager returns either bare strings or {name, count} objects
    // depending on version; both are reduced to a name here.
    return (data?.base_models ?? [])
      .map((entry) => (typeof entry === "string" ? entry : entry?.name))
      .filter(Boolean);
  } catch (err) {
    return [];
  }
}

/**
 * A manager item as this node stores it.
 *
 * `file_name` is the manager's display name and carries no extension, so the
 * name ComfyUI can resolve is rebuilt from the real basename of `file_path`
 * under the library-relative `folder`.  `lmPath` keeps the absolute path the
 * manager reported, which is the only handle on a library configured outside
 * `models/loras`.
 */
function rowFromManagerItem(item) {
  const full = String(item.file_path || "").replace(/\\/g, "/");
  const base = full.split("/").pop() || `${item.file_name}.safetensors`;
  const folder = String(item.folder || "").replace(/\\/g, "/").replace(/^\/+|\/+$/g, "");
  // No ``on``: a row being repointed keeps whatever enable state it had.
  const row = {
    lora: folder ? `${folder}/${base}` : base,
    strength: managerPresetStrength(item),
    lmPath: full || undefined,
  };
  const words = item?.civitai?.trainedWords;
  if (Array.isArray(words) && words.length) {
    row.triggerWords = words.filter((w) => typeof w === "string");
  }
  if (item.base_model) row.baseModel = String(item.base_model);
  // Stored as the manager gave it: `api_base` is a deployment detail and
  // has no business being baked into a saved workflow.
  if (item.preview_url) row.previewUrl = String(item.preview_url);
  return row;
}

/** The strength preset saved on the model in the manager, else 1.0. */
function managerPresetStrength(item) {
  let tips = item?.usage_tips;
  if (typeof tips === "string") {
    try {
      tips = JSON.parse(tips || "{}");
    } catch (err) {
      tips = null;
    }
  }
  const value = Number(tips?.strength);
  return Number.isFinite(value) ? round2(value) : 1.0;
}

const VIDEO_PREVIEW = /\.(mp4|webm|mov|m4v)(\?|$)/i;

function isVideoPreview(url) {
  try {
    return VIDEO_PREVIEW.test(decodeURIComponent(url));
  } catch (err) {
    return VIDEO_PREVIEW.test(url);
  }
}

/* ----------------------------------------------------------- local loras -- */

/**
 * The stack node declares no ``lora_name`` input of its own -- rows are added
 * in the browser -- so the current folder listing is borrowed from a node that
 * does.  This is the fallback for when the LoRA Manager is not installed.
 */
function fetchLoraList() {
  return api
    .fetchApi("/object_info/LoraLoaderModelOnly")
    .then((res) => res.json())
    .then(
      (info) => info?.LoraLoaderModelOnly?.input?.required?.lora_name?.[0] ?? []
    )
    .catch((err) => {
      console.warn("[H3PowerLoraManagerStack] could not fetch lora list", err);
      return [];
    });
}

/**
 * Ask the server what each LoRA actually does to the weights.
 *
 * "Strength 1.0" is not a unit -- across the local H3 collection the
 * perturbation it produces spans 65x -- so the server measures each file and
 * returns the multiplier that puts it on the same scale as the rest.  Results
 * are cached server-side on (path, mtime, size), so pressing the button again
 * costs one round trip and no disk.  Rows carry their manager path so a library
 * outside ``models/loras`` can still be measured.
 */
async function fetchBalance(rows) {
  const res = await api.fetchApi(BALANCE_ROUTE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ loras: rows }),
  });
  if (!res.ok) throw new Error(`balance request failed (${res.status})`);
  return res.json();
}

/* ---------------------------------------------------------- row painting -- */

// Preview thumbnails are drawn straight onto the canvas, so each URL is loaded
// once and kept; a failed load is remembered as `null` so a broken preview does
// not re-request on every repaint.
const thumbnails = new Map();

function thumbnail(url, node) {
  if (!url || isVideoPreview(url)) return null;
  const src = siteUrl(url);
  if (thumbnails.has(src)) return thumbnails.get(src);
  const image = new Image();
  thumbnails.set(src, image);
  image.onload = () => node?.setDirtyCanvas(true, true);
  image.onerror = () => thumbnails.set(src, null);
  image.src = src;
  return image;
}

function drawToggle(ctx, x, y, size, on) {
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(x, y + 3, size * 1.7, size - 6, (size - 6) / 2);
  ctx.fillStyle = on ? "#3d8c5a" : "#3a3a3a";
  ctx.fill();
  ctx.beginPath();
  const knob = (size - 8) / 2;
  ctx.arc(x + (on ? size * 1.7 - knob - 3 : knob + 3), y + size / 2, knob, 0, Math.PI * 2);
  ctx.fillStyle = on ? "#d8f5e4" : "#8a8a8a";
  ctx.fill();
  ctx.restore();
}

function drawThumb(ctx, image, x, y, size) {
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(x, y, size, size, 3);
  ctx.clip();
  // cover-fit: previews are portrait more often than not, and letterboxing a
  // 20px tile leaves nothing recognisable
  const scale = Math.max(size / image.naturalWidth, size / image.naturalHeight);
  const w = image.naturalWidth * scale;
  const h = image.naturalHeight * scale;
  ctx.drawImage(image, x + (size - w) / 2, y + (size - h) / 2, w, h);
  ctx.restore();
}

function shortName(name) {
  if (!name || name === "None") return "click to choose";
  const base = String(name).replace(/\\/g, "/").split("/").pop();
  return base.replace(/\.(safetensors|ckpt|pt|bin)$/i, "");
}

/**
 * One stack row: enable toggle, preview, lora name, strength, remove button.
 * Values serialize as {on, lora, strength, ...}, which is what the Python
 * side's flexible `lora_N` inputs expect.
 */
function makeLoraWidget(node, name, value) {
  const widget = {
    name,
    type: ROW_TYPE,
    value: Object.assign({ on: true, lora: "None", strength: 1.0 }, value || {}),
    options: { serialize: true },
    hitAreas: {},

    computeSize() {
      return [node.size[0], ROW_HEIGHT];
    },

    draw(ctx, node_, widgetWidth, y, height) {
      const margin = MARGIN;
      const left = margin;
      const right = widgetWidth - margin;
      const midY = y + height / 2;
      ctx.save();

      ctx.beginPath();
      ctx.roundRect(left, y + 1, right - left, height - 2, 6);
      ctx.fillStyle = this.value.on ? "#2b2b2b" : "#232323";
      ctx.fill();

      let cursor = left + 8;
      drawToggle(ctx, cursor, y, height, this.value.on);
      this.hitAreas.toggle = [cursor, cursor + height * 1.7];
      cursor += height * 1.7 + 8;

      // the manager's own preview, when the row came from the library
      const image = thumbnail(this.value.previewUrl, node_);
      if (image && image.complete && image.naturalWidth) {
        const size = height - 8;
        ctx.globalAlpha = this.value.on ? 1 : 0.45;
        drawThumb(ctx, image, cursor, y + 4, size);
        ctx.globalAlpha = 1;
        cursor += size + 7;
      }

      // remove button, laid out from the right edge inward
      const removeX = right - 20;
      ctx.fillStyle = "#8a8a8a";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("✕", removeX + 6, midY);
      this.hitAreas.remove = [removeX, removeX + 16];

      const strengthRight = removeX - 10;
      const strengthLeft = strengthRight - 74;
      ctx.fillStyle = "#1e1e1e";
      ctx.beginPath();
      ctx.roundRect(strengthLeft, y + 4, strengthRight - strengthLeft, height - 8, 4);
      ctx.fill();
      ctx.fillStyle = "#b8b8b8";
      ctx.font = "11px sans-serif";
      ctx.fillText("◀", strengthLeft + 9, midY);
      ctx.fillText("▶", strengthRight - 9, midY);
      ctx.fillStyle = this.value.on ? "#e8e8e8" : "#777";
      ctx.font = "12px sans-serif";
      ctx.fillText(
        Number(this.value.strength).toFixed(2),
        (strengthLeft + strengthRight) / 2,
        midY
      );
      this.hitAreas.strengthDown = [strengthLeft, strengthLeft + 18];
      this.hitAreas.strengthUp = [strengthRight - 18, strengthRight];
      this.hitAreas.strengthValue = [strengthLeft + 18, strengthRight - 18];

      // auto-balance badge: what the measurement did to this row.  Amber flags
      // a note from the server (a duplicate adapter, or an unreadable file).
      let badgeLeft = strengthLeft - 10;
      if (this.value.autoApplied && this.value.factor !== undefined) {
        const badge = `⚖×${Number(this.value.factor).toFixed(2)}`;
        ctx.font = "10px sans-serif";
        ctx.textAlign = "right";
        ctx.fillStyle = this.value.note
          ? "#d99a4e"
          : this.value.factor < 0.995
            ? "#7fd6a0"
            : "#6a6a6a";
        ctx.fillText(badge, strengthLeft - 8, midY);
        badgeLeft = strengthLeft - 14 - ctx.measureText(badge).width;
      }

      const nameLeft = cursor;
      const nameRight = badgeLeft;
      ctx.textAlign = "left";
      ctx.fillStyle = this.value.on ? "#e8e8e8" : "#777";
      ctx.font = "12px sans-serif";
      const label = shortName(this.value.lora);
      let text = label;
      const maxWidth = nameRight - nameLeft;
      if (ctx.measureText(text).width > maxWidth) {
        while (text.length > 4 && ctx.measureText(text + "…").width > maxWidth) {
          text = text.slice(0, -1);
        }
        text += "…";
      }
      ctx.fillText(text, nameLeft, midY);
      this.hitAreas.name = [nameLeft, nameRight];

      ctx.restore();
    },

    mouse(event, pos, node_) {
      if (event.type !== "pointerdown") return false;
      const x = pos[0];
      const hit = (area) => area && x >= area[0] && x <= area[1];

      if (hit(this.hitAreas.toggle)) {
        this.value = { ...this.value, on: !this.value.on };
        node_.setDirtyCanvas(true, true);
        // a row switched on inside a balanced stack has not been measured yet
        if (this.value.on && isBalanced(node_)) applyBalance(node_);
        return true;
      }
      if (hit(this.hitAreas.remove)) {
        removeLoraWidget(node_, this);
        return true;
      }
      if (hit(this.hitAreas.strengthDown)) {
        this.setStrength(node_, round2(this.value.strength - 0.05));
        return true;
      }
      if (hit(this.hitAreas.strengthUp)) {
        this.setStrength(node_, round2(this.value.strength + 0.05));
        return true;
      }
      if (hit(this.hitAreas.strengthValue)) {
        app.canvas.prompt(
          "Strength",
          this.value.strength,
          (v) => {
            const parsed = parseFloat(v);
            if (!Number.isNaN(parsed)) this.setStrength(node_, parsed);
          },
          event
        );
        return true;
      }
      if (hit(this.hitAreas.name)) {
        showLoraMenu(node_, this, event);
        return true;
      }
      return false;
    },

    /**
     * Editing a strength by hand overrides the balance for this row: the badge
     * clears and a later recompute leaves it alone, but the stashed manual
     * value stays put so Restore still returns to what was there before.
     */
    setStrength(node_, value) {
      this.value = { ...this.value, strength: value, autoApplied: false };
      node_.setDirtyCanvas(true, true);
    },

    serializeValue() {
      return { ...this.value };
    },
  };
  return widget;
}

function round2(v) {
  return Math.round(v * 100) / 100;
}

/* ---------------------------------------------------------------- picker -- */

const PICKER_CSS = `
.h3lm-picker{position:fixed;z-index:10000;display:flex;flex-direction:column;
  width:520px;max-width:94vw;background:#1e1e1e;border:1px solid #4a4a4a;
  border-radius:8px;box-shadow:0 12px 32px rgba(0,0,0,.6);overflow:hidden;
  font-family:system-ui,-apple-system,sans-serif;}
.h3lm-picker input[type=text]{margin:8px 8px 6px;padding:7px 9px;background:#111;
  color:#eee;border:1px solid #555;border-radius:5px;font-size:13px;outline:none;}
.h3lm-picker input[type=text]:focus{border-color:#3d8c5a;}
.h3lm-bar{display:flex;align-items:center;gap:8px;padding:0 12px 7px;
  font-size:11px;color:#888;}
.h3lm-bar select{background:#161616;color:#ccc;border:1px solid #444;
  border-radius:4px;font-size:11px;padding:2px 4px;max-width:150px;}
.h3lm-bar label{display:flex;align-items:center;gap:4px;cursor:pointer;
  user-select:none;}
.h3lm-count{margin-left:auto;white-space:nowrap;}
.h3lm-list{overflow-y:auto;max-height:52vh;}
.h3lm-item{display:flex;align-items:center;gap:9px;padding:5px 12px;
  font-size:12.5px;color:#ddd;cursor:pointer;}
.h3lm-item:hover{background:#2c2c2c;}
.h3lm-item.sel{background:#31543f;}
.h3lm-item.cur .h3lm-name::after{content:" · current";color:#666;font-size:10px;}
.h3lm-thumb{flex:0 0 auto;width:34px;height:34px;border-radius:4px;
  object-fit:cover;background:#2a2a2a;border:1px solid #333;}
.h3lm-text{min-width:0;flex:1 1 auto;}
.h3lm-name{display:block;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;}
.h3lm-sub{display:block;font-size:10.5px;color:#7a7a7a;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis;}
.h3lm-tag{flex:0 0 auto;font-size:10px;color:#8fb9a1;background:#22352a;
  border-radius:3px;padding:1px 5px;}
.h3lm-fav{flex:0 0 auto;font-size:11px;color:#d9b44e;}
.h3lm-item .dir{color:#7a7a7a;}
.h3lm-item .hit{color:#7fd6a0;font-weight:600;}
.h3lm-empty{padding:14px 12px;font-size:12px;color:#888;}
.h3lm-foot{display:flex;align-items:center;gap:8px;padding:6px 12px;
  border-top:1px solid #333;font-size:10.5px;color:#777;}
.h3lm-foot a{color:#7fd6a0;text-decoration:none;margin-left:auto;}
.h3lm-foot a:hover{text-decoration:underline;}
.h3lm-mode{background:none;border:1px solid #444;border-radius:4px;color:#aaa;
  font-size:10.5px;padding:1px 6px;cursor:pointer;font-family:inherit;}
.h3lm-mode:hover{border-color:#666;color:#ddd;}
`;

function ensurePickerCss() {
  if (document.getElementById("h3lm-css")) return;
  const style = document.createElement("style");
  style.id = "h3lm-css";
  style.textContent = PICKER_CSS;
  document.head.appendChild(style);
}

/** Case-insensitive AND-match of every whitespace-separated term. */
function matchTerms(name, terms) {
  const hay = name.toLowerCase();
  const spans = [];
  for (const term of terms) {
    const at = hay.indexOf(term);
    if (at < 0) return null;
    spans.push([at, at + term.length]);
  }
  return spans;
}

/** Basename hits and early hits rank first; then shortest name. */
function score(name, terms, spans) {
  if (!terms.length) return 0;
  const slash = name.lastIndexOf("/") + 1;
  let s = 0;
  for (const [start] of spans) {
    if (start >= slash) s -= 1000;
    s += start;
  }
  return s + name.length * 0.01;
}

function highlight(name, spans) {
  const slash = name.lastIndexOf("/") + 1;
  const merged = [];
  for (const [a, b] of [...spans].sort((p, q) => p[0] - q[0])) {
    const last = merged[merged.length - 1];
    if (last && a <= last[1]) last[1] = Math.max(last[1], b);
    else merged.push([a, b]);
  }
  let html = "";
  let at = 0;
  for (const [a, b] of merged) {
    html += wrapDir(escapeHtml(name.slice(at, a)), at, slash);
    html += `<span class="hit">${escapeHtml(name.slice(a, b))}</span>`;
    at = b;
  }
  html += wrapDir(escapeHtml(name.slice(at)), at, slash);
  return html;
}

function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function wrapDir(text, offset, slash) {
  if (!text) return "";
  if (offset + text.length <= slash) return `<span class="dir">${text}</span>`;
  if (offset >= slash) return text;
  const cut = slash - offset;
  return `<span class="dir">${text.slice(0, cut)}</span>${text.slice(cut)}`;
}

const SEARCH_DEBOUNCE = 180;

/**
 * The search bar.
 *
 * With ComfyUI-Lora-Manager installed this queries the library live -- names,
 * previews, base models, favourites and the strength preset saved on each
 * model -- and adds what you pick as a stack row.  Without it, it falls back to
 * filtering ComfyUI's own `models/loras` listing, so the node stays usable on
 * its own.  The two modes share the keyboard: type to narrow, arrows to move,
 * Enter to take it.
 */
async function showLoraMenu(node, widget, event, { removeOnCancel = false } = {}) {
  ensurePickerCss();

  const current = widget.value.lora;
  let managerMode = await managerAvailable();
  let localNames = null;          // lazily fetched, only in fallback mode
  let baseModels = null;

  const root = document.createElement("div");
  root.className = "h3lm-picker";
  const input = document.createElement("input");
  input.type = "text";
  input.spellcheck = false;
  const bar = document.createElement("div");
  bar.className = "h3lm-bar";
  const baseSelect = document.createElement("select");
  const favLabel = document.createElement("label");
  const favBox = document.createElement("input");
  favBox.type = "checkbox";
  favLabel.append(favBox, document.createTextNode("favourites"));
  const count = document.createElement("div");
  count.className = "h3lm-count";
  const list = document.createElement("div");
  list.className = "h3lm-list";
  const foot = document.createElement("div");
  foot.className = "h3lm-foot";
  const modeButton = document.createElement("button");
  modeButton.type = "button";
  modeButton.className = "h3lm-mode";
  const link = document.createElement("a");
  link.href = siteUrl(LM.page);
  link.target = "_blank";
  link.rel = "noreferrer";
  link.textContent = "open library ↗";
  foot.append(modeButton, link);
  root.append(input, bar, list, foot);
  document.body.appendChild(root);

  // place near the click, clamped into the viewport
  const px = event?.clientX ?? window.innerWidth / 2;
  const py = event?.clientY ?? window.innerHeight / 2;
  root.style.left = `${Math.max(8, Math.min(px, window.innerWidth - 530))}px`;
  root.style.top = `${Math.max(8, Math.min(py, window.innerHeight - 360))}px`;

  let rows = [];
  let sel = 0;
  let committed = false;
  let pending = null;             // debounce timer
  let inflight = null;            // AbortController for the live query
  let generation = 0;             // drops results that arrive out of order

  const close = () => {
    document.removeEventListener("pointerdown", onOutside, true);
    clearTimeout(pending);
    inflight?.abort();
    root.remove();
    // a row opened straight from "Add LoRA" and then dismissed was never
    // wanted, so take it back out rather than leaving an empty slot behind
    if (removeOnCancel && !committed) removeLoraWidget(node, widget);
  };

  const commit = (patch) => {
    const wasBalanced = isBalanced(node);
    committed = true;
    const next = { ...widget.value, ...patch };
    // the balance factor belonged to the file that was here before, so drop the
    // whole measurement and hand the row back its strength; clearing the stash
    // too is what marks it for re-measuring
    if (next.manual !== undefined || next.autoApplied) {
      if (patch.strength === undefined) {
        next.strength = next.manual !== undefined ? next.manual : next.strength;
      }
      delete next.manual;
      delete next.factor;
      delete next.rel;
      delete next.note;
      delete next.autoApplied;
    }
    // fields belonging to whatever used to be in this row
    for (const key of ["lmPath", "triggerWords", "baseModel", "previewUrl"]) {
      if (!(key in patch)) delete next[key];
    }
    widget.value = next;
    node.setDirtyCanvas(true, true);
    close();
    if (wasBalanced) applyBalance(node);
  };

  const onOutside = (e) => {
    if (!root.contains(e.target)) close();
  };

  /* ---- rendering ---- */

  const renderRows = (total) => {
    count.textContent = total === null ? "" : `${rows.length} of ${total}`;
    list.replaceChildren();
    if (!rows.length) {
      const empty = document.createElement("div");
      empty.className = "h3lm-empty";
      empty.textContent = managerMode
        ? "Nothing in the LoRA Manager library matches that."
        : "No LoRA matches that filter.";
      list.appendChild(empty);
      return;
    }
    rows.forEach((row, i) => {
      const item = document.createElement("div");
      item.className =
        "h3lm-item" + (i === sel ? " sel" : "") + (row.name === current ? " cur" : "");
      item.title = row.title ?? row.name;
      if (row.previewUrl && !isVideoPreview(row.previewUrl)) {
        const img = document.createElement("img");
        img.className = "h3lm-thumb";
        img.loading = "lazy";
        img.src = siteUrl(row.previewUrl);
        img.onerror = () => img.remove();
        item.appendChild(img);
      }
      const text = document.createElement("div");
      text.className = "h3lm-text";
      const nameEl = document.createElement("span");
      nameEl.className = "h3lm-name";
      if (row.spans) nameEl.innerHTML = highlight(row.name, row.spans);
      else nameEl.textContent = row.label ?? row.name;
      text.appendChild(nameEl);
      if (row.sub) {
        const sub = document.createElement("span");
        sub.className = "h3lm-sub";
        sub.textContent = row.sub;
        text.appendChild(sub);
      }
      item.appendChild(text);
      if (row.favorite) {
        const star = document.createElement("span");
        star.className = "h3lm-fav";
        star.textContent = "★";
        item.appendChild(star);
      }
      if (row.badge) {
        const tag = document.createElement("span");
        tag.className = "h3lm-tag";
        tag.textContent = row.badge;
        item.appendChild(tag);
      }
      item.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        e.stopPropagation();
        commit(row.patch);
      });
      list.appendChild(item);
    });
    scrollToSel();
  };

  /* ---- the two sources ---- */

  const searchManager = async () => {
    const mine = ++generation;
    inflight?.abort();
    inflight = new AbortController();
    let data;
    try {
      data = await managerSearch(input.value.trim(), {
        baseModel: baseSelect.value,
        favorites: favBox.checked,
        signal: inflight.signal,
      });
    } catch (err) {
      if (err?.name === "AbortError" || mine !== generation) return;
      console.warn("[H3PowerLoraManagerStack] LoRA Manager search failed", err);
      healthState = { ok: false, at: 0 };
      managerMode = false;
      applyMode();
      render();
      return;
    }
    if (mine !== generation) return;
    rows = data.items.map((item) => {
      const patch = rowFromManagerItem(item);
      const folder = String(item.folder || "").replace(/\\/g, "/");
      const version = item?.civitai?.name;
      return {
        name: patch.lora,
        label: item.model_name || item.file_name || patch.lora,
        title: patch.lmPath || patch.lora,
        sub: [folder ? `${folder}/` : "", item.file_name, version ? `· ${version}` : ""]
          .filter(Boolean)
          .join(" "),
        badge: item.base_model || "",
        favorite: !!item.favorite,
        previewUrl: patch.previewUrl,
        patch,
      };
    });
    sel = 0;
    renderRows(data.total);
  };

  const searchLocal = async () => {
    if (localNames === null) localNames = await fetchLoraList();
    const query = input.value.trim().toLowerCase();
    const terms = query ? query.split(/\s+/) : [];
    const matched = [];
    for (const name of localNames) {
      const spans = matchTerms(name, terms);
      if (spans === null) continue;
      matched.push({ name, spans, s: score(name, terms, spans) });
    }
    if (terms.length) matched.sort((a, b) => a.s - b.s);
    rows = matched.map((row) => ({
      name: row.name,
      spans: row.spans,
      title: row.name,
      patch: { lora: row.name },
    }));
    sel = 0;
    if (!terms.length) {
      const at = rows.findIndex((r) => r.name === current);
      if (at > 0) sel = at;
    }
    renderRows(localNames.length);
  };

  const render = () => (managerMode ? searchManager() : searchLocal());

  const schedule = () => {
    clearTimeout(pending);
    // the local list is already in the browser, so only the live query waits
    pending = setTimeout(render, managerMode ? SEARCH_DEBOUNCE : 0);
  };

  /* ---- mode ---- */

  const applyMode = () => {
    input.placeholder = managerMode
      ? "Search the LoRA Manager library…"
      : "Type to filter models/loras…  (space-separated terms all must match)";
    bar.replaceChildren();
    if (managerMode) bar.append(baseSelect, favLabel, count);
    else bar.append(count);
    modeButton.textContent = managerMode
      ? "source: LoRA Manager"
      : "source: models/loras";
    modeButton.title = managerMode
      ? "Switch to ComfyUI's own models/loras listing"
      : "Search the LoRA Manager library instead";
    link.style.display = managerMode ? "" : "none";
  };

  modeButton.addEventListener("pointerdown", async (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!managerMode && !(await managerAvailable())) {
      modeButton.textContent = "LoRA Manager not reachable";
      return;
    }
    managerMode = !managerMode;
    applyMode();
    if (managerMode) await fillBaseModels();
    render();
    input.focus();
  });

  const fillBaseModels = async () => {
    if (baseModels !== null) return;
    baseModels = await managerBaseModels();
    baseSelect.replaceChildren();
    const any = document.createElement("option");
    any.value = "";
    any.textContent = "any base model";
    baseSelect.appendChild(any);
    for (const name of baseModels) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      baseSelect.appendChild(option);
    }
  };

  /* ---- keyboard ---- */

  const scrollToSel = () => {
    const el = list.children[sel];
    if (el && el.scrollIntoView) el.scrollIntoView({ block: "nearest" });
  };

  const move = (delta) => {
    if (!rows.length) return;
    const prev = list.children[sel];
    if (prev) prev.classList.remove("sel");
    sel = (sel + delta + rows.length) % rows.length;
    const next = list.children[sel];
    if (next) next.classList.add("sel");
    scrollToSel();
  };

  input.addEventListener("input", schedule);
  baseSelect.addEventListener("change", render);
  favBox.addEventListener("change", render);
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
    else if (e.key === "PageDown") { e.preventDefault(); move(10); }
    else if (e.key === "PageUp") { e.preventDefault(); move(-10); }
    else if (e.key === "Enter") {
      e.preventDefault();
      if (rows[sel]) commit(rows[sel].patch);
    } else if (e.key === "Escape") {
      e.preventDefault();
      close();
    }
    e.stopPropagation();   // keep ComfyUI's global hotkeys out of the box
  });

  applyMode();
  if (managerMode) await fillBaseModels();
  render();

  const grabFocus = () => {
    if (root.isConnected && document.activeElement !== input) input.focus();
  };
  // Defer by a full task, not a microtask: the pointerdown that opened this is
  // still being dispatched, and a capture listener added now would see it and
  // close the picker immediately.
  setTimeout(() => {
    document.addEventListener("pointerdown", onOutside, true);
    grabFocus();
  }, 0);
  // LiteGraph pulls focus back to the canvas on a delay in some paths; re-assert
  // once so the caret really is in the search box and typing goes there.
  setTimeout(grabFocus, 60);
}

function loraWidgets(node) {
  return (node.widgets || []).filter((w) => w.type === ROW_TYPE);
}

/** Shape test for a serialized stack row, as it appears in widgets_values. */
function isRowValue(v) {
  return !!v && typeof v === "object" && !Array.isArray(v) && "lora" in v;
}

/* --------------------------------------------------------------- balance -- */

function activeRows(node) {
  return loraWidgets(node).filter(
    (w) => w.value.on && w.value.lora && w.value.lora !== "None"
  );
}

/**
 * Whether the stack is in auto-balance mode.  Derived from the rows rather than
 * held as node state: ``manual`` is stashed on every balanced row and rows are
 * already serialized, so the mode survives a save/reload for free.
 */
function isBalanced(node) {
  return loraWidgets(node).some((w) => w.value.manual !== undefined);
}

function updateBalanceLabel(node) {
  const button = (node.widgets || []).find((w) => w.h3Role === "balance");
  if (!button) return;
  if (!isBalanced(node)) {
    button.name = BALANCE_LABEL;
  } else {
    const applied = loraWidgets(node).filter((w) => w.value.autoApplied);
    const trimmed = applied.filter((w) => (w.value.factor ?? 1) < 0.995).length;
    button.name = `⚖ Balanced — ${trimmed}/${applied.length} trimmed`;
  }
  node.setDirtyCanvas(true, true);
}

/**
 * Put every active row onto one strength unit.
 *
 * The measured factor multiplies the strength the user already chose -- or the
 * preset the LoRA Manager holds for that model -- so their relative intent
 * between rows survives; what changes is that a LoRA which perturbs the model
 * 18x harder than usual stops arriving at full force.  The factor only ever
 * trims (server-side it is clamped to <= 1): a LoRA measuring below the
 * reference may be quiet deliberately -- distillation adapters sit an order of
 * magnitude down and are correct at 1.0 -- so boosting is not safe, while a
 * LoRA measuring far above it essentially never is.
 *
 * ``force`` re-measures every active row, which is what the button does.  The
 * implicit calls -- made when a row is switched on or repointed inside an
 * already-balanced stack -- only touch rows that carry no stash yet, so a row
 * the user has since edited by hand is not quietly pulled back to its
 * calibrated value.
 */
async function applyBalance(node, { force = false } = {}) {
  const rows = activeRows(node).filter((w) => force || w.value.manual === undefined);
  if (!rows.length) {
    updateBalanceLabel(node);
    return;
  }
  const request = [];
  const seen = new Set();
  for (const w of rows) {
    if (seen.has(w.value.lora)) continue;
    seen.add(w.value.lora);
    request.push({ name: w.value.lora, path: w.value.lmPath ?? null });
  }
  let data;
  try {
    data = await fetchBalance(request);
  } catch (err) {
    console.error("[H3PowerLoraManagerStack] auto-balance failed", err);
    const button = (node.widgets || []).find((w) => w.h3Role === "balance");
    if (button) {
      button.name = "⚖ Balance failed";
      button.tooltip = err?.message || "Could not measure the selected LoRAs";
      node.setDirtyCanvas(true, true);
    }
    return;
  }
  const results = data?.results ?? {};
  for (const widget of rows) {
    const result = results[widget.value.lora];
    if (!result) continue;
    const manual =
      widget.value.autoApplied && widget.value.manual !== undefined
        ? widget.value.manual
        : widget.value.strength;
    const rawFactor = Number(result.factor ?? 1);
    if (!Number.isFinite(rawFactor) || rawFactor < 0) continue;
    const factor = Math.min(1, rawFactor);
    widget.value = {
      ...widget.value,
      manual,
      strength: manual * factor,
      factor,
      rel: result.rel ?? null,
      note: result.note ?? "",
      autoApplied: true,
    };
    if (result.note) {
      console.warn(`[H3PowerLoraManagerStack] ${widget.value.lora}: ${result.note}`);
    }
  }
  updateBalanceLabel(node);
  node.setDirtyCanvas(true, true);
}

/** Hand every row back the strength it had before auto-balance touched it. */
function restoreManual(node) {
  let restored = 0;
  for (const widget of loraWidgets(node)) {
    if (widget.value.manual === undefined) continue;
    const next = { ...widget.value, strength: widget.value.manual };
    delete next.manual;
    delete next.factor;
    delete next.rel;
    delete next.note;
    delete next.autoApplied;
    widget.value = next;
    restored += 1;
  }
  updateBalanceLabel(node);
  node.setDirtyCanvas(true, true);
  return restored;
}

/* ----------------------------------------------------------------- rows -- */

function renumber(node) {
  loraWidgets(node).forEach((w, i) => {
    w.name = `lora_${i + 1}`;
  });
}

function addLoraWidget(node, value) {
  const widget = makeLoraWidget(node, `lora_${loraWidgets(node).length + 1}`, value);
  node.widgets = node.widgets || [];
  // keep the "Add LoRA" button last
  const buttonIndex = node.widgets.findIndex((w) => w.h3Role === "add");
  if (buttonIndex >= 0) node.widgets.splice(buttonIndex, 0, widget);
  else node.widgets.push(widget);
  renumber(node);
  resize(node);
  return widget;
}

function removeLoraWidget(node, widget) {
  const i = node.widgets.indexOf(widget);
  if (i >= 0) node.widgets.splice(i, 1);
  renumber(node);
  resize(node);
  node.setDirtyCanvas(true, true);
}

function resize(node) {
  const [minWidth, minHeight] = node.computeSize();
  // Width stays the user's choice; height tracks the row count so removing a
  // row hands the space back instead of leaving a hole.
  node.setSize([Math.max(node.size[0], minWidth), minHeight]);
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "h3.power.lora.manager.stack",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      this.serialize_widgets = true;

      // (value, canvas, node, pos, event) -- the event is the pointerdown that
      // started the click, so the search bar opens right under the button.
      const add = this.addWidget("button", "🔍 Add LoRA", null,
        (_v, _canvas, _node, _pos, event) => {
          const widget = addLoraWidget(this);
          // open the search bar immediately: the row is added to be filled in,
          // so the box takes the keyboard straight away
          showLoraMenu(this, widget, event, { removeOnCancel: true });
        });
      add.h3Role = "add";

      const balance = this.addWidget("button", BALANCE_LABEL, null, () => {
        applyBalance(this, { force: true });
      });
      balance.h3Role = "balance";

      const restore = this.addWidget("button", RESTORE_LABEL, null, () => {
        if (!restoreManual(this)) {
          console.info("[H3PowerLoraManagerStack] nothing to restore, no balance applied");
        }
      });
      restore.h3Role = "restore";

      // The buttons hold no state, and leaving them in widgets_values is what
      // makes a stack grow on every load: LiteGraph restores that array
      // positionally over the widgets that exist before onConfigure runs -- the
      // fixed inputs and these three buttons -- so saved rows land in a
      // button's `value`, are written back out on the next save, and come home
      // as extra rows.  Opting out of serialization drops them from both halves
      // of that round trip.
      for (const button of [add, balance, restore]) button.serialize = false;

      resize(this);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      onConfigure?.apply(this, arguments);
      // ComfyUI restores widgets_values positionally against the widgets that
      // existed before this ran, which is only the fixed inputs and the
      // buttons.  The stack rows are recovered by shape instead of position,
      // so a slot shifting cannot corrupt them.
      const values = info?.widgets_values;
      if (!Array.isArray(values)) return;
      const rows = values.filter(isRowValue);
      for (const w of loraWidgets(this)) removeLoraWidget(this, w);
      for (const row of rows) addLoraWidget(this, row);
      updateBalanceLabel(this);   // rows carry the balance state, so recover it
      resize(this);
    };
  },
});
