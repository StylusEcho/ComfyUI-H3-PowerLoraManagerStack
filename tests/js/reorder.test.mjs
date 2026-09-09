/**
 * Row reordering and picker-filter persistence, run under plain `node`.
 *
 *   node tests/js/reorder.test.mjs
 *
 * The widget module is written for ComfyUI's browser runtime, so it is loaded
 * into a VM context with `app`, `api` and the handful of DOM and storage calls
 * it makes at load time stubbed.  Only the logic that is independent of the
 * canvas is exercised: which slot a row lands in, that the fixed widgets and
 * the buttons are left alone, and that a filter round-trips through storage.
 * Drawing, dragging and hovering need a real browser and are not covered.
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const WIDGET = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../web/h3_power_lora_manager_stack.js",
);
const ROW_TYPE = "H3LM_LORA";

function load() {
  // The imports are ComfyUI-only; everything they provide is stubbed below.
  const source = fs.readFileSync(WIDGET, "utf8").replace(/^import .*$/gm, "");
  const store = new Map();
  const element = () => ({
    style: {},
    classList: { add() {}, remove() {} },
    children: [],
    appendChild(child) { this.children.push(child); return child; },
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    addEventListener() {}, removeEventListener() {}, remove() {}, focus() {},
    getBoundingClientRect: () => ({ top: 0, left: 0, width: 200, height: 120 }),
    querySelector: () => null,
    contains: () => false,
  });

  const context = {
    console, setTimeout, clearTimeout, URLSearchParams,
    requestAnimationFrame: (fn) => fn(),
    document: {
      createElement: element, getElementById: () => null,
      head: element(), body: element(),
      addEventListener() {}, removeEventListener() {},
    },
    window: { innerWidth: 1920, innerHeight: 1080 },
    localStorage: {
      getItem: (key) => (store.has(key) ? store.get(key) : null),
      setItem: (key, value) => store.set(key, value),
    },
    Image: class { set src(_value) {} },
    app: { registerExtension() {}, canvas: { ds: { scale: 1 } } },
    api: { fetchApi: async () => ({ ok: false }) },
  };
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(
    `${source}\n;globalThis.__api = { moveRow, loraWidgets, readPrefs, writePrefs };`,
    context,
  );
  return { ...context.__api, store };
}

/** A node shaped like the real one: a fixed widget, N rows, then the buttons. */
function makeNode(names) {
  const node = {
    widgets: [{ type: "combo", name: "quantized_layers" }],
    size: [300, 100],
    computeSize: () => [300, 100],
    setSize() {}, setDirtyCanvas() {},
  };
  for (const lora of names) {
    node.widgets.push({ type: ROW_TYPE, name: "unnumbered", value: { lora } });
  }
  node.widgets.push({ type: "button", h3Role: "add" });
  return node;
}

const lm = load();
const order = (node) => lm.loraWidgets(node).map((w) => w.value.lora);

// Objects built inside the VM carry that context's Object.prototype, which a
// strict deep-equal counts as a difference; compare their contents instead.
const plain = (value) => JSON.parse(JSON.stringify(value));

test("a row moves to the top", () => {
  const node = makeNode(["a", "b", "c", "d"]);
  lm.moveRow(node, lm.loraWidgets(node)[2], 0);
  assert.deepEqual(order(node), ["c", "a", "b", "d"]);
});

test("a row moves to the bottom", () => {
  const node = makeNode(["a", "b", "c", "d"]);
  lm.moveRow(node, lm.loraWidgets(node)[0], 3);
  assert.deepEqual(order(node), ["b", "c", "d", "a"]);
});

test("moving down by one lands past the neighbour, not on it", () => {
  // The case an index-arithmetic implementation gets wrong: removing the row
  // first shifts every later index left by one.
  const node = makeNode(["a", "b", "c", "d"]);
  lm.moveRow(node, lm.loraWidgets(node)[1], 2);
  assert.deepEqual(order(node), ["a", "c", "b", "d"]);
});

test("moving up by one lands before the neighbour", () => {
  const node = makeNode(["a", "b", "c", "d"]);
  lm.moveRow(node, lm.loraWidgets(node)[2], 1);
  assert.deepEqual(order(node), ["a", "c", "b", "d"]);
});

test("a move onto its own slot changes nothing", () => {
  const node = makeNode(["a", "b", "c"]);
  assert.equal(lm.moveRow(node, lm.loraWidgets(node)[1], 1), false);
  assert.deepEqual(order(node), ["a", "b", "c"]);
});

test("targets outside the list are clamped", () => {
  const past = makeNode(["a", "b", "c"]);
  lm.moveRow(past, lm.loraWidgets(past)[0], 99);
  assert.deepEqual(order(past), ["b", "c", "a"]);

  const before = makeNode(["a", "b", "c"]);
  lm.moveRow(before, lm.loraWidgets(before)[2], -5);
  assert.deepEqual(order(before), ["c", "a", "b"]);
});

test("rows are renumbered so lora_N stays dense and in order", () => {
  // This is what the Python side sorts rows by, and what a schedule's row
  // selectors address.
  const node = makeNode(["a", "b", "c"]);
  lm.moveRow(node, lm.loraWidgets(node)[0], 2);
  assert.deepEqual(lm.loraWidgets(node).map((w) => w.name),
                   ["lora_1", "lora_2", "lora_3"]);
});

test("the fixed widgets and the buttons are left where they are", () => {
  const node = makeNode(["a", "b", "c"]);
  lm.moveRow(node, lm.loraWidgets(node)[2], 0);
  assert.deepEqual(node.widgets.filter((w) => w.type !== ROW_TYPE).map((w) => w.type),
                   ["combo", "button"]);
  assert.equal(node.widgets[node.widgets.length - 1].h3Role, "add");
});

test("a moved row keeps its whole value, balance stash included", () => {
  const node = makeNode(["a", "b"]);
  const row = lm.loraWidgets(node)[1];
  Object.assign(row.value, { strength: 0.42, manual: 0.8, factor: 0.5,
                             autoApplied: true, lmPath: "/loras/b.safetensors" });
  lm.moveRow(node, row, 0);
  assert.deepEqual(lm.loraWidgets(node)[0].value, {
    lora: "b", strength: 0.42, manual: 0.8, factor: 0.5,
    autoApplied: true, lmPath: "/loras/b.safetensors",
  });
});

test("picker filters round-trip through storage and merge", () => {
  assert.deepEqual(plain(lm.readPrefs()), {});
  lm.writePrefs({ baseModel: "MiniMax H3" });
  lm.writePrefs({ folder: "styles" });
  assert.deepEqual(plain(lm.readPrefs()),
                   { baseModel: "MiniMax H3", folder: "styles" });
});

test("unreadable stored filters fall back to none", () => {
  lm.store.set("h3lm.picker.filters", "{ not json");
  assert.deepEqual(plain(lm.readPrefs()), {});
});
