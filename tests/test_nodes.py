"""Row collection and LoRA resolution for the single stack node."""

import json
import os
import tempfile
import unittest
from unittest import mock

from h3lora import loramanager, nodes


class FakeFolderPaths:
    """Just the two ``folder_paths`` calls the node makes."""

    def __init__(self, mapping):
        self.mapping = dict(mapping)

    def get_filename_list(self, kind):
        return sorted(self.mapping) if kind == "loras" else []

    def get_full_path(self, kind, name):
        return self.mapping.get(name) if kind == "loras" else None


def patch_loras(mapping):
    return mock.patch.object(nodes, "folder_paths", FakeFolderPaths(mapping))


class ResolveTests(unittest.TestCase):
    def test_exact_name_wins(self):
        with patch_loras({"sub/a.safetensors": "/models/loras/sub/a.safetensors"}):
            self.assertEqual(nodes._resolve_lora("sub/a.safetensors"),
                             "/models/loras/sub/a.safetensors")

    def test_backslashes_and_case_are_tolerated(self):
        with patch_loras({"sub/a.safetensors": "/models/loras/sub/a.safetensors"}):
            self.assertEqual(nodes._resolve_lora("SUB\\A.safetensors"),
                             "/models/loras/sub/a.safetensors")

    def test_bare_basename_resolves_when_it_is_unambiguous(self):
        with patch_loras({"sub/a.safetensors": "/models/loras/sub/a.safetensors"}):
            self.assertEqual(nodes._resolve_lora("a.safetensors"),
                             "/models/loras/sub/a.safetensors")

    def test_ambiguous_basename_is_refused(self):
        with patch_loras({
            "one/a.safetensors": "/models/loras/one/a.safetensors",
            "two/a.safetensors": "/models/loras/two/a.safetensors",
        }):
            self.assertIsNone(nodes._resolve_lora("a.safetensors"))

    def test_manager_path_is_the_fallback_for_a_library_outside_models_loras(self):
        with tempfile.TemporaryDirectory() as tmp:
            outside = os.path.join(tmp, "extra.safetensors")
            open(outside, "wb").close()
            with patch_loras({}):
                self.assertEqual(nodes._resolve_lora("extra.safetensors", outside),
                                 outside)

    def test_a_stale_manager_path_falls_back_to_the_local_basename(self):
        with patch_loras({"sub/a.safetensors": "/models/loras/sub/a.safetensors"}):
            self.assertEqual(
                nodes._resolve_lora("moved/a.safetensors", "/gone/a.safetensors"),
                "/models/loras/sub/a.safetensors",
            )

    def test_nothing_resolves_to_none(self):
        with patch_loras({}):
            self.assertIsNone(nodes._resolve_lora("missing.safetensors"))
            self.assertIsNone(nodes._resolve_lora("None"))


class CollectTests(unittest.TestCase):
    LORAS = {
        "a.safetensors": "/models/loras/a.safetensors",
        "b.safetensors": "/models/loras/b.safetensors",
    }

    def collect(self, kwargs):
        issues = []
        with patch_loras(self.LORAS):
            return nodes._collect(kwargs, issues), issues

    def test_rows_keep_ui_order_regardless_of_kwargs_order(self):
        entries, issues = self.collect({
            "lora_2": {"on": True, "lora": "b.safetensors", "strength": 0.5},
            "lora_1": {"on": True, "lora": "a.safetensors", "strength": 1.0},
        })
        self.assertEqual([e["name"] for e in entries],
                         ["a.safetensors", "b.safetensors"])
        self.assertEqual([e["row"] for e in entries], [1, 2])
        self.assertEqual(issues, [])

    def test_disabled_zeroed_and_unfilled_rows_are_dropped_silently(self):
        entries, issues = self.collect({
            "lora_1": {"on": False, "lora": "a.safetensors", "strength": 1.0},
            "lora_2": {"on": True, "lora": "b.safetensors", "strength": 0.0},
            "lora_3": {"on": True, "lora": "None", "strength": 1.0},
        })
        self.assertEqual(entries, [])
        self.assertEqual(issues, [])

    def test_a_bad_strength_is_reported_not_guessed(self):
        entries, issues = self.collect({
            "lora_1": {"on": True, "lora": "a.safetensors", "strength": "loud"},
            "lora_2": {"on": True, "lora": "b.safetensors", "strength": float("inf")},
        })
        self.assertEqual(entries, [])
        self.assertEqual(len(issues), 2)

    def test_an_unresolvable_row_is_reported(self):
        entries, issues = self.collect({
            "lora_1": {"on": True, "lora": "ghost.safetensors", "strength": 1.0},
        })
        self.assertEqual(entries, [])
        self.assertIn("could not resolve", issues[0])

    def test_manager_fields_ride_along_on_the_row(self):
        entries, _ = self.collect({
            "lora_1": {
                "on": True, "lora": "a.safetensors", "strength": 0.8,
                "lmPath": "/models/loras/a.safetensors",
                "triggerWords": ["neon", 7, "glow"],
                "baseModel": "MiniMax H3",
            },
        })
        self.assertEqual(entries[0]["trigger_words"], ["neon", "glow"])

    def test_non_row_kwargs_are_ignored(self):
        entries, issues = self.collect({
            "lora_stack": [("a.safetensors", 1.0, 1.0)],
            "loras": {"not": "a row"},
            "lora_1": {"strength": 1.0},          # no "lora" key
        })
        self.assertEqual(entries, [])
        self.assertEqual(issues, [])


class LoraStackTests(unittest.TestCase):
    LORAS = {"a.safetensors": "/models/loras/a.safetensors"}

    def test_entries_are_adopted_and_clip_strength_ignored(self):
        issues = []
        with patch_loras(self.LORAS):
            entries = nodes._collect_lora_stack(
                [("a.safetensors", 0.7, 0.3)], issues)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["name"], "a.safetensors")
        self.assertAlmostEqual(entries[0]["strength"], 0.7)
        self.assertNotIn("row", entries[0])
        self.assertEqual(issues, [])

    def test_os_separators_from_the_manager_stacker_resolve(self):
        loras = {"sub/a.safetensors": "/models/loras/sub/a.safetensors"}
        issues = []
        with patch_loras(loras):
            entries = nodes._collect_lora_stack(
                [(os.path.join("sub", "a.safetensors"), 1.0, 1.0)], issues)
        self.assertEqual(entries[0]["path"], "/models/loras/sub/a.safetensors")

    def test_empty_and_malformed_entries(self):
        issues = []
        with patch_loras(self.LORAS):
            self.assertEqual(nodes._collect_lora_stack(None, issues), [])
            self.assertEqual(nodes._collect_lora_stack([], issues), [])
            self.assertEqual(nodes._collect_lora_stack([("a.safetensors",)], issues), [])
            self.assertEqual(nodes._collect_lora_stack(
                [("ghost.safetensors", 1.0, 1.0)], issues), [])
        self.assertEqual(len(issues), 2)

    def test_zero_strength_entries_are_skipped_without_complaint(self):
        issues = []
        with patch_loras(self.LORAS):
            entries = nodes._collect_lora_stack([("a.safetensors", 0.0, 0.0)], issues)
        self.assertEqual(entries, [])
        self.assertEqual(issues, [])


class TriggerWordTests(unittest.TestCase):
    def setUp(self):
        loramanager._cache.clear()

    def test_row_words_are_used_and_deduplicated_in_order(self):
        entries = [
            {"path": "/nowhere/a.safetensors", "trigger_words": ["neon", "glow"]},
            {"path": "/nowhere/b.safetensors", "trigger_words": ["glow", "rim"]},
        ]
        self.assertEqual(nodes._trigger_words(entries), "neon,, glow,, rim")

    def test_a_row_without_words_falls_back_to_the_manager_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = os.path.join(tmp, "a.safetensors")
            open(model, "wb").close()
            with open(loramanager.sidecar_path(model), "w", encoding="utf-8") as h:
                json.dump({"civitai": {"trainedWords": ["from sidecar"]}}, h)
            self.assertEqual(
                nodes._trigger_words([{"path": model, "trigger_words": None}]),
                "from sidecar",
            )

    def test_no_words_anywhere_is_an_empty_string(self):
        self.assertEqual(nodes._trigger_words([]), "")
        self.assertEqual(
            nodes._trigger_words([{"path": "/nowhere/x.safetensors",
                                   "trigger_words": []}]),
            "",
        )


class RegistrationTests(unittest.TestCase):
    def test_the_pack_registers_exactly_one_node(self):
        self.assertEqual(list(nodes.NODE_CLASS_MAPPINGS), ["H3PowerLoraManagerStack"])
        self.assertEqual(sorted(nodes.NODE_DISPLAY_NAME_MAPPINGS),
                         sorted(nodes.NODE_CLASS_MAPPINGS))

    def test_the_node_id_does_not_collide_with_the_original_pack(self):
        self.assertNotIn("H3PowerLoraStack", nodes.NODE_CLASS_MAPPINGS)

    def test_the_inputs_the_frontend_relies_on_are_declared(self):
        optional = nodes.H3PowerLoraManagerStack.INPUT_TYPES()["optional"]
        for name in ("model", "quantized_layers", "adaln_port", "adaln_video",
                     "adaln_text", "adaln_audio", "lora_stack"):
            self.assertIn(name, optional.data)
        # any lora_N the browser invents is accepted with a permissive type
        self.assertEqual(optional["lora_17"], (nodes.ANY,))

    def test_outputs_include_the_manager_trigger_words(self):
        cls = nodes.H3PowerLoraManagerStack
        self.assertEqual(cls.RETURN_NAMES, ("MODEL", "report", "trigger_words"))
        self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES))
        self.assertEqual(len(cls.OUTPUT_TOOLTIPS), len(cls.RETURN_NAMES))

    def test_no_model_is_a_clear_error(self):
        with self.assertRaises(ValueError):
            nodes.H3PowerLoraManagerStack().apply(model=None)


if __name__ == "__main__":
    unittest.main()
