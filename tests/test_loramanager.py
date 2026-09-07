"""The LoRA Manager sidecar bridge."""

import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from h3lora import loramanager


class SidecarTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.model = os.path.join(self.dir.name, "style_v2.safetensors")
        open(self.model, "wb").close()
        loramanager._cache.clear()

    def write(self, payload):
        path = loramanager.sidecar_path(self.model)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def test_sidecar_path_replaces_the_extension(self):
        self.assertEqual(
            loramanager.sidecar_path("/loras/sub/a.safetensors"),
            "/loras/sub/a.metadata.json",
        )

    def test_missing_sidecar_is_not_an_error(self):
        self.assertIsNone(loramanager.read_sidecar(self.model))
        self.assertEqual(loramanager.trigger_words(self.model), [])
        self.assertEqual(loramanager.usage_tips(self.model), {})
        self.assertIsNone(loramanager.preferred_strength(self.model))
        self.assertEqual(loramanager.describe(self.model), {})

    def test_trigger_words_come_from_civitai_block(self):
        self.write({"civitai": {"trainedWords": ["neon glow", "", "  ", "rimlight"]}})
        self.assertEqual(loramanager.trigger_words(self.model),
                         ["neon glow", "rimlight"])

    def test_trigger_words_fall_back_to_the_top_level_key(self):
        self.write({"trainedWords": ["cinematic"]})
        self.assertEqual(loramanager.trigger_words(self.model), ["cinematic"])

    def test_usage_tips_parse_from_the_stored_json_string(self):
        self.write({"usage_tips": json.dumps({"strength": 0.65})})
        self.assertEqual(loramanager.usage_tips(self.model), {"strength": 0.65})
        self.assertAlmostEqual(loramanager.preferred_strength(self.model), 0.65)

    def test_usage_tips_survive_a_dict_or_junk(self):
        self.write({"usage_tips": {"strength": "0.4"}})
        self.assertAlmostEqual(loramanager.preferred_strength(self.model), 0.4)
        loramanager._cache.clear()
        self.write({"usage_tips": "not json at all"})
        self.assertEqual(loramanager.usage_tips(self.model), {})
        self.assertIsNone(loramanager.preferred_strength(self.model))

    def test_unparseable_sidecar_is_ignored(self):
        with open(loramanager.sidecar_path(self.model), "w", encoding="utf-8") as h:
            h.write("{ this is not json")
        self.assertIsNone(loramanager.read_sidecar(self.model))

    def test_a_rewritten_sidecar_invalidates_the_cache(self):
        path = self.write({"civitai": {"trainedWords": ["first"]}})
        self.assertEqual(loramanager.trigger_words(self.model), ["first"])
        # The cache key is (mtime_ns, size); a same-length rewrite in the same
        # nanosecond is the only way to fool it, so change the length too.
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"civitai": {"trainedWords": ["second", "third"]}}, handle)
        os.utime(path, (0, 0))
        self.assertEqual(loramanager.trigger_words(self.model), ["second", "third"])

    def test_describe_reports_the_manager_fields(self):
        self.write({
            "model_name": "Neon Style",
            "base_model": "MiniMax H3",
            "favorite": True,
            "civitai": {"name": "v2"},
        })
        self.assertEqual(loramanager.describe(self.model), {
            "model_name": "Neon Style",
            "base_model": "MiniMax H3",
            "version": "v2",
            "favorite": True,
        })


class InstallDetectionTests(unittest.TestCase):
    """`root()` finds the manager by its files, never by importing it."""

    def make_custom_nodes(self, tmp, packs):
        parent = os.path.join(tmp, "custom_nodes")
        for name, markers in packs.items():
            for marker in markers:
                path = os.path.join(parent, name, marker)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                open(path, "w", encoding="utf-8").close()
            os.makedirs(os.path.join(parent, name), exist_ok=True)
        os.makedirs(parent, exist_ok=True)
        return parent

    def with_custom_nodes(self, parent):
        stub = types.ModuleType("folder_paths")
        stub.base_path = os.path.dirname(parent)
        stub.get_folder_paths = lambda kind: [parent] if kind == "custom_nodes" else []
        return mock.patch.dict(sys.modules, {"folder_paths": stub})

    def test_a_real_checkout_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = self.make_custom_nodes(tmp, {
                "ComfyUI-Lora-Manager": loramanager._MARKERS,
                "some-other-pack": ("nodes.py",),
            })
            with self.with_custom_nodes(parent):
                self.assertEqual(loramanager.root(),
                                 os.path.join(parent, "ComfyUI-Lora-Manager"))
                self.assertTrue(loramanager.installed())

    def test_a_pack_with_only_one_marker_is_not_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = self.make_custom_nodes(tmp, {
                "lookalike": (loramanager._MARKERS[0],),
            })
            with self.with_custom_nodes(parent):
                self.assertIsNone(loramanager.root())
                self.assertFalse(loramanager.installed())

    def test_no_custom_nodes_directory_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.with_custom_nodes(os.path.join(tmp, "nothing-here")):
                self.assertIsNone(loramanager.root())

    def test_without_folder_paths_it_simply_says_no(self):
        with mock.patch.dict(sys.modules, {"folder_paths": None}):
            self.assertIsNone(loramanager.root())


if __name__ == "__main__":
    unittest.main()
