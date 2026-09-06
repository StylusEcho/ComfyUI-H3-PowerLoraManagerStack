import unittest
from types import SimpleNamespace

import torch

from h3lora.sidecars import (
    STASH_ATTR,
    attach,
    collect,
    is_h3_base,
    peel,
    wrap_load_model_weights,
)


class _WarnConfig:
    """Mimics Comfy ``supported_models_base.BASE.__getattr__``."""

    def __init__(self, unet_config=None, sidecars=None):
        self.unet_config = unet_config or {"image_model": "minimax_h3"}
        if sidecars is not None:
            self.adaln_sidecars = sidecars
        self.accessed = []

    def __getattr__(self, name):
        self.accessed.append(name)
        raise AssertionError(
            "model config __getattr__(%r) must not be used" % (name,)
        )


class _H3Base:
    def __init__(self, config=None, leftover_capture=None):
        self.model_config = config or _WarnConfig()
        self.diffusion_model = SimpleNamespace()
        self.leftover_capture = leftover_capture if leftover_capture is not None else []
        self.current_patcher = None

    def load_model_weights(self, sd, unet_prefix="", assign=False):
        self.leftover_capture.extend(
            k[len(unet_prefix):] for k in list(sd) if k.startswith(unet_prefix)
        )
        return self


class SidecarPeelTests(unittest.TestCase):
    def test_peel_pops_prefixed_tensors(self):
        basis = torch.randn(8, 3)
        mean = torch.randn(8)
        sd = {
            "diffusion_model.adaln_basis": basis,
            "diffusion_model.adaln_mean": mean,
            "diffusion_model.adaln_t_table": torch.randn(16, 3),
            "diffusion_model.blocks.0.weight": torch.randn(2, 2),
        }
        got = peel(sd, "diffusion_model.")
        self.assertTrue(torch.equal(got["adaln_basis"], basis))
        self.assertTrue(torch.equal(got["adaln_mean"], mean))
        self.assertNotIn("silu_t_emb_grid", got)
        self.assertEqual(sorted(sd), [
            "diffusion_model.adaln_t_table",
            "diffusion_model.blocks.0.weight",
        ])

    def test_peel_bare_keys_and_grid(self):
        grid = torch.randn(4, 8)
        sd = {"adaln_basis": torch.randn(8, 2), "silu_t_emb_grid": grid}
        got = peel(sd, "")
        self.assertIn("adaln_basis", got)
        self.assertTrue(torch.equal(got["silu_t_emb_grid"], grid))
        self.assertFalse(sd)

    def test_peel_ignores_non_tensors(self):
        sd = {"adaln_basis": "not-a-tensor", "adaln_mean": torch.randn(3)}
        got = peel(sd)
        self.assertEqual(list(got), ["adaln_mean"])
        self.assertIn("adaln_basis", sd)


class SidecarCollectTests(unittest.TestCase):
    def test_collect_uses_stashed_model_attr(self):
        basis = torch.randn(8, 3)
        mean = torch.randn(8)
        model = _H3Base()
        attach(model, {"adaln_basis": basis, "adaln_mean": mean})
        patcher = SimpleNamespace(model=model, get_attachment=lambda key: None)
        got = collect(patcher, model.diffusion_model)
        self.assertTrue(torch.equal(got["adaln_basis"], basis))
        self.assertTrue(torch.equal(got["adaln_mean"], mean))
        self.assertEqual(model.model_config.accessed, [])

    def test_collect_reads_config_dict_without_getattr(self):
        sidecars = {"adaln_basis": torch.randn(4, 2), "adaln_mean": torch.randn(4)}
        cfg = _WarnConfig(sidecars=sidecars)
        model = SimpleNamespace(model_config=cfg, diffusion_model=SimpleNamespace())
        patcher = SimpleNamespace(model=model, get_attachment=lambda key: None)
        got = collect(patcher)
        self.assertTrue(torch.equal(got["adaln_basis"], sidecars["adaln_basis"]))
        self.assertEqual(cfg.accessed, [])

    def test_collect_skips_missing_config_attr(self):
        cfg = _WarnConfig()
        model = SimpleNamespace(model_config=cfg, diffusion_model=SimpleNamespace())
        patcher = SimpleNamespace(model=model)
        self.assertIsNone(collect(patcher))
        self.assertEqual(cfg.accessed, [])


class SidecarHookTests(unittest.TestCase):
    def test_is_h3_from_unet_config_without_getattr(self):
        model = _H3Base()
        self.assertTrue(is_h3_base(model))
        self.assertEqual(model.model_config.accessed, [])
        other = SimpleNamespace(
            model_config=_WarnConfig({"image_model": "flux"}),
            diffusion_model=SimpleNamespace(),
        )
        self.assertFalse(is_h3_base(other))
        self.assertEqual(other.model_config.accessed, [])

    def test_hook_peels_before_load_and_is_idempotent(self):
        leftover = []
        wrap_load_model_weights(_H3Base)
        wrap_load_model_weights(_H3Base)
        model = _H3Base(leftover_capture=leftover)
        basis = torch.randn(8, 3)
        mean = torch.randn(8)
        sd = {
            "adaln_basis": basis,
            "adaln_mean": mean,
            "blocks.0.weight": torch.randn(2, 2),
        }
        model.load_model_weights(sd, unet_prefix="")
        self.assertEqual(leftover, ["blocks.0.weight"])
        self.assertNotIn("adaln_basis", sd)
        stashed = getattr(model, STASH_ATTR)
        self.assertTrue(torch.equal(stashed["adaln_basis"], basis))
        self.assertTrue(torch.equal(stashed["adaln_mean"], mean))
        self.assertTrue(torch.equal(
            getattr(model.diffusion_model, STASH_ATTR)["adaln_basis"], basis))
        self.assertEqual(model.model_config.accessed, [])

    def test_hook_leaves_non_h3_state_dict_alone(self):
        class Other:
            def __init__(self):
                self.model_config = _WarnConfig({"image_model": "flux"})
                self.seen = None

            def load_model_weights(self, sd, unet_prefix="", assign=False):
                self.seen = dict(sd)
                return self

        wrap_load_model_weights(Other)
        model = Other()
        sd = {"adaln_basis": torch.randn(2, 2), "w": torch.randn(1)}
        model.load_model_weights(sd)
        self.assertIn("adaln_basis", model.seen)
        self.assertFalse(hasattr(model, STASH_ATTR))


if __name__ == "__main__":
    unittest.main()
