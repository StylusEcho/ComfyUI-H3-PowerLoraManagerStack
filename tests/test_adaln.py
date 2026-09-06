import unittest

import torch

from h3lora.adaln import (
    AdalnContext, fit_basis, mark_stack_adaln, port_adaln_pairs,
    port_attached_patches, sidecar_basis, upstream_adaln_fix_warning,
)


class AdalnSidecarTests(unittest.TestCase):
    def test_fit_basis_recovers_affine(self):
        g, k, d = 32, 4, 16
        table = torch.randn(g, k)
        v = torch.randn(d, k)
        c = torch.randn(d)
        grid = c.unsqueeze(0) + table @ v.T
        v2, c2, residual = fit_basis(table, grid)
        self.assertLess(residual, 1e-5)
        self.assertTrue(torch.allclose(v2, v, atol=1e-4))
        self.assertTrue(torch.allclose(c2, c, atol=1e-4))

    def test_sidecar_v_c_preferred(self):
        v = torch.randn(16, 4)
        c = torch.randn(16)
        got = sidecar_basis({"adaln_basis": v, "adaln_mean": c}, torch.randn(8, 4))
        self.assertIsNotNone(got)
        self.assertEqual(got[3], "checkpoint sidecars")
        self.assertEqual(got[2], 0.0)
        self.assertTrue(torch.equal(got[0], v.float()))

    def test_context_uses_sidecars_without_grid_file(self):
        g, k, d = 16, 3, 8
        table = torch.randn(g, k)
        v = torch.randn(d, k)
        c = torch.randn(d)
        ctx = AdalnContext(k, table, grid_path="", sidecars={"adaln_basis": v, "adaln_mean": c})
        basis = ctx.basis()
        self.assertIsNotNone(basis)
        self.assertEqual(ctx.source, "checkpoint sidecars")
        a = torch.randn(2, d)
        b = torch.randn(6, 2)
        sd = {
            "blocks.0.adaln_proj.linear.lora_A.weight": a,
            "blocks.0.adaln_proj.linear.lora_B.weight": b,
        }
        out, stats = port_adaln_pairs(sd, ctx)
        self.assertEqual(stats["ported"], 1)
        self.assertEqual(out["blocks.0.adaln_proj.linear.lora_A.weight"].shape[1], k)

    def test_fit_basis_normalizes_mixed_devices(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        table = torch.randn(16, 3, device="cuda")
        grid = torch.randn(16, 8)
        _v, _c, residual = fit_basis(table, grid)
        self.assertTrue(torch.isfinite(torch.tensor(residual)))

    def test_default_suffix_is_ported(self):
        table = torch.randn(16, 3)
        ctx = AdalnContext(
            3,
            table,
            sidecars={"adaln_basis": torch.randn(8, 3), "adaln_mean": torch.randn(8)},
        )
        sd = {
            "blocks.0.adaln_proj.linear.lora_A.default.weight": torch.randn(2, 8),
            "blocks.0.adaln_proj.linear.lora_B.default.weight": torch.randn(6, 2),
        }
        out, stats = port_adaln_pairs(sd, ctx)
        self.assertEqual(stats["ported"], 1)
        self.assertIn("blocks.0.adaln_proj.linear.lora_A.weight", out)
        self.assertIn("blocks.0.adaln_proj.linear.lora_B.weight", out)

    def test_strip_drops_mismatched_pairs(self):
        table = torch.randn(16, 3)
        ctx = AdalnContext(
            3,
            table,
            sidecars={"adaln_basis": torch.randn(8, 3), "adaln_mean": torch.randn(8)},
        )
        sd = {
            "blocks.0.adaln_proj.linear.lora_A.weight": torch.randn(2, 8),
            "blocks.0.adaln_proj.linear.lora_B.weight": torch.randn(6, 2),
            "blocks.0.other.lora_A.weight": torch.randn(2, 4),
        }
        out, stats = port_adaln_pairs(sd, ctx, mode="strip")
        self.assertEqual(stats["skipped"], 1)
        self.assertNotIn("blocks.0.adaln_proj.linear.lora_A.weight", out)
        self.assertIn("blocks.0.other.lora_A.weight", out)


class _Adapter:
    def __init__(self, weights):
        self.weights = weights


class _Patcher:
    def __init__(self, patches, sd):
        self.patches = {k: list(v) for k, v in patches.items()}
        self._sd = sd
        self.patches_uuid = None

    def clone(self):
        return _Patcher(self.patches, self._sd)

    def model_state_dict(self):
        return self._sd

    def add_patches(self, patches, strength):
        for key, value in patches.items():
            self.patches.setdefault(key, []).append((strength, value, 1.0, None, None))
        return list(patches)


class AttachedPatchTests(unittest.TestCase):
    def _ctx(self, k=3, d=8, g=16):
        table = torch.randn(g, k)
        v = torch.randn(d, k)
        c = torch.randn(d)
        ctx = AdalnContext(k, table, sidecars={"adaln_basis": v, "adaln_mean": c})
        return ctx, v, c

    def test_ports_dense_lora_already_on_model(self):
        ctx, v, c = self._ctx()
        r, out = 2, 6
        a = torch.randn(r, v.shape[0])
        b = torch.randn(out, r)
        key = "diffusion_model.blocks.0.adaln_proj.linear.weight"
        bias_key = "diffusion_model.blocks.0.adaln_proj.linear.bias"
        sd = {key: torch.randn(out, v.shape[1]), bias_key: torch.randn(out)}
        patcher = _Patcher(
            {key: [(0.5, _Adapter((b, a, None, None, None, None)), 1.0, None, None)]},
            sd,
        )
        out_p, stats = port_attached_patches(patcher, ctx)
        self.assertEqual(stats["ported"], 1)
        self.assertEqual(stats["keys"], 1)
        rebuilt = out_p.patches[key][-1][1]
        self.assertEqual(tuple(rebuilt.weights[1].shape), (r, v.shape[1]))
        self.assertIn(bias_key, out_p.patches)
        expected_a = a.float() @ v.float()
        self.assertTrue(torch.allclose(rebuilt.weights[1].float(), expected_a, atol=1e-4))

    def test_strip_removes_mismatched_attached(self):
        ctx, v, _c = self._ctx()
        key = "diffusion_model.blocks.0.adaln_proj.linear.weight"
        sd = {key: torch.randn(6, v.shape[1])}
        patcher = _Patcher(
            {key: [(1.0, _Adapter((torch.randn(6, 2), torch.randn(2, v.shape[0]),
                                   None, None, None, None)), 1.0, None, None)]},
            sd,
        )
        out_p, stats = port_attached_patches(patcher, ctx, mode="strip")
        self.assertEqual(stats["stripped"], 1)
        self.assertNotIn(key, out_p.patches)

    def test_skips_locon_mid(self):
        ctx, v, _c = self._ctx()
        key = "diffusion_model.blocks.0.adaln_proj.linear.weight"
        sd = {key: torch.randn(6, v.shape[1])}
        locon = _Adapter((torch.randn(6, 2), torch.randn(2, v.shape[0]),
                          None, torch.randn(2, 2), None, None))
        patcher = _Patcher({key: [(1.0, locon, 1.0, None, None)]}, sd)
        _out_p, stats = port_attached_patches(patcher, ctx)
        self.assertEqual(stats["ported"], 0)
        self.assertEqual(stats["unportable"], 1)


class UpstreamFixWarningTests(unittest.TestCase):
    def _patcher(self, n, marked=False):
        patches = {}
        for i in range(n):
            wkey = f"diffusion_model.blocks.{i}.adaln_proj.linear.weight"
            bkey = f"diffusion_model.blocks.{i}.adaln_proj.linear.bias"
            w = torch.randn(6, 3)
            b = torch.randn(6)
            patches[wkey] = [(1.0, ("diff", (w,)), 1.0, None, None)]
            patches[bkey] = [(1.0, ("diff", (b,)), 1.0, None, None)]
        patcher = _Patcher(patches, {})
        patcher.model_options = {}
        if marked:
            mark_stack_adaln(patcher)
        return patcher

    def test_warns_on_multiplied_out_adaln_diffs(self):
        note = upstream_adaln_fix_warning(self._patcher(10))
        self.assertIn("post-hoc AdaLN fix", note)
        self.assertIn("Disable the extra AdaLN LoRA Fix node", note)
        self.assertIn("adaln_port=off", note)

    def test_silent_when_too_few_keys(self):
        self.assertEqual(upstream_adaln_fix_warning(self._patcher(2)), "")

    def test_silent_when_this_stack_already_marked(self):
        self.assertEqual(upstream_adaln_fix_warning(self._patcher(10, marked=True)), "")


if __name__ == "__main__":
    unittest.main()
