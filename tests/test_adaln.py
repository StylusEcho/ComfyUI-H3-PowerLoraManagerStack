import unittest

import torch

from h3lora.adaln import AdalnContext, fit_basis, port_adaln_pairs, sidecar_basis


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


if __name__ == "__main__":
    unittest.main()
