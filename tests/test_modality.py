import unittest

import torch

from h3lora.modality import apply_to_state_dict


class ModalityTests(unittest.TestCase):
    def test_default_b_and_diff_b_are_scaled(self):
        expand, hidden = 2, 4
        rows = 3 * expand * hidden
        sd = {
            "diffusion_model.blocks.0.adaln_proj.linear.lora_B.default.weight": torch.ones(rows, 2),
            "diffusion_model.blocks.0.adaln_proj.linear.diff_b": torch.ones(rows),
        }
        out, stats = apply_to_state_dict(sd, (0.0, 1.0, 1.0), (expand, 3, hidden))
        self.assertEqual(stats["scaled"], 2)
        self.assertTrue(torch.equal(out[next(iter(sd))][: expand * hidden], torch.zeros(expand * hidden, 2)))
        self.assertTrue(torch.equal(out[next(reversed(sd))][: expand * hidden], torch.zeros(expand * hidden)))


if __name__ == "__main__":
    unittest.main()
