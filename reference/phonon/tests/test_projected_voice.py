"""Linear algebra/layout contract for the independent projected-voice exporter."""
import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_projected_voice import project


class ProjectedVoiceTests(unittest.TestCase):
    def test_layout_and_single_projection(self):
        raw = torch.zeros((1, 512, 2), dtype=torch.float32)
        raw[0, 0] = torch.tensor([2., 3.])
        raw[0, 1] = torch.tensor([5., 7.])
        weight = torch.zeros((512, 512), dtype=torch.float32)
        weight[0, 0], weight[0, 1] = 11., 13.
        weight[1, 0], weight[1, 1] = 17., 19.
        output = project(raw, weight)
        # y[t,o] = sum_c raw[c,t]*weight[o,c], without a second projection.
        expected = torch.zeros((1, 2, 512), dtype=torch.float32)
        expected[0, 0, :2] = torch.tensor([87., 129.])
        expected[0, 1, :2] = torch.tensor([124., 184.])
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_rejects_unbound_shapes_and_nonfinite_source(self):
        weight = torch.eye(512, dtype=torch.float32)
        for raw in (torch.zeros((1, 2, 512)), torch.zeros((2, 512, 1)),
                    torch.zeros((1, 512, 0)), torch.full((1, 512, 1), float("nan"))):
            with self.assertRaises(ValueError):
                project(raw, weight)


if __name__ == "__main__":
    unittest.main()
