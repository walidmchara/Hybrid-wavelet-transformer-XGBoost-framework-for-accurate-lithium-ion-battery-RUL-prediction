"""Recovery contract: optimizer state, epoch, RNG, integrity and mismatch guard."""
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_recovery import EpochCheckpoint


class RecoveryTests(unittest.TestCase):
    def test_round_trip_and_config_protection(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        x = torch.ones(2, 3)
        optimizer.zero_grad()
        (model(x).square().sum()).backward()
        optimizer.step()
        weights = {k: t.detach().clone() for k, t in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as temp:
            store = EpochCheckpoint(Path(temp), {"dataset": "HUST", "epochs": 50})
            store.save(1, {"model": model, "optimizer": optimizer}, {"best": 0.12})
            expected_random = random.random()
            expected_np = float(np.random.random())
            expected_torch = torch.rand(1).item()
            random.random()
            np.random.random()
            torch.rand(1)
            for param in model.parameters():
                param.data.zero_()
            restored = store.restore({"model": model, "optimizer": optimizer})
            self.assertEqual(restored["epoch"], 1)
            self.assertEqual(restored["best"], 0.12)
            self.assertTrue(all(torch.equal(model.state_dict()[k], v) for k, v in weights.items()))
            self.assertEqual(random.random(), expected_random)
            self.assertEqual(float(np.random.random()), expected_np)
            self.assertEqual(torch.rand(1).item(), expected_torch)
            with self.assertRaises(ValueError):
                EpochCheckpoint(Path(temp), {"dataset": "MIT", "epochs": 50}).restore(
                    {"model": model, "optimizer": optimizer}
                )


if __name__ == "__main__":
    unittest.main()
