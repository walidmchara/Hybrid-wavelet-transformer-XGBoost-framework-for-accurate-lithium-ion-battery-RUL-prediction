"""Atomic epoch-level PyTorch recovery snapshots.

Checkpoints may contain pickled Python objects. Restore ONLY files generated
by your own trusted runs. Store on a persistent volume (Drive, RunPod, etc.).
This module does not publish model weights or experiment results to GitHub.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def fingerprint(metadata: dict) -> str:
    canonical = json.dumps(metadata, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def rng_snapshot() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class EpochCheckpoint:
    """Stateful, atomic and configuration-checked epoch boundary snapshots."""

    def __init__(self, directory: str | Path, metadata: dict):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "latest.pt"
        self.metadata = metadata
        self.signature = fingerprint(metadata)

    def save(self, epoch: int, objects: dict, extra: dict | None = None) -> None:
        """epoch = number of *completed* epochs. Objects expose state_dict()."""
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        payload = {
            "format": "epoch-recovery-v1",
            "epoch": int(epoch),
            "metadata": self.metadata,
            "fingerprint": self.signature,
            "objects": {name: obj.state_dict() for name, obj in objects.items()},
            "extra": extra or {},
            "rng": rng_snapshot(),
        }
        fd, tmp = tempfile.mkstemp(prefix=".snapshot-", suffix=".pt", dir=self.directory)
        os.close(fd)
        try:
            torch.save(payload, tmp)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def restore(self, objects: dict, *, device: str | torch.device = "cpu") -> dict:
        if not self.path.is_file():
            raise FileNotFoundError(f"Recovery checkpoint not found: {self.path}")
        # Local, trusted checkpoints only. Do not load untrusted pickle files.
        payload = torch.load(self.path, map_location=device, weights_only=False)
        if payload.get("format") != "epoch-recovery-v1":
            raise ValueError("Unsupported checkpoint format")
        if payload.get("fingerprint") != self.signature:
            raise ValueError("Configuration/dataset mismatch: refusing an unsafe resume")
        if set(payload["objects"]) != set(objects):
            raise ValueError("Stateful component mismatch: refusing an unsafe resume")
        for name, obj in objects.items():
            obj.load_state_dict(payload["objects"][name])
        restore_rng(payload["rng"])
        return {"epoch": int(payload["epoch"]), **payload.get("extra", {})}
