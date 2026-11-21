"""Unit tests for checkpoint management, safe loading, and state isolation."""
import os
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from medimg.checkpoints import (
    CheckpointMetadata,
    ModelStatus,
    clone_state_dict,
    compute_file_sha256,
    load_checkpoint_verified,
    save_checkpoint,
)
from medimg.models import UNet, classifier


def test_checkpoint_clone_state_dict_isolation():
    """Verify that clone_state_dict produces an isolated snapshot on CPU,

    preventing subsequent in-place model mutations from corrupting the best state.
    (Regression test for DEF-06).
    """
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    ).to("cpu")

    # Take snapshot using clone_state_dict
    snapshot = clone_state_dict(model)

    # Record initial weights of first layer
    w_initial = snapshot["0.weight"].clone()

    # Mutate model weights in-place
    with torch.no_grad():
        model[0].weight.add_(10.0)

    # Snapshot must remain completely unchanged
    w_after_mutation = snapshot["0.weight"]
    assert torch.equal(w_initial, w_after_mutation), "clone_state_dict failed: snapshot was mutated in-place!"
    assert not torch.equal(model[0].weight, snapshot["0.weight"]), "Model weight did not change!"


def test_checkpoint_metadata_and_sha256(tmp_path):
    """Verify safe checkpoint serialization with SHA-256 hash and metadata sidecar."""
    model = nn.Linear(4, 2)
    meta = CheckpointMetadata(
        architecture="linear_test",
        sha256="",
        num_parameters=sum(p.numel() for p in model.parameters()),
        input_size=128,
        notes="Test unit checkpoint"
    )
    ckpt_path = tmp_path / "test_model.pt"
    saved_path = save_checkpoint(model, ckpt_path, metadata=meta)

    assert saved_path.exists()
    meta_path = saved_path.with_suffix(".meta.json")
    assert meta_path.exists()

    file_sha = compute_file_sha256(saved_path)
    assert len(file_sha) == 64
    assert meta.sha256 == file_sha


def test_missing_checkpoint_returns_proper_status():
    """Verify that loading from a non-existent path returns CHECKPOINT_MISSING without crashing."""
    model = nn.Linear(4, 2)
    fake_path = "runs/checkpoints/non_existent_fake_ckpt.pt"
    ok, status, msg, meta = load_checkpoint_verified(model, fake_path, expected_arch="test")

    assert not ok
    assert status == ModelStatus.CHECKPOINT_MISSING
    assert "not found" in msg.lower()
    assert meta is None


def test_incompatible_checkpoint_returns_proper_status(tmp_path):
    """Verify that architecture key mismatch returns CHECKPOINT_INCOMPATIBLE."""
    unet_model = UNet(in_ch=3, base=8)
    ckpt_path = tmp_path / "unet_test.pt"
    torch.save(unet_model.state_dict(), ckpt_path)

    # Attempt to load UNet weights into a ResNet-18 classifier
    cls_model = classifier("resnet18", pretrained=False)
    ok, status, msg, _ = load_checkpoint_verified(cls_model, ckpt_path, strict=True)

    assert not ok
    assert status == ModelStatus.CHECKPOINT_INCOMPATIBLE
    assert "mismatch" in msg.lower()
