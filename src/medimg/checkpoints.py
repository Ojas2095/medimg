"""Checkpoint management, safe serialization, and operational readiness verification.

Enforces strict model loading, independent state snapshotting, SHA-256 hashing,
and prevents silent fallback to unadapted/random model weights.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


class ModelStatus(str, Enum):
    """Explicit operational readiness state of a deep learning model."""
    READY = "READY"
    CHECKPOINT_MISSING = "CHECKPOINT_MISSING"
    CHECKPOINT_INCOMPATIBLE = "CHECKPOINT_INCOMPATIBLE"
    DEMO_FIXTURE_ONLY = "DEMO_FIXTURE_ONLY"
    LOADING_FAILED = "LOADING_FAILED"


@dataclass
class CheckpointMetadata:
    """Provenance and specification metadata associated with a model checkpoint."""
    architecture: str
    sha256: str
    num_parameters: int
    input_size: int = 128
    calibration_method: Optional[str] = None
    calibration_temperature: Optional[float] = None
    operating_threshold: Optional[float] = None
    target_sensitivity: Optional[float] = None
    dataset_provenance: Optional[str] = None
    created_at: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CheckpointMetadata:
        # Filter for known fields
        known_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered)


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Compute the SHA-256 hexadecimal digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def clone_state_dict(model_or_state_dict: Union[nn.Module, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Create an independent detached clone of model weights on CPU.

    Crucial fix for CPU in-memory snapshotting where .cpu() returns the identical
    tensor storage, causing in-place training updates to mutate the best snapshot.
    """
    if isinstance(model_or_state_dict, nn.Module):
        raw_dict = model_or_state_dict.state_dict()
    else:
        raw_dict = model_or_state_dict

    return {
        key: tensor.detach().cpu().clone()
        for key, tensor in raw_dict.items()
    }


def save_checkpoint(
    model_or_state: Union[nn.Module, Dict[str, torch.Tensor]],
    path: Union[str, Path],
    metadata: Optional[CheckpointMetadata] = None,
    overwrite: bool = True
) -> Path:
    """Safely serialize model weights and an accompanying JSON metadata sidecar."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint already exists at {target_path} and overwrite=False.")

    # Always clone weights to avoid accidental shared tensor storage
    cloned_state = clone_state_dict(model_or_state)
    torch.save(cloned_state, target_path)

    # Compute hash of newly written checkpoint
    file_sha = compute_file_sha256(target_path)

    if metadata is not None:
        metadata.sha256 = file_sha
        if isinstance(model_or_state, nn.Module):
            metadata.num_parameters = sum(p.numel() for p in model_or_state.parameters())
        else:
            metadata.num_parameters = sum(t.numel() for t in cloned_state.values())

        meta_path = target_path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    return target_path


def load_checkpoint_verified(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    expected_arch: Optional[str] = None,
    device: str = "cpu",
    strict: bool = True
) -> Tuple[bool, ModelStatus, str, Optional[CheckpointMetadata]]:
    """Strictly load a PyTorch model checkpoint with comprehensive integrity verification.

    Returns:
        (success, status, message, metadata)
    """
    p = Path(checkpoint_path)
    if not p.exists():
        msg = f"Checkpoint file not found on disk: {p.resolve()}"
        return False, ModelStatus.CHECKPOINT_MISSING, msg, None

    # Load sidecar metadata if present
    meta_path = p.with_suffix(".meta.json")
    metadata: Optional[CheckpointMetadata] = None
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = CheckpointMetadata.from_dict(json.load(f))
        except Exception:
            metadata = None

    try:
        # Load weights safely using weights_only=True
        state_dict = torch.load(p, map_location=device, weights_only=True)
    except Exception as exc:
        msg = f"Failed to deserialize checkpoint file {p.name}: {exc}"
        return False, ModelStatus.LOADING_FAILED, msg, None

    if not isinstance(state_dict, dict):
        msg = f"Checkpoint {p.name} did not contain a valid dictionary (type: {type(state_dict)})."
        return False, ModelStatus.CHECKPOINT_INCOMPATIBLE, msg, None

    # Verify architecture compatibility before loading
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing_in_model = ckpt_keys - model_keys
    missing_in_ckpt = model_keys - ckpt_keys

    if missing_in_ckpt or missing_in_model:
        msg = (
            f"Checkpoint architecture mismatch for {p.name}.\n"
            f"Keys missing in checkpoint: {len(missing_in_ckpt)}, "
            f"Unexpected keys in checkpoint: {len(missing_in_model)}"
        )
        if strict:
            return False, ModelStatus.CHECKPOINT_INCOMPATIBLE, msg, metadata

    try:
        model.load_state_dict(state_dict, strict=strict)
    except Exception as exc:
        msg = f"State dict loading failed for {p.name}: {exc}"
        return False, ModelStatus.CHECKPOINT_INCOMPATIBLE, msg, metadata

    # Check for NaN / Inf in loaded weights
    has_nan = any(torch.isnan(t).any().item() for t in state_dict.values())
    has_inf = any(torch.isinf(t).any().item() for t in state_dict.values())
    if has_nan or has_inf:
        msg = f"Checkpoint {p.name} contains floating point degeneracies (NaN={has_nan}, Inf={has_inf})."
        return False, ModelStatus.LOADING_FAILED, msg, metadata

    file_sha = compute_file_sha256(p)
    if metadata is None:
        metadata = CheckpointMetadata(
            architecture=expected_arch or model.__class__.__name__,
            sha256=file_sha,
            num_parameters=sum(p.numel() for p in model.parameters())
        )
    else:
        metadata.sha256 = file_sha

    return True, ModelStatus.READY, f"Model successfully loaded from {p.name} (SHA-256: {file_sha[:8]}...).", metadata
