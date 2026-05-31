from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PartialLoadReport:
    loaded_tensors: int
    skipped_tensors: int
    missing_tensors: int
    source_path: Path


def _extract_state_dict(payload):
    for key in ("model", "state_dict", "module", "net"):
        if isinstance(payload, dict) and key in payload and isinstance(payload[key], dict):
            return payload[key]
    return payload


def _candidate_target_keys(source_key: str) -> tuple[str, ...]:
    prefixes = (
        "model.",
        "module.",
        "_orig_mod.",
        "net.",
        "model.net.",
        "module.net.",
        "model.module.",
        "model.module.net.",
        "diffusion_model.",
        "model.diffusion_model.",
    )
    candidates = [source_key]
    changed = True
    while changed:
        changed = False
        for key in tuple(candidates):
            for prefix in prefixes:
                if key.startswith(prefix):
                    stripped = key[len(prefix) :]
                    if stripped not in candidates:
                        candidates.append(stripped)
                        changed = True
    return tuple(candidates)


def save_checkpoint(path: str | Path, model, optimizer=None, step: int = 0, extra: dict | None = None) -> None:
    payload = {"model": model.state_dict(), "step": step}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model, optimizer=None, strict: bool = True) -> int:
    payload = torch.load(path, map_location="cpu")
    state = _extract_state_dict(payload)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if not strict and (missing or unexpected):
        print(f"Loaded with missing={len(missing)} unexpected={len(unexpected)}")
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0))


def load_compatible_checkpoint(path: str | Path, model) -> PartialLoadReport:
    """Load only tensors whose keys and shapes match the TokenLight model.

    This is intentionally strict about silent failures: callers should reject a
    zero-tensor load when they require a real pretrained base initialization.
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    source_state = _extract_state_dict(payload)
    target_state = model.state_dict()
    compatible = {}
    skipped = 0
    for key, value in source_state.items():
        if not isinstance(value, torch.Tensor):
            skipped += 1
            continue
        matched_key = None
        for candidate in _candidate_target_keys(key):
            if candidate in target_state and tuple(target_state[candidate].shape) == tuple(value.shape):
                matched_key = candidate
                break
        if matched_key is None:
            skipped += 1
            continue
        compatible[matched_key] = value
    target_state.update(compatible)
    model.load_state_dict(target_state, strict=True)
    return PartialLoadReport(
        loaded_tensors=len(compatible),
        skipped_tensors=skipped,
        missing_tensors=len(target_state) - len(compatible),
        source_path=path,
    )
