from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class CameraLightTransform:
    camera: torch.Tensor
    light: torch.Tensor
    intensity: torch.Tensor
    radius: torch.Tensor


def _tensor3(value, device=None) -> torch.Tensor:
    out = torch.as_tensor(value, dtype=torch.float32, device=device)
    if out.shape[-1] != 3:
        raise ValueError(f"Expected a 3D vector, got shape {tuple(out.shape)}")
    return out


def apply_sim3(
    points,
    canonical_center,
    target_center,
    scale,
    rotation=None,
) -> torch.Tensor:
    """Apply the paper's canonical-to-scene Sim(3) transform."""
    points = _tensor3(points)
    device = points.device
    canonical_center = _tensor3(canonical_center, device=device)
    target_center = _tensor3(target_center, device=device)
    scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
    if rotation is None:
        rotation = torch.eye(3, dtype=torch.float32, device=device)
    else:
        rotation = torch.as_tensor(rotation, dtype=torch.float32, device=device)
        if rotation.shape[-2:] != (3, 3):
            raise ValueError(f"Expected a 3x3 rotation, got {tuple(rotation.shape)}")
    shifted = points - canonical_center
    rotated = shifted @ rotation.transpose(-1, -2)
    return target_center + scale * rotated


def transform_camera_light(
    camera,
    light,
    intensity,
    radius,
    canonical_center,
    target_center,
    scale,
    rotation=None,
) -> CameraLightTransform:
    """Transform camera/light and preserve inverse-square falloff behavior.

    The paper states:
      p'cam   = Ct + s(pcam - C)
      p'light = Ct + s(plight - C)
      E'      = s^2 E
      d'      = s d
    This implementation also accepts an optional rotation matrix for the scene
    Sim(3), which is required for non-axis-aligned scene placement.
    """
    device = torch.as_tensor(camera).device
    scale_t = torch.as_tensor(scale, dtype=torch.float32, device=device)
    return CameraLightTransform(
        camera=apply_sim3(camera, canonical_center, target_center, scale_t, rotation),
        light=apply_sim3(light, canonical_center, target_center, scale_t, rotation),
        intensity=scale_t.square() * torch.as_tensor(intensity, dtype=torch.float32, device=device),
        radius=scale_t * torch.as_tensor(radius, dtype=torch.float32, device=device),
    )
