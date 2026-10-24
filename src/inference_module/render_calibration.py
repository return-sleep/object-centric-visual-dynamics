"""Validation-safe renderer calibration utilities."""

import torch
import torch.nn.functional as F


def apply_static_residual(
    prediction,
    condition_rgb,
    observed_reconstruction,
    strength=0.0,
    motion_threshold=0.03,
    dilation=7,
):
    """Correct static pixels using only the six observed conditioning frames.

    All tensors use the model's [-1, 1] RGB range. ``prediction`` is [T,C,H,W],
    conditioning data are [B,T,C,H,W], and reconstruction is [B,C,H,W].
    """
    if strength == 0:
        return prediction
    residual = condition_rgb[:, -1] - observed_reconstruction
    motion = (
        (condition_rgb - condition_rgb[:, :1])
        .abs()
        .amax(dim=1)
        .mean(dim=1, keepdim=True)
        / 2.0
    )
    moving = (motion >= motion_threshold).to(prediction.dtype)
    if dilation > 1:
        moving = F.max_pool2d(
            moving, kernel_size=dilation, stride=1, padding=dilation // 2
        )
    return prediction + strength * (residual * (1.0 - moving))[0]
