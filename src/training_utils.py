"""Small parameter-freezing helpers shared by training and tests."""

import torch


def enable_decoder_only(perception_model, dynamics_model):
    for parameter in dynamics_model.parameters():
        parameter.requires_grad_(False)
    for parameter in perception_model.parameters():
        parameter.requires_grad_(False)
    for parameter in perception_model.decoder.parameters():
        parameter.requires_grad_(True)


def to_piqa_range(model_rgb):
    """Convert renderer RGB from [-1,1] to PIQA's evaluation range [0,1]."""
    return ((model_rgb + 1.0) / 2.0).clamp(0.0, 1.0)
