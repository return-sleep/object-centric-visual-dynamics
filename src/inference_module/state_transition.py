"""Pure state-transition helpers for autoregressive object dynamics.

State layout with ``num_velocity = n``:

    [xy_t, velocity_t, velocity_{t-1}, ..., velocity_{t-n+1}, appearance_t]

The velocity slots are a history of *one-frame finite differences*.  They are
not cumulative displacements.  Keeping these operations independent from
``main.py`` makes that invariant directly unit-testable.
"""

import torch


TRANSITION_MODES = ("legacy", "history_shift", "acceleration_shift")


def perturb_delta(delta, num_velocity, std, mode="legacy"):
    """Add rollout noise without breaking the selected state semantics.

    Legacy checkpoints independently autoregress every velocity slot, so their
    historical sampling behavior is preserved.  Shift-register modes derive
    all velocity slots from the noisy motion update and therefore perturb only
    the two motion outputs.
    """
    if std <= 0:
        return delta
    eps = torch.randn_like(delta[..., :2]) * std
    if mode != "legacy":
        return torch.cat([delta[..., :2] + eps, delta[..., 2:]], dim=-1)

    parts = [delta[..., :2] + eps]
    for lag in range(num_velocity):
        start = 2 + 2 * lag
        parts.append(delta[..., start:start + 2] + eps)
    parts.append(delta[..., 2 + 2 * num_velocity:])
    return torch.cat(parts, dim=-1)


def transition_state(
    cur_state,
    delta_state,
    num_velocity,
    mode="legacy",
    coef_delta_decay=1.0,
    app_anchor=None,
    absolute_app=False,
    rotation_step_max=0.0,
    scale_step_max=0.0,
    velocity_xy_blend=1.0,
):
    """Apply one dynamics step while preserving velocity-history semantics."""
    if mode not in TRANSITION_MODES:
        raise ValueError("unknown transition mode: %s" % mode)
    if mode == "acceleration_shift" and num_velocity < 1:
        raise ValueError("acceleration_shift requires num_velocity >= 1")

    vel_end = 2 + 2 * num_velocity
    motion_delta = delta_state[..., :vel_end]
    rotation_output = delta_state[..., vel_end:-1]
    scale_output = delta_state[..., -1:]
    if rotation_step_max > 0:
        rotation_output = rotation_step_max * torch.tanh(
            rotation_output / rotation_step_max
        )
    if scale_step_max > 0:
        scale_output = scale_step_max * torch.tanh(
            scale_output / scale_step_max
        )
    app_output = torch.cat([rotation_output, scale_output], dim=-1)
    app_output = app_output * coef_delta_decay

    if mode == "legacy":
        next_motion = cur_state[..., :vel_end] + motion_delta
        if num_velocity > 0 and velocity_xy_blend < 1.0:
            # Legacy checkpoints predict both an xy increment and the next
            # finite-difference velocity.  Blend these redundant estimates
            # without changing any velocity-history or appearance semantics.
            predicted_velocity = next_motion[..., 2:4]
            position_step = (
                velocity_xy_blend * motion_delta[..., :2]
                + (1.0 - velocity_xy_blend) * predicted_velocity
            )
            next_motion = torch.cat(
                [cur_state[..., :2] + position_step, next_motion[..., 2:]],
                dim=-1,
            )
    else:
        if mode == "history_shift":
            step_velocity = delta_state[..., :2]
        else:
            step_velocity = cur_state[..., 2:4] + delta_state[..., :2]
        next_xy = cur_state[..., :2] + step_velocity
        velocity_history = [step_velocity]
        for lag in range(1, num_velocity):
            start = 2 + 2 * (lag - 1)
            velocity_history.append(cur_state[..., start:start + 2])
        next_motion = torch.cat([next_xy, *velocity_history], dim=-1)

    if absolute_app:
        next_app = app_output
    elif app_anchor is not None:
        next_app = app_anchor + app_output
    else:
        next_app = cur_state[..., vel_end:] + app_output
    return torch.cat([next_motion, next_app], dim=-1)
