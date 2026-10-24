#!/usr/bin/env python
# coding: gbk
"""
author: Huilin XU
date: 2022/12/07
description : ?????????????????????????????
test: python train_my_model.py --epoch 0 --weight_path 
python train_my_model.py --epoch 0 --given_frames 3 --num_gn 2 --Inference_model predictor_module_with_appr --consider_appr
"""

import pandas as pd
import os
import sys
import json
import math
import random
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import lpips
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils import *
from Dataloader.general_Video_Loaders import *
from src.baseline_model.FID_metric import calculate_fid
from src.KeypointNet.Derendering import Derendering
from src.KeypointNet.Derendering_with_rotate import Derendering_with_rotate
from src.KeypointNet.Derendering_newest import Derendering_newest
from src.baseline_model.GN_modules import (
    predictor_module,
    predictor_module_with_appr,
    predictor_module_apprmap,
    predictor_module_mlp,
    WholeRolloutTrajectoryCorrector,
)
from src.inference_module.record_dynamic import *
from src.inference_module.render_calibration import apply_static_residual
from src.inference_module.state_transition import (
    TRANSITION_MODES,
    perturb_delta as apply_rollout_noise,
    transition_state as apply_state_transition,
)
from src.training_utils import enable_decoder_only, to_piqa_range

# from train_derendering import validate as Perception_validate
from tensorboardX import SummaryWriter
import warnings

warnings.filterwarnings("ignore")


def visualize(cd, cd_hat, log_dir, batch_idx):
    import matplotlib.pyplot as plt

    plt.style.use("seaborn")
    # [b horzion C H W] ???batch ??????????????
    cd = (cd[-1].numpy().transpose(0, 2, 3, 1) + 1) / 2  # [-1,1] -> [0,1]
    cd_hat = (cd_hat[-1].numpy().transpose(0, 2, 3, 1) + 1) / 2
    fig, ax = plt.subplots(2, 8, figsize=(12, 4))

    # ???????8??
    idx = np.linspace(0, cd.shape[0] - 1, 8).astype(int)
    # print(idx)
    for k in range(8):
        ax[0][k].imshow(cd[idx[k]])
        ax[1][k].imshow(cd_hat[idx[k]])
        ax[0][k].axis("off")
        ax[1][k].axis("off")
        ax[0][k].set_title("t=" + str(idx[k]))
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, batch_idx + "_compare.jpg"))


def _damp_coef_delta(delta_state):
    """Scale the delta on appearance dims [2+2*num_velocity:] by
    args.coef_delta_decay. 1.0 (default) returns the delta unchanged."""
    if args.coef_delta_decay == 1.0:
        return delta_state
    d = 2 + 2 * args.num_velocity
    return torch.cat(
        [delta_state[..., :d], delta_state[..., d:] * args.coef_delta_decay],
        dim=-1,
    )


def _perturb_delta(delta_state, enabled=True):
    if not enabled or args.rollout_noise_std <= 0:
        return delta_state
    return apply_rollout_noise(
        delta_state,
        args.num_velocity,
        args.rollout_noise_std,
        mode=args.transition_mode,
    )


def _transition_state(cur_state, delta_state, app_anchor=None):
    return apply_state_transition(
        cur_state,
        delta_state,
        args.num_velocity,
        mode=args.transition_mode,
        coef_delta_decay=args.coef_delta_decay,
        app_anchor=app_anchor,
        absolute_app=args.absolute_app_head,
        rotation_step_max=args.rotation_step_max,
        scale_step_max=args.scale_step_max,
        velocity_xy_blend=args.velocity_xy_blend,
    )


def creat_state_squence(ab, n=2):
    """Append ``n`` one-frame finite-difference velocity-history slots.

    At time t the layout is ``[xy_t, v_t, v_{t-1}, ..., appearance_t]``.
    """
    B, T, K, S = ab.shape
    num_coef = S - 2
    if n == 0:
        new_ab = ab
    else:
        new_ab = torch.zeros(B, T - n, K, S + n * 2).to(ab.device)
        new_ab[..., :2] = ab[:, n:, :, :2]  # ????
        new_ab[..., -num_coef:] = ab[:, n:, :, -num_coef:]

        for i in range(n):
            if i == 0:
                new_ab[..., 2:4] = ab[:, n:, :, :2] - ab[:, n - 1 : -1, :, :2]
            else:
                new_ab[..., 2 * (i + 1) : 2 * (i + 2)] = (
                    ab[:, n - i : -i, :, :2] - ab[:, n - 1 - i : -1 - i, :, :2]
                )
    return new_ab


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch",
        default=300,
        type=int,
        help="Number of Epoch for training. Can be set to 0 for evaluation",
    )
    parser.add_argument(
        "--batchsize",
        default=16,
        type=int,
        help="Batchsize. Can be set to 1 for evaluation",
    )
    parser.add_argument("--lr", default=5e-4, type=float, help="Learning rate")
    parser.add_argument(
        "--dataset",
        default="blocktowerCF",
        type=str,
        choices=["ballsCF", "blocktowerCF", "obj3d"],
        help="Dataset used by the shared video-prediction pipeline",
    )
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument(
        "--given_frames",
        type=int,
        default=1,
        help="Given previous n frames to predict states of future frames, must > 1",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=25,
        help="predict states of horizon-step future frames",
    )
    parser.add_argument("--num_velocity", default=0, type=int)
    parser.add_argument(
        "--name",
        default="my_model_with_env",
        type=str,
        help="Name for weight saving, consider feature map around object positions",
    )
    # parser.add_argument('--pre_train', action='store_true')
    parser.add_argument("--weight_path", type=str, default="")
    parser.add_argument("--Perception_model_weight_path", type=str, default="")
    parser.add_argument(
        "--Perception_model",
        type=str,
        default="derendering",
        help="or derendering_with_rotate",
    )
    parser.add_argument(
        "--Inference_model",
        type=str,
        default="predictor_module",
        help="or else predictor_module_with_appr, predictor_module_apprmap",
    )
    parser.add_argument("--num_gn", default=4, type=int)
    parser.add_argument("--num_layer", default=2, type=int)
    parser.add_argument("--embed_size", default=128, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument(
        "--is_rnn", action="store_true", help="if use rnn why predict next state vector"
    )
    parser.add_argument(
        "--pos_emb", default=0, type=int, help="dim of pos embedding like nerf"
    )
    parser.add_argument(
        "--n_coefficients", default=4, type=int, help="Number of coefficients to use"
    )
    parser.add_argument(
        "--n_keypoints", default=4, type=int, help="Number of keypoints to use"
    )
    parser.add_argument(
        "--std", default=0.1, type=float, help="std of guassian-like heatmap"
    )
    parser.add_argument(
        "--filter_size",
        default=5,
        type=int,
        help="coefficient filter-bank kernel size (Derendering); must match "
        "the stage-1 run (auto-read from log.json with --perception_auto_config)",
    )
    parser.add_argument(
        "--use_pixelshuffle",
        action="store_true",
        help="perception decoder used PixelShuffle upsampling (must match the "
        "stage-1 run; auto-read from log.json with --perception_auto_config)",
    )
    parser.add_argument(
        "--shared_object_decoder",
        action="store_true",
        help="perception used permutation-invariant shared object decoder",
    )
    parser.add_argument(
        "--consider_appr",
        action="store_true",
        help="if consider feature map around each object",
    )
    parser.add_argument(
        "--agg_reduction",
        default="sum",
        choices=["sum", "avg", "max"],
        help="spatial reduction in context-aware aggregator",
    )
    parser.add_argument(
        "--mask_orientation",
        action="store_true",
        help="zero out the 4 rotation coefficients in the state sequence (dynamics-side ablation)",
    )
    parser.add_argument(
        "--mask_scale",
        action="store_true",
        help="zero out the scale coefficient in the state sequence (dynamics-side ablation)",
    )
    parser.add_argument(
        "--use_scale",
        action="store_true",
        help="When n_coefficients=0, keep the 1-d scale in the dynamics state (pos+scale perception).",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.0,
        help="add noise in training phase, 0.0003",
    )
    parser.add_argument(
        "--train_mode_verbose",
        default=0,
        type=int,
        help="0: free running mode, 1: teacher forcing, 2: Curriculum Learning",
    )
    parser.add_argument("--warm_epoch", default=5, type=int)
    parser.add_argument("--decay_epoch", default=20, type=int)
    parser.add_argument("--df_epoch", default=20, type=int)
    parser.add_argument("--autoenc_loss", action="store_true")
    parser.add_argument(
        "--loss_type",
        default="absoluate",
        type=str,
        help="predict states or states displacement, absoluate/relative/merge",
    )
    parser.add_argument("--pos_weight", default=1.0, type=float)
    parser.add_argument("--xy_loss_weight", default=-1.0, type=float,
                        help="explicit xy loss weight; -1 preserves pos_weight")
    parser.add_argument("--velocity_loss_weight", default=-1.0, type=float,
                        help="explicit velocity loss weight; -1 preserves pos_weight")
    parser.add_argument("--appearance_loss_weight", default=1.0, type=float)
    parser.add_argument("--scale_loss_weight", default=1.0, type=float)
    parser.add_argument("--delta_loss_weight", default=0.0, type=float,
                        help="weight for one-step state-delta MSE (already x1000)")
    parser.add_argument("--temporal_late_weight", default=1.0, type=float,
                        help="linear rollout weighting from 1 to this value")
    parser.add_argument("--discount_factor", action="store_true")
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help="optional suffix appended to the log dir name to distinguish reruns",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="deterministic training output directory; empty keeps the historical timestamp layout",
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="CUDA device index. Replaces the old hardcoded cuda:6.",
    )
    parser.add_argument(
        "--resolution",
        default=0,
        type=int,
        help="image resolution; 0 = auto (112 for CoPhy, 128 for OBJ3D)",
    )
    parser.add_argument(
        "--num_filter",
        default=32,
        type=int,
        help="perception encoder/decoder base channels (stage-1 num_filter)",
    )
    parser.add_argument("--obj3d_root", type=str, default="../Datasets/OBJ3D")
    parser.add_argument(
        "--episode_count_file", type=str, default="",
        help="JSON count labels used for count-OOD subset training")
    parser.add_argument(
        "--train_counts", type=str, default="",
        help="comma-separated allowed OBJ3D counts for dynamics training")
    parser.add_argument(
        "--val_counts", type=str, default="",
        help="comma-separated allowed OBJ3D counts for checkpoint selection")
    parser.add_argument(
        "--perception_auto_config",
        action="store_true",
        help="read K/C/num_filter/std/resolution from log.json next to the "
        "perception weight, so 128px nf=64 checkpoints load correctly",
    )
    parser.add_argument(
        "--export_pred_dir",
        type=str,
        default="",
        help="if set, test() exports best_val_model predictions as {ex}.npy "
        "[horizon,H,W,C] float32 in [0,1] (DDLP-protocol eval input)",
    )
    parser.add_argument(
        "--eval_split",
        default="test",
        choices=["val", "test"],
        help="dataset split used by epoch=0 export/evaluation",
    )
    parser.add_argument(
        "--decoder_checkpoint",
        default="",
        help="optional decoder-only state_dict; permits validation-safe cross-decoder probes",
    )
    parser.add_argument(
        "--static_residual_strength",
        type=float,
        default=0.0,
        help="add a conditioning-only reconstruction residual on static pixels",
    )
    parser.add_argument(
        "--static_residual_threshold",
        type=float,
        default=0.03,
        help="conditioning-window motion threshold in [0,1] units",
    )
    parser.add_argument(
        "--static_residual_dilation",
        type=int,
        default=7,
        help="odd max-pool kernel used to exclude moving pixels from residual correction",
    )
    parser.add_argument(
        "--hflip_self_ensemble",
        action="store_true",
        help="average the original rollout with a horizontally flipped copy",
    )
    parser.add_argument(
        "--feature_anchor",
        default="first",
        choices=["first", "last"],
        help="which conditioning frame supplies the decoder feature map",
    )
    parser.add_argument(
        "--max_videos",
        default=0,
        type=int,
        help=">0: subset every split to N videos (smoke test; cache is tagged)",
    )
    parser.add_argument(
        "--random_window",
        action="store_true",
        help="OBJ3D only: cache the full episode and train on "
        "random-offset windows (DDLP-style; collisions happen mid-episode). "
        "Val/test stay prefix-based. Cache is tagged T100, existing caches "
        "and default prefix behavior are untouched.",
    )
    parser.add_argument(
        "--windows_per_video",
        type=int,
        default=1,
        help="number of random-offset training windows drawn from each cached "
        "video per epoch; replicas use disjoint temporal strata. Has an effect "
        "only with --random_window.",
    )
    parser.add_argument(
        "--collision_sample_prob",
        type=float,
        default=0.0,
        help="probability of sampling random windows proportional to peak "
        "keypoint acceleration; remaining samples stay uniform",
    )
    parser.add_argument(
        "--rollout_noise_train_only",
        action="store_true",
        help="apply --rollout_noise_std only while the dynamics model is in "
        "train mode. Required for training runs; leave off to draw noisy "
        "samples at eval.",
    )
    parser.add_argument(
        "--detach_rollout",
        action="store_true",
        help="stop the gradient on the state fed to the next rollout step "
        "(pushforward trick, Brandstetter et al. 2022). Training still sees "
        "the free-running distribution, but each step is supervised through a "
        "single transition instead of a 14-step BPTT chain.",
    )
    parser.add_argument(
        "--rollout_noise_std",
        type=float,
        default=0.0,
        help="perturb the predicted position increment during free-running "
        "rollout. Shift-register modes derive velocity history from the noisy "
        "motion update; legacy preserves the historical sampler. Each sample is a "
        "dynamically plausible alternative future; averaging the renders of "
        "several samples hedges the collision outcome the way a model "
        "ensemble does (+0.45 dB PSNR / +0.005 SSIM measured over 7 models).",
    )
    parser.add_argument(
        "--transition_mode",
        default="legacy",
        choices=TRANSITION_MODES,
        help="state updater: legacy independently autoregresss every channel; "
        "history_shift treats the velocity channels as one-frame velocity "
        "history; acceleration_shift predicts acceleration and integrates it.",
    )
    parser.add_argument(
        "--absolute_app_head",
        action="store_true",
        help="interpret dynamics appearance/scale outputs as the absolute next "
        "state instead of a delta. This changes output semantics and should be "
        "trained from scratch.",
    )
    parser.add_argument(
        "--contact_adapter",
        action="store_true",
        help="add a zero-initialized relative-geometry contact correction to "
        "the predicted motion.",
    )
    parser.add_argument(
        "--contact_adapter_only",
        action="store_true",
        help="freeze the original GSGN and train only the zero-initialized "
        "relative-geometry contact adapter.",
    )
    parser.add_argument(
        "--temporal_adapter",
        action="store_true",
        help="add a zero-initialized Transformer correction over recent states.",
    )
    parser.add_argument(
        "--temporal_adapter_only",
        action="store_true",
        help="freeze the original GSGN and train only the temporal adapter.",
    )
    parser.add_argument(
        "--temporal_history",
        type=int,
        default=6,
        help="number of recent rolled-out states used by the temporal adapter.",
    )
    parser.add_argument(
        "--rollout_corrector",
        action="store_true",
        help="apply a zero-initialized non-autoregressive Transformer residual "
        "to XY after the complete baseline rollout has been produced.",
    )
    parser.add_argument(
        "--rollout_corrector_only",
        action="store_true",
        help="freeze the original GSGN and train only the whole-rollout corrector.",
    )
    parser.add_argument("--corrector_hidden_size", type=int, default=128)
    parser.add_argument("--corrector_layers", type=int, default=2)
    parser.add_argument("--corrector_heads", type=int, default=4)
    parser.add_argument(
        "--corrector_scale",
        type=float,
        default=0.05,
        help="maximum absolute XY residual in normalized coordinates.",
    )
    parser.add_argument(
        "--contact_radius",
        type=float,
        default=0.5,
        help="distance scale of the soft proximity gate in normalized xy units.",
    )
    parser.add_argument(
        "--rotation_step_max",
        type=float,
        default=0.0,
        help="positive value bounds each rotation-coefficient increment with "
        "m*tanh(delta/m); 0 preserves the original update exactly.",
    )
    parser.add_argument(
        "--scale_step_max",
        type=float,
        default=0.0,
        help="positive value bounds the scale increment with m*tanh(delta/m); "
        "0 preserves the original update exactly.",
    )
    parser.add_argument(
        "--velocity_xy_blend",
        type=float,
        default=1.0,
        help="legacy rollout only: blend the xy-head displacement with the "
        "velocity-head next-frame displacement; 1 preserves old checkpoints.",
    )
    parser.add_argument(
        "--long_horizon_calibrator",
        action="store_true",
        help="add a zero-initialized bounded correction after the original "
        "GSGN delta head.",
    )
    parser.add_argument(
        "--calibrator_scope",
        default="app",
        choices=["app", "full"],
        help="correct only rotation/scale or the complete state delta.",
    )
    parser.add_argument(
        "--calibrator_scale",
        type=float,
        default=0.05,
        help="maximum absolute correction contributed by the calibrator.",
    )
    parser.add_argument(
        "--adapter_only",
        action="store_true",
        help="freeze the original GSGN and train only the long-horizon calibrator.",
    )
    parser.add_argument(
        "--appearance_anchor",
        action="store_true",
        help="predict appearance/scale channels as an absolute offset from the "
        "last observed frame instead of accumulating a per-step delta. The "
        "state decomposition attributes +3.1 dB at step 44 to appearance "
        "error, most of it accumulation drift.",
    )
    parser.add_argument(
        "--train_window_max_frame",
        type=int,
        default=0,
        help="restrict --random_window offsets so windows end at or before "
        "this episode frame (0 = whole 100-frame episode). OBJ3D carries 85%% "
        "of its motion in the first 50 frames, and SlotFormer/G-SWM/DDLP all "
        "train on that prefix only; set 50 to match them.",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="number of micro-batches accumulated per optimizer update",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=5,
        help="ReduceLROnPlateau patience in validation epochs",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-8,
        help="minimum learning rate used by ReduceLROnPlateau",
    )
    parser.add_argument(
        "--dyn_perceptual_weight",
        type=float,
        default=0.0,
        help="weight of predicted-frame perceptual (LPIPS) loss in stage-2 "
        "training (DDLP beta_dyn_rec style). 0 = disabled (default, old "
        "behavior). Gradients flow through the frozen renderer into the "
        "dynamics states. Requires GT rgb: cached datasets get an _rgb cache.",
    )
    parser.add_argument(
        "--dyn_perceptual_net",
        default="vgg",
        choices=["vgg", "alex"],
        help="LPIPS backbone for the training perceptual loss (vgg matches "
        "the DDLP eval metric).",
    )
    parser.add_argument(
        "--dyn_perceptual_frames",
        type=int,
        default=0,
        help="render only the last N predicted frames for the perceptual "
        "loss; 0 = all horizon frames.",
    )
    parser.add_argument(
        "--metric_loss_backend",
        default="lpips",
        choices=["lpips", "piqa"],
        help="perceptual training backend; piqa exactly matches OBJ3D evaluation",
    )
    parser.add_argument(
        "--ssim_loss_weight",
        type=float,
        default=0.0,
        help="weight of differentiable 1-PIQA-SSIM on rendered rollout frames",
    )
    parser.add_argument("--rgb_loss_weight", type=float, default=0.0,
                        help="weight of robust pixel reconstruction loss")
    parser.add_argument(
        "--rgb_loss_type", default="charbonnier", choices=["charbonnier", "mse"],
        help="pixel objective; MSE directly aligns with PSNR",
    )
    parser.add_argument(
        "--render_frame_sampling", default="last", choices=["last", "uniform"],
        help="choose rendered loss frames from rollout tail or uniformly",
    )
    parser.add_argument(
        "--enforce_kinematic_velocity", action="store_true",
        help="deprecated alias for --transition_mode history_shift",
    )
    parser.add_argument(
        "--inertial_xy_update", action="store_true",
        help="deprecated alias for --transition_mode acceleration_shift",
    )
    parser.add_argument("--decoder_recon_weight", type=float, default=0.0,
                        help="GT-latent reconstruction regularizer for decoder fine-tuning")
    parser.add_argument(
        "--unfreeze_decoder",
        action="store_true",
        help="mode B: also backprop the perceptual loss into the perception "
        "decoder (encoder/bank stay frozen). Only effective with "
        "--dyn_perceptual_weight > 0. Decoder snapshots are saved as "
        "<ckpt>.decoder.pt next to each dynamics checkpoint.",
    )
    parser.add_argument(
        "--decoder_only",
        action="store_true",
        help="freeze dynamics and train only the perception decoder",
    )
    parser.add_argument(
        "--decoder_lr",
        type=float,
        default=1e-5,
        help="lr for the unfrozen decoder param group (mode B only).",
    )
    parser.add_argument(
        "--kpt_margin",
        type=float,
        default=0.0,
        help="perception keypoint coordinate margin (B1); must match the "
        "stage-1 run (auto-read from log.json with --perception_auto_config)",
    )
    parser.add_argument(
        "--coef_delta_decay",
        type=float,
        default=1.0,
        help="damp the predicted delta on appearance-coefficient dims "
        "[2+2*num_velocity:] during free-running rollout (1.0 = unchanged). "
        "<1 biases coefficients toward their last observed values. NOTE: the "
        "D1 diagnostic showed full static copy of appearance HURTS (25.4 vs "
        "26.4 dB at step 44), so use mild values (e.g. 0.5) if at all.",
    )
    parser.add_argument(
        "--freeze_appearance_after",
        type=int,
        default=-1,
        help="lock appearance/scale after this many free-run steps. "
        "-1=off, 0=copy last conditioning appearance for the whole horizon, "
        "14=predict appearance for 14 steps then freeze.",
    )
    parser.add_argument(
        "--state_mean_samples",
        type=int,
        default=1,
        help="if >1 at eval, average this many noisy free-run state rollouts "
        "and decode once (state-space ensemble). Use with --rollout_noise_std.",
    )
    parser.add_argument("--val_pixel_every", type=int, default=0,
                        help="evaluate validation 6->44 pixel metrics every N epochs")
    parser.add_argument("--val_pixel_videos", type=int, default=200,
                        help="maximum validation videos used for pixel model selection")
    parser.add_argument(
        "--val_skip_lpips",
        action="store_true",
        help="skip validation LPIPS (avoids metric-network downloads for PSNR sweeps)",
    )
    parser.add_argument("--skip_test_after_train", action="store_true",
                        help="do not evaluate test checkpoints after screening training")
    parser.add_argument("--eval_only_checkpoint", type=str, default="",
                        help="evaluate only this checkpoint filename inside weight_path")
    # Ignore empty tokens from pasted "\ " line-continuations.
    argv = [a for a in sys.argv[1:] if a.strip()]
    return parser.parse_args(argv)


args = get_args()
if args.inertial_xy_update:
    args.transition_mode = "acceleration_shift"
elif args.enforce_kinematic_velocity:
    args.transition_mode = "history_shift"
if args.windows_per_video < 1:
    raise ValueError("--windows_per_video must be >= 1")
if args.grad_accum_steps < 1:
    raise ValueError("--grad_accum_steps must be >= 1")
if not 0.0 <= args.collision_sample_prob <= 1.0:
    raise ValueError("--collision_sample_prob must be in [0, 1]")
if args.temporal_late_weight <= 0:
    raise ValueError("--temporal_late_weight must be > 0")
if args.contact_radius <= 0:
    raise ValueError("--contact_radius must be > 0")
if args.rotation_step_max < 0 or args.scale_step_max < 0:
    raise ValueError("bounded step maxima must be >= 0")
if not 0.0 <= args.velocity_xy_blend <= 1.0:
    raise ValueError("--velocity_xy_blend must be in [0, 1]")
if args.calibrator_scale <= 0:
    raise ValueError("--calibrator_scale must be > 0")
if args.adapter_only and not args.long_horizon_calibrator:
    raise ValueError("--adapter_only requires --long_horizon_calibrator")
if args.contact_adapter_only and not args.contact_adapter:
    raise ValueError("--contact_adapter_only requires --contact_adapter")
if args.temporal_adapter_only and not args.temporal_adapter:
    raise ValueError("--temporal_adapter_only requires --temporal_adapter")
if args.rollout_corrector_only and not args.rollout_corrector:
    raise ValueError("--rollout_corrector_only requires --rollout_corrector")
if args.temporal_history < 1:
    raise ValueError("--temporal_history must be >= 1")
if args.corrector_hidden_size < 1 or args.corrector_layers < 1:
    raise ValueError("corrector hidden size and layers must be >= 1")
if args.corrector_heads < 1:
    raise ValueError("--corrector_heads must be >= 1")
if args.corrector_hidden_size % args.corrector_heads:
    raise ValueError("--corrector_hidden_size must be divisible by --corrector_heads")
if args.corrector_scale <= 0:
    raise ValueError("--corrector_scale must be > 0")
if args.temporal_adapter and args.Inference_model != "predictor_module_with_appr":
    raise ValueError(
        "--temporal_adapter requires --Inference_model predictor_module_with_appr"
    )
_dynamics_only_modes = (
    int(args.adapter_only)
    + int(args.contact_adapter_only)
    + int(args.temporal_adapter_only)
    + int(args.rollout_corrector_only)
)
if _dynamics_only_modes > 1:
    raise ValueError("dynamics adapter-only modes are mutually exclusive")
if args.decoder_only and _dynamics_only_modes:
    raise ValueError(
        "--decoder_only is mutually exclusive with dynamics adapter-only modes"
    )
if args.decoder_only:
    args.unfreeze_decoder = True
if args.static_residual_strength < 0:
    raise ValueError("--static_residual_strength must be >= 0")
if args.static_residual_threshold < 0:
    raise ValueError("--static_residual_threshold must be >= 0")
if args.static_residual_dilation < 1 or args.static_residual_dilation % 2 == 0:
    raise ValueError("--static_residual_dilation must be a positive odd integer")
if args.absolute_app_head and args.appearance_anchor:
    raise ValueError("--absolute_app_head and --appearance_anchor are mutually exclusive")
if args.freeze_appearance_after < -1:
    raise ValueError("--freeze_appearance_after must be >= -1")
if args.state_mean_samples < 1:
    raise ValueError("--state_mean_samples must be >= 1")
print(args)

assert args.num_velocity < args.given_frames

if args.resolution == 0:
    args.resolution = 128 if args.dataset == "obj3d" else 112

# Align perception architecture with the stage-1 run that produced the weights
# (128px runs use num_filter=64 / std=0.15, which differ from CoPhy defaults).
if args.perception_auto_config and args.Perception_model_weight_path:
    _cfg_json = os.path.join(
        os.path.dirname(args.Perception_model_weight_path), "log.json"
    )
    if os.path.isfile(_cfg_json):
        with open(_cfg_json) as _f:
            _cfg = json.load(_f)
        # NOTE: std is intentionally NOT auto-read. train_derendering.py did not
        # pass --std into Derendering before 2026-09-03, so every existing
        # stage-1 checkpoint effectively used std=0.1 regardless of log.json.
        # filter_size / use_pixelshuffle are safe: old logs record the defaults
        # (5 / absent->False), which matches how those checkpoints were built.
        for _k in ("n_keypoints", "n_coefficients", "num_filter", "resolution",
                   "filter_size", "use_pixelshuffle", "kpt_margin",
                   "shared_object_decoder"):
            if _k in _cfg and getattr(args, _k) != _cfg[_k]:
                print(
                    "perception_auto_config: %s %s -> %s"
                    % (_k, getattr(args, _k), _cfg[_k])
                )
                setattr(args, _k, _cfg[_k])
        if "fg_region_weight" in _cfg and "std" in _cfg \
                and args.std != _cfg["std"]:
            print("perception_auto_config: std %s -> %s"
                  % (args.std, _cfg["std"]))
            args.std = _cfg["std"]
    else:
        print("perception_auto_config: no log.json next to weight, keep CLI args")

if args.seed:  # ?????????????
    random.seed(args.seed)
    torch.manual_seed(args.seed)  # ?CPU?????????????????????????????????????
    torch.cuda.manual_seed(args.seed)  # ?????GPU????????????
    torch.backends.cudnn.deterministic = True

MSE = nn.MSELoss()
use_cuda = True
device = torch.device(
    f"cuda:{args.gpu}" if torch.cuda.is_available() and use_cuda else "cpu"
)


# ?????????
datasets = {
    "blocktowerCF": blocktowerCF_Video,
    "ballsCF": ballsCF_Video,
}
_need_frames = args.given_frames + args.horizon
# Random-offset training needs the whole OBJ3D episode cached so windows can
# be sliced after perception. The T100 cache tag keeps caches disjoint.
_cache_frames = _need_frames
if args.random_window:
    if args.dataset == "obj3d":
        _cache_frames = 100
    else:
        print("WARNING: --random_window is only wired for OBJ3D; ignoring")
_loader_kw = dict(
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

if args.dataset == "obj3d":
    # DDLP protocol: 128x128, 50-frame windows (condition 6, predict 44).
    from Dataloader.OBJ3D_Loader import OBJ3D_Video as _NewVideo
    _root = args.obj3d_root

    def _make_ds(mode):
        return _NewVideo(
            mode=mode,
            resolution=args.resolution,
            sampling_mode="full",
            root=_root,
            max_frames=_cache_frames if mode in ("train", "val") else _need_frames,
        )

    _train_split_name, _val_split_name = "train", "val"
    train_dataset = _make_ds("train")
    val_dataset = _make_ds("val")
    test_dataset = _make_ds("test")
else:
    _train_split_name, _val_split_name = "train_large", "val"
    train_dataset = datasets[args.dataset](
        mode="train_large",
        resolution=args.resolution,
        sampling_mode="full",
        env=args.dataset,
        max_frames=_need_frames,
    )
    val_dataset = datasets[args.dataset](
        mode="val",
        resolution=args.resolution,
        sampling_mode="full",
        env=args.dataset,
        max_frames=_need_frames,
    )
    test_dataset = datasets[args.dataset](
        mode="test",
        resolution=args.resolution,
        sampling_mode="full",
        env=args.dataset,
        max_frames=_need_frames,
    )

if args.max_videos > 0:
    # Smoke-test subset; the perception cache path is tagged below to avoid
    # poisoning the full-data cache.
    train_dataset = torch.utils.data.Subset(
        train_dataset, list(range(min(args.max_videos, len(train_dataset))))
    )
    val_dataset = torch.utils.data.Subset(
        val_dataset, list(range(min(args.max_videos, len(val_dataset))))
    )
    test_dataset = torch.utils.data.Subset(
        test_dataset, list(range(min(args.max_videos, len(test_dataset))))
    )

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batchsize,
    shuffle=True,
    drop_last=True,
    **_loader_kw,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    drop_last=True,
    **_loader_kw,
)
val_eval_dataloader = DataLoader(
    val_dataset,
    batch_size=1,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=False,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=True,
)

# ???????(?????????)
K = args.n_keypoints
# 2d ???? + 1d ?? scale ? + n_coefficients(??????????) +(??? ??????????t?????????)
S = 2 + 1 + args.n_coefficients + 2 * args.num_velocity
if args.n_coefficients == 0:
    S = 2 + 2 * args.num_velocity
    if args.use_scale:
        S += 1
print("object representation shape:", S)
# S = 2

Perception_weight_path = args.Perception_model_weight_path
if args.Perception_model == "derendering":
    if not Perception_weight_path:
        if args.n_coefficients == 4:
            weight_path_dict = {
                "ballsCF": "src/KeypointNet/ballsCF/2023-02-02/06-54/best_val_ballsCF_4kpts.nn",
                "blocktowerCF": "src/KeypointNet/blocktower_4kpts.nn",
            }
        else:
            weight_path_dict = {
                "ballsCF": "src/KeypointNet/ballsCF/2023-02-01/13:06/best_val_ballsCF_4kpts.nn",
                "blocktowerCF": "src/KeypointNet/blocktower_4kpts_nocoef.nn",
            }
        Perception_weight_path = weight_path_dict[args.dataset]
    Perception_model = Derendering(
        n_keypoints=args.n_keypoints,
        mode="fixed",
        n_coefficients=args.n_coefficients,
        use_scale=args.use_scale,
        device=device,
        std=args.std,
        img_width=args.resolution,
        num_filter=args.num_filter,
        filter_size=args.filter_size,
        use_pixelshuffle=args.use_pixelshuffle,
        kpt_margin=args.kpt_margin,
        shared_object_decoder=args.shared_object_decoder,
    ).to(device)

else:
    Perception_weight_path = (
        "src/KeypointNet/blocktowerCF/"
        + args.Perception_model_weight_path
        + "/best_val_blocktowerCF_4kpts.nn"
    )
    # Perception_weight_path = 'src/KeypointNet/blocktowerCF/2023-01-14/17:59/best_val_blocktowerCF_4kpts.nn'
    if args.Perception_model == "derendering_with_rotate":
        Perception_model = Derendering_with_rotate(
            n_keypoints=args.n_keypoints,
            std=args.std,
            n_coefficients=args.n_coefficients,
            is_depth=False,
            device=device,
        ).to(device)
    else:
        Perception_model = Derendering_newest(
            n_keypoints=args.n_keypoints,
            std=args.std,
            n_coefficients=args.n_coefficients,
            is_depth=False,
            device=device,
        ).to(device)
args.Perception_model_weight_path = Perception_weight_path
print(args.Perception_model, Perception_weight_path)
Perception_model.load_state_dict(
    torch.load(Perception_weight_path, map_location=device)
)
if args.decoder_checkpoint:
    Perception_model.decoder.load_state_dict(
        torch.load(args.decoder_checkpoint, map_location=device)
    )
    print("Loaded explicit decoder checkpoint:", args.decoder_checkpoint)
Perception_model.eval()
for _p in Perception_model.parameters():
    _p.requires_grad = False

"""
    # ???12-10 00:06??14:56 (blocktowerCF)?? 22:32(balls)????Delta_Module_newest
    # ???????? Delta_Module
    # dynamics_model = Delta_Module(
    #     state_size=S + 16, target_size=S, n_gn=args.num_gn, emb_size=128, hidden_size=128).to(device)
    # # dynamics_model = Delta_Module_newest(
    #     state_size=S + 16, target_size=S, n_gn=args.num_gn, emb_size=128, hidden_size=128).to(device)
#     dynamics_model = predictor_module(
#         input_dim=S * (args.pos_emb*2+1), appr_dim=16, output_dim=S, is_rnn=args.is_rnn).to(device)  # ??????? ??????nerf?????????~~~
# else:
#     # dynamics_model = Delta_Module(
#     #     state_size=S, target_size=S, n_gn=args.num_gn, emb_size=128, hidden_size=128).to(device)
#     # dynamics_model = Delta_Module_newest(
#     #     state_size=S, target_size=S, n_gn=args.num_gn, emb_size=128, hidden_size=128).to(device)
"""
if args.consider_appr:
    appr_dim = 16
    # appr_dim = 16 * args.n_coefficients
else:
    appr_dim = 0
print("appr_dim:", appr_dim)
Inference_model_dict = {
    "predictor_module": predictor_module,
    "predictor_module_with_appr": predictor_module_with_appr,
    "predictor_module_apprmap": predictor_module_apprmap,
    "predictor_module_mlp": predictor_module_mlp,
}
_dynamics_kwargs = dict(
    input_dim=S * (args.pos_emb * 2 + 1),  # pos_emb=0
    appr_dim=appr_dim,
    output_dim=S,
    num_gn=args.num_gn,
    embed_size=args.embed_size,  # 128
    hidden_size=args.hidden_size,  # 256
    is_rnn=args.is_rnn,  # false
    num_layer=args.num_layer,
)
if args.Inference_model == "predictor_module_with_appr":
    _dynamics_kwargs.update(
        use_contact_adapter=args.contact_adapter,
        num_velocity=args.num_velocity,
        contact_radius=args.contact_radius,
        use_long_horizon_calibrator=args.long_horizon_calibrator,
        calibrator_scope=args.calibrator_scope,
        calibrator_scale=args.calibrator_scale,
        use_temporal_adapter=args.temporal_adapter,
        temporal_history=args.temporal_history,
    )
dynamics_model = Inference_model_dict[args.Inference_model](**_dynamics_kwargs).to(device)
if args.rollout_corrector:
    dynamics_model.rollout_corrector = WholeRolloutTrajectoryCorrector(
        state_dim=S,
        hidden_size=args.corrector_hidden_size,
        num_layers=args.corrector_layers,
        num_heads=args.corrector_heads,
        correction_scale=args.corrector_scale,
    ).to(device)
para_num = count_parameters(dynamics_model)
print("dynamics_model parameter = ", para_num)
# dynamics_model = predictor_module(
#     input_dim=S * (args.pos_emb*2+1), appr_dim=appr_dim, output_dim=S, num_gn=args.num_gn, is_rnn=args.is_rnn).to(device)


_dyn_lpips_fn = None
_dyn_ssim_fn = None


def _load_dynamics_state(pt_path):
    _has_optional_adapter = (
        args.contact_adapter
        or args.long_horizon_calibrator
        or args.temporal_adapter
        or args.rollout_corrector
    )
    result = dynamics_model.load_state_dict(
        torch.load(pt_path, map_location=device),
        strict=not _has_optional_adapter,
    )
    if _has_optional_adapter and (result.missing_keys or result.unexpected_keys):
        allowed_prefixes = []
        if args.contact_adapter:
            allowed_prefixes.append("contact_adapter.")
        if args.long_horizon_calibrator:
            allowed_prefixes.append("long_horizon_calibrator.")
        if args.temporal_adapter:
            allowed_prefixes.append("temporal_adapter.")
        if args.rollout_corrector:
            allowed_prefixes.append("rollout_corrector.")
        allowed_missing = [
            key for key in result.missing_keys
            if any(key.startswith(prefix) for prefix in allowed_prefixes)
        ]
        if len(allowed_missing) != len(result.missing_keys) or result.unexpected_keys:
            raise RuntimeError(
                "incompatible dynamics checkpoint: missing=%s unexpected=%s"
                % (result.missing_keys, result.unexpected_keys)
            )
        print("Initialized optional adapters from zero-output initialization")


def _get_dyn_lpips_fn():
    """Lazily build the LPIPS net for the predicted-frame perceptual loss.

    Inputs are in [-1, 1]. Returns None (with a warning) if unavailable.
    """
    global _dyn_lpips_fn
    if _dyn_lpips_fn is None:
        try:
            if args.metric_loss_backend == "piqa":
                from piqa import LPIPS as PIQALPIPS
                _dyn_lpips_fn = PIQALPIPS(
                    network=args.dyn_perceptual_net, reduction="none"
                ).to(device)
            else:
                _dyn_lpips_fn = lpips.LPIPS(
                    net=args.dyn_perceptual_net, verbose=False
                ).to(device)
            for _p in _dyn_lpips_fn.parameters():
                _p.requires_grad_(False)
        except Exception as e:
            print("WARNING: lpips unavailable (%s); dyn perceptual loss disabled" % e)
            _dyn_lpips_fn = False
    return _dyn_lpips_fn if _dyn_lpips_fn is not False else None


def _get_dyn_ssim_fn():
    global _dyn_ssim_fn
    if _dyn_ssim_fn is None:
        try:
            from piqa import SSIM as PIQASSIM
            _dyn_ssim_fn = PIQASSIM(reduction="none").to(device)
            for _p in _dyn_ssim_fn.parameters():
                _p.requires_grad_(False)
        except Exception as e:
            print("WARNING: piqa SSIM unavailable (%s); SSIM loss disabled" % e)
            _dyn_ssim_fn = False
    return _dyn_ssim_fn if _dyn_ssim_fn is not False else None


def _needs_rgb_cache():
    return (
        args.dyn_perceptual_weight > 0
        or args.ssim_loss_weight > 0
        or args.rgb_loss_weight > 0
        or args.decoder_recon_weight > 0
        or args.val_pixel_every > 0
    )


class CachedPerceptionDataset(Dataset):
    """Frozen-perception outputs. Same content as running Derendering on rgb.

    window > 0 slices pos to a sub-window of that length: offset 0 by default
    (identical to before when the cache already holds exactly `window` frames),
    or a random offset per access when random_window=True (train split only).
    feat stays anchored at episode frame 0, matching the eval protocol.

    rgb (optional, uint8 [N,T,C,H,W]) carries the GT frames for the
    predicted-frame perceptual loss; it is sliced with the SAME offset as pos.
    """

    def __init__(
        self, feat, pos, window=0, random_window=False, rgb=None,
        windows_per_video=1, collision_sample_prob=0.0, window_max_frame=0,
    ):
        if windows_per_video < 1:
            raise ValueError("windows_per_video must be >= 1")
        self.feat = feat
        self.pos = pos
        self.window = window
        self.random_window = random_window
        self.rgb = rgb
        self.windows_per_video = windows_per_video if random_window else 1
        self.collision_sample_prob = collision_sample_prob if random_window else 0.0
        self.window_max_frame = window_max_frame
        self.collision_scores = None
        if self.collision_sample_prob > 0 and self.window:
            self.collision_scores = []
            # Per-offset score: strongest keypoint acceleration inside a
            # candidate window. It is computed once, not in every __getitem__.
            for video_pos in self.pos:
                xy = video_pos[..., :2].float()
                accel = (xy[2:] - 2 * xy[1:-1] + xy[:-2]).square().sum(-1)
                accel = accel.max(-1).values  # [T-2], strongest object
                n_offsets = self._n_offsets(int(video_pos.shape[0]))
                scores = torch.empty(n_offsets, dtype=torch.float32)
                span = max(1, self.window - 2)
                for t0 in range(n_offsets):
                    scores[t0] = accel[t0 : t0 + span].max()
                self.collision_scores.append(scores)

    def _n_offsets(self, n_frames):
        """Number of legal window starts, honouring --train_window_max_frame.

        OBJ3D episodes are 100 frames but the public protocol only ever
        looks at the first given_frames+44. Capping the last usable frame keeps
        training windows inside the same regime.
        """
        usable = n_frames
        if self.window_max_frame > 0:
            usable = min(usable, self.window_max_frame)
        return max(1, usable - self.window + 1)

    def __len__(self):
        return int(self.feat.shape[0]) * self.windows_per_video

    def __getitem__(self, item):
        video_idx = item // self.windows_per_video
        replica_idx = item % self.windows_per_video
        pos = self.pos[video_idx]
        rgb = None if self.rgb is None else self.rgb[video_idx]
        if self.window and pos.shape[0] > self.window:
            if self.random_window:
                n_offsets = self._n_offsets(pos.shape[0])
                # Give each replica a disjoint temporal stratum. This retains
                # stochastic windows while preventing same-video replicas from
                # repeatedly sampling the same region within an epoch.
                lo = replica_idx * n_offsets // self.windows_per_video
                hi = (replica_idx + 1) * n_offsets // self.windows_per_video
                hi = max(lo + 1, hi)
                if (
                    self.collision_scores is not None
                    and float(torch.rand(())) < self.collision_sample_prob
                ):
                    weights = self.collision_scores[video_idx][lo:hi].clamp_min(0)
                    weights = weights + weights.mean().clamp_min(1e-8) * 0.05
                    t0 = lo + int(torch.multinomial(weights, 1))
                else:
                    t0 = int(torch.randint(lo, hi, (1,)))
            else:
                t0 = 0
            pos = pos[t0 : t0 + self.window]
            if rgb is not None:
                rgb = rgb[t0 : t0 + self.window]
        out = {"feat": self.feat[video_idx], "pos": pos}
        if rgb is not None:
            out["rgb"] = rgb
        return out


def _perception_cache_path(split):
    w = args.Perception_model_weight_path
    wtag = os.path.basename(os.path.dirname(w)) + "_" + os.path.basename(w)
    if args.max_videos > 0:
        wtag = "smoke%d_" % args.max_videos + wtag
    if args.n_coefficients == 0:
        name = "%s_%s_T%d_k%d_c0_s%d_%s.pt" % (
            args.dataset,
            split,
            _cache_frames,
            args.n_keypoints,
            1 if getattr(args, "use_scale", False) else 0,
            wtag,
        )
    else:
        name = "%s_%s_T%d_k%d_c%d_%s.pt" % (
            args.dataset,
            split,
            _cache_frames,
            args.n_keypoints,
            args.n_coefficients,
            wtag,
        )
    if _needs_rgb_cache():
        # GT frames are cached alongside feat/pos for the perceptual loss;
        # keep these caches disjoint from the feat/pos-only ones.
        name = name[:-3] + "_rgb.pt"
    return os.path.join("new_output", args.dataset, "perception_cache", name)


@torch.no_grad()
def build_or_load_perception_cache(video_dataset, cache_path, batch_size=16):
    if os.path.isfile(cache_path):
        print("Load perception cache:", cache_path)
        data = torch.load(cache_path, map_location="cpu")
        return data["feat"], data["pos"], data.get("rgb")

    Perception_model.eval()
    loader = DataLoader(
        video_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    want_rgb = _needs_rgb_cache()
    feats, poss, rgbs = [], [], []
    for x in tqdm(loader, desc="Precompute perception"):
        rgb = x["rgb_ab"].to(device, non_blocking=True)
        rgb = rgb[:, :_cache_frames]
        B, T, C, H, W = rgb.shape
        _anchor = 0 if args.feature_anchor == "first" else min(
            args.given_frames - 1, rgb.shape[1] - 1
        )
        feat = Perception_model.get_features_only(rgb[:, _anchor])
        state_1, state_2 = Perception_model.get_keypoints_only(
            rgb.reshape(-1, C, H, W)
        )
        pos = torch.cat([state_1, state_2], dim=-1).view(B, T, state_1.shape[1], -1)
        feats.append(feat.cpu())
        poss.append(pos.cpu())
        if want_rgb:
            # uint8 keeps the full-episode cache ~4x smaller; converted back
            # to float [-1,1] in _cached_kwargs.
            rgbs.append(((rgb.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu())
    feat = torch.cat(feats, 0)
    pos = torch.cat(poss, 0)
    data = {"feat": feat, "pos": pos}
    if want_rgb:
        data["rgb"] = torch.cat(rgbs, 0)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(data, cache_path)
    print("Saved perception cache:", cache_path, tuple(feat.shape), tuple(pos.shape))
    return feat, pos, data.get("rgb")


def _filter_cache_by_count(feat, pos, rgb, split, count_spec):
    if not count_spec:
        return feat, pos, rgb
    if not args.episode_count_file:
        raise ValueError("--train_counts/--val_counts require --episode_count_file")
    allowed = {int(value) for value in count_spec.split(",") if value.strip()}
    with open(args.episode_count_file) as file:
        count_data = json.load(file)
    labels = count_data["splits"][split]["labels"]
    episode_ids = sorted(int(key) for key in labels)
    if len(episode_ids) != len(feat):
        raise ValueError(
            "count labels/cache mismatch for %s: %d labels vs %d videos"
            % (split, len(episode_ids), len(feat)))
    indices = [
        index for index, episode in enumerate(episode_ids)
        if int(labels[str(episode)]) in allowed]
    if not indices:
        raise ValueError("count filter %s produced empty %s split"
                         % (sorted(allowed), split))
    indices = torch.tensor(indices, dtype=torch.long)
    print("count filter %s: %d -> %d videos, counts=%s"
          % (split, len(feat), len(indices), sorted(allowed)))
    return (
        feat.index_select(0, indices),
        pos.index_select(0, indices),
        None if rgb is None else rgb.index_select(0, indices),
    )


def _cached_kwargs(x):
    if "feat" not in x:
        return {}
    kw = {
        "cached_feat": x["feat"].to(device, non_blocking=True),
        "cached_pos": x["pos"].to(device, non_blocking=True),
    }
    if "rgb" in x:
        kw["cached_rgb"] = x["rgb"].to(device, non_blocking=True).float() / 127.5 - 1.0
    return kw


def _batch_rgb(x):
    if "rgb_ab" not in x:
        return None
    return x["rgb_ab"].to(device, non_blocking=True)[
        :, 0 : args.given_frames + args.horizon
    ]


def train_one_batch(
    rgb,
    Perception_model,
    model,
    args,
    p=0,
    batch_idx=1,
    epoch_idx=1,
    generate_img=False,
    df=1.0,
    cached_feat=None,
    cached_pos=None,
    cached_rgb=None,
):
    Perception_model.eval()

    # 1. Perception is frozen (optimizer only updates dynamics).
    if cached_feat is not None:
        inital_feature_map = cached_feat
        pos = cached_pos
        B, T = pos.shape[0], pos.shape[1]
    else:
        B, T, C, H, W = rgb.shape
        with torch.no_grad():
            _anchor = 0 if args.feature_anchor == "first" else min(
                args.given_frames - 1, rgb.shape[1] - 1
            )
            inital_feature_map = Perception_model.get_features_only(rgb[:, _anchor])
            state_1, state_2 = Perception_model.get_keypoints_only(
                rgb.contiguous().view(-1, C, H, W)
            )
        pos = torch.cat([state_1, state_2], dim=-1)
        pos = pos.view(B, T, pos.shape[-2], -1)
    # ? ?????????? ?????????????????????? ?? pos[...,2:] *2 - 1 ?????????????
    # pos[..., 2:] = pos[..., 2:] * 2 - 1  # ????????? ??1/8 22:43???
    K = pos.shape[-2]
    # pos = pos[..., 0:2]  # ????2d???? ??????????????????????
    # pos = pos[..., 0:-1] # ??????????? --- ???(0.5d ?????????)
    new_pos = creat_state_squence(pos, n=args.num_velocity)
    if args.n_coefficients == 0 and not args.use_scale:
        new_pos = new_pos[..., 0:-1]  # drop unused extractor scale for pos-only
    # Dynamics-side component ablation: zero the target dims but keep S unchanged.
    # Layout for n_coefficients > 0: [x, y, v_1..v_{2*tau}, r_1..r_C, s]
    if args.n_coefficients > 0:
        _ori_start = 2 + 2 * args.num_velocity
        _ori_end = _ori_start + args.n_coefficients
        if args.mask_orientation:
            new_pos[..., _ori_start:_ori_end] = 0.0
        if args.mask_scale:
            new_pos[..., -1] = 0.0
    input_state = new_pos[:, 0 : args.given_frames - args.num_velocity]
    target_state = new_pos[:, args.given_frames - args.num_velocity :]

    # 2. ????????
    predict_list = []
    delta_list = []
    observed_history = []
    # Encoder ???? ????
    for i in range(input_state.shape[1] - 1):
        cur_obj = input_state[:, i]
        observed_history.append(cur_obj)
        temporal_kwargs = (
            {"state_history": torch.stack(observed_history, dim=1)}
            if args.temporal_adapter
            else {}
        )
        delta_state = model(
            inpt=cur_obj,
            args=args,
            env_features=inital_feature_map,
            pos_dim=args.pos_emb,
            first_timestep=(i == 0),
            **temporal_kwargs,
        )
        next_state = _transition_state(cur_obj, delta_state)
        predict_list.append(next_state)
        delta_list.append(delta_state)
    # Decoder ???? ?????
    decoder_input = input_state[:, -1]
    rollout_history = list(input_state.unbind(dim=1))
    _app_start = 2 + 2 * args.num_velocity
    app_anchor = (
        decoder_input[..., _app_start:] if args.appearance_anchor else None
    )
    # Training noise must not reach the validation rollout, or it corrupts the
    # best_val_psnr selection with a single noisy sample. State-mean eval is
    # the exception: those extra samples are supposed to be noisy.
    _noise_on = model.training or not args.rollout_noise_train_only
    _n_mean = (
        args.state_mean_samples
        if (generate_img and (not model.training) and args.state_mean_samples > 1)
        else 1
    )
    if _n_mean > 1:
        _noise_on = True

    def _maybe_freeze_app(next_state, frozen_app, t):
        if args.freeze_appearance_after < 0:
            return next_state, frozen_app
        if t == args.freeze_appearance_after - 1:
            frozen_app = next_state[..., _app_start:]
        if t >= args.freeze_appearance_after:
            next_state = torch.cat(
                [next_state[..., :_app_start], frozen_app], dim=-1
            )
        return next_state, frozen_app

    def _free_run_once():
        local_predict = []
        local_delta = []
        cur = decoder_input
        hist = list(rollout_history)
        frozen_app = cur[..., _app_start:]
        for t in range(target_state.shape[1]):
            temporal_kwargs = (
                {"state_history": torch.stack(hist, dim=1)}
                if args.temporal_adapter
                else {}
            )
            delta_state = model(
                inpt=cur,
                args=args,
                env_features=inital_feature_map,
                pos_dim=args.pos_emb,
                first_timestep=(t == 0 and input_state.shape[1] == 1),
                rollout_phase=float(t + 1) / max(1, args.horizon),
                **temporal_kwargs,
            )
            next_state = _transition_state(
                cur, _perturb_delta(delta_state, _noise_on), app_anchor=app_anchor
            )
            next_state, frozen_app = _maybe_freeze_app(next_state, frozen_app, t)
            local_predict.append(next_state)
            local_delta.append(delta_state)
            if np.random.random(1) < p:
                cur = target_state[:, t] + torch.randn_like(
                    target_state[:, t], device=target_state.device
                ) * args.noise_std * p * math.sqrt(t)
            else:
                cur = next_state.detach() if args.detach_rollout else next_state
            hist.append(cur)
        return local_predict, local_delta

    if _n_mean == 1:
        tail, tail_delta = _free_run_once()
        predict_list.extend(tail)
        delta_list.extend(tail_delta)
    else:
        tails = []
        for _sample in range(_n_mean):
            tail, tail_delta = _free_run_once()
            tails.append(torch.stack(tail, dim=1))
            if _sample == 0:
                delta_list.extend(tail_delta)
        tail_mean = torch.stack(tails, dim=0).mean(0)
        predict_list.extend(
            [tail_mean[:, step] for step in range(tail_mean.shape[1])]
        )

    """
    if args.consider_appr:
        predict_list = []
        delta_list = []
        # featuremap_per_object = crop_feature_map(inital_feature_map.unsqueeze(
        #     1).repeat(1, K, 1, 1, 1), pos[:, 0, :, 0:2], crop_size=7)
        # crop_size = featuremap_per_object.shape[-1]
        # appr = nn.AdaptiveAvgPool2d(1)(
        #     featuremap_per_object.view(B*K, -1, crop_size, crop_size)).view(B, K, -1).to(rgb.device)
        # # inital_keypoints_map = Perception_model.get_keypoints_map(rgb[:, 0]) # B K 16 7 7
        # predict_list = []
        # b 16 28 28 -> b 16 1 1 -> b 1 16 -> b K 16
        # dynamics module ???? ??? appr_vector
        # appr = nn.AdaptiveAvgPool2d(1)(inital_feature_map).view(
        #     B, 1, -1).repeat(1, K, 1).to(rgb.device)
        # appr = appr.sigmoid # range [0,1] ?????????
        # # Encoder ????
        # for i in range(input_state.shape[1]-1):
        #     cur_obj = torch.cat([input_state[:, i], appr], dim=-1)
        #     delta_state = model(cur_obj)
        #     next_state = delta_state + input_state[:, i]
        #     predict_list.append(next_state)
        # # Dncoder ????
        # decoder_input = input_state[:, -1]
        # use_teacher_forcing = np.random.binomial(1, p, args.horizon)
        # for t in range(target_state.shape[1]):
        #     cur_obj = torch.cat([decoder_input, appr], dim=-1)
        #     delta_state = model(cur_obj)
        #     next_state = delta_state + decoder_input
        #     predict_list.append(next_state)
        #     if use_teacher_forcing[t]:
        #         decoder_input = target_state[:, t] + torch.randn_like(
        #             target_state[:, t], device=target_state.device) * add_noise_std
        #     else:
        #         decoder_input = next_state

        # dynamics module ???? ?? appr_vector
        # Encoder ????
        # print(inital_feature_map.device)
        for i in range(input_state.shape[1]-1):
            cur_obj = input_state[:, i]
            delta_state = model(cur_obj, inital_feature_map,
                                pos_dim=args.pos_emb, first_timestep=(i == 0))
            next_state = delta_state + cur_obj 
            predict_list.append(next_state)
            delta_list.append(delta_state)
        # Decoder ????
        decoder_input = input_state[:, -1]
        use_teacher_forcing = np.random.binomial(1, p, args.horizon)
        for t in range(target_state.shape[1]):
            cur_obj = decoder_input
            delta_state = model(cur_obj, inital_feature_map,
                                pos_dim=args.pos_emb, first_timestep=(t == 0 and input_state.shape[1] == 1))
            next_state = _damp_coef_delta(delta_state) + cur_obj
            predict_list.append(next_state)
            delta_list.append(delta_state)
            if use_teacher_forcing[t]:
                decoder_input = target_state[:, t] + torch.randn_like(
                    target_state[:, t], device=target_state.device) * args.noise_std * p * math.sqrt(t)  # p=0 ????????????p=1???????
            else:
                decoder_input = next_state
    else:
        predict_list = []
        delta_list = []
        # Encoder ????
        for i in range(input_state.shape[1]-1):
            cur_obj = input_state[:, i]
            delta_state = model(cur_obj, pos_dim=args.pos_emb,
                                first_timestep=(i == 0))
            next_state = delta_state + cur_obj
            predict_list.append(next_state)
            delta_list.append(delta_state)
        # Dncoder ????
        decoder_input = input_state[:, -1]
        use_teacher_forcing = np.random.binomial(1, p, args.horizon)
        for t in range(target_state.shape[1]):
            cur_obj = decoder_input
            delta_state = model(cur_obj, pos_dim=args.pos_emb, first_timestep=(
                t == 0 and input_state.shape[1] == 1))
            next_state = _damp_coef_delta(delta_state) + cur_obj
            predict_list.append(next_state)
            delta_list.append(delta_state)
            if use_teacher_forcing[t]:
                decoder_input = target_state[:, t] + torch.randn_like(
                    target_state[:, t], device=target_state.device) * args.noise_std * p  # ????????????
            else:
                decoder_input = next_state
    """
    gt_states = new_pos[:, 1:]
    predicted_states = torch.stack(predict_list, dim=1)
    if args.rollout_corrector:
        # The GNN trajectory above is produced exactly as before. In
        # corrector-only mode it is treated as frozen input, while gradients
        # through the corrector output remain intact for state and RGB losses.
        _corrector_base = (
            predicted_states.detach()
            if args.rollout_corrector_only
            else predicted_states
        )
        predicted_states = model.rollout_corrector.correct(
            input_state.detach(), _corrector_base
        )

    # Per-component state errors for ablation analysis (not part of training loss).
    if args.n_coefficients > 0:
        _ori_start = 2 + 2 * args.num_velocity
        _ori_end = _ori_start + args.n_coefficients
        loss_ori = MSE(
            predicted_states[..., _ori_start:_ori_end],
            gt_states[..., _ori_start:_ori_end],
        )
        loss_scale = MSE(predicted_states[..., -1], gt_states[..., -1])
    elif args.use_scale:
        loss_ori = torch.zeros((), device=predicted_states.device)
        loss_scale = MSE(predicted_states[..., -1], gt_states[..., -1])
    else:
        loss_ori = torch.zeros((), device=predicted_states.device)
        loss_scale = torch.zeros((), device=predicted_states.device)

    # initial_state = new_pos[:, 0] #????????gt ?? B TAU K S

    # loss, loss_pos, loss_vel, loss_coef, loss_exist = get_loss_with_speed_exist(
    #     predicted_states, gt_states, pos_weight=args.pos_weight)
    if args.n_coefficients != 0:
        loss, loss_pos, loss_vel, loss_coef, loss_exist = get_loss_with_df(
            predicted_states,
            gt_states,
            pos_weight=args.pos_weight,
            df=df,
            n_coef=args.n_coefficients + 1,
            xy_weight=None if args.xy_loss_weight < 0 else args.xy_loss_weight,
            velocity_weight=(
                None if args.velocity_loss_weight < 0
                else args.velocity_loss_weight
            ),
            appearance_weight=args.appearance_loss_weight,
            scale_weight=args.scale_loss_weight,
            temporal_late_weight=args.temporal_late_weight,
        )  # ??????????????
    else:
        loss, loss_pos, loss_vel, loss_coef, loss_exist = get_loss_without_coef(
            predicted_states, gt_states, pos_weight=args.pos_weight, df=df
        )
    predicted_delta_states = torch.stack(delta_list, dim=1)
    gt_delta_state = new_pos[:, 1:] - new_pos[:, 0:-1]
    delta_loss = MSE(gt_delta_state, predicted_delta_states) * 1e3
    loss = loss + args.delta_loss_weight * delta_loss

    # Predicted-frame perceptual loss (DDLP beta_dyn_rec style). Opt-in via
    # --dyn_perceptual_weight; only computed during training. Gradients flow
    # through the differentiable renderer (create_img) into the predicted
    # states, i.e. into the dynamics model. With --unfreeze_decoder the
    # perception decoder also receives gradients; everything else of the
    # perception model stays frozen.
    loss_pix = torch.zeros((), device=predicted_states.device)
    loss_ssim = torch.zeros((), device=predicted_states.device)
    loss_rgb = torch.zeros((), device=predicted_states.device)
    loss_decoder_recon = torch.zeros((), device=predicted_states.device)
    _use_render_loss = (
        args.dyn_perceptual_weight > 0
        or args.ssim_loss_weight > 0
        or args.rgb_loss_weight > 0
        or args.decoder_recon_weight > 0
    )
    if _use_render_loss and model.training and not generate_img:
        _gt_src = rgb if rgb is not None else cached_rgb
        if _gt_src is None:
            if batch_idx == 0 and epoch_idx == 0:
                print("WARNING: render loss requested but no rgb available; skipped")
        else:
            n_f = args.dyn_perceptual_frames
            if n_f <= 0 or n_f > args.horizon:
                n_f = args.horizon
            if args.render_frame_sampling == "uniform" and n_f < args.horizon:
                _frame_idx = torch.linspace(
                    0, args.horizon - 1, n_f, device=predicted_states.device
                ).round().long()
            else:
                _frame_idx = torch.arange(
                    args.horizon - n_f, args.horizon,
                    device=predicted_states.device,
                )
            _pred_idx = _frame_idx + predicted_states.shape[1] - args.horizon
            _pred = predicted_states.index_select(1, _pred_idx)
            _rec_xy = _pred[..., 0:2].reshape(-1, K, 2)
            if args.n_coefficients > 0:
                _rec_app = _pred[..., -args.n_coefficients - 1 :].reshape(
                    -1, K, args.n_coefficients + 1
                )
            elif args.use_scale:
                _rec_app = _pred[..., -1:].reshape(-1, K, 1)
            else:
                _rec_app = torch.ones_like(_rec_xy[..., :1])
            _feat = (
                inital_feature_map.unsqueeze(1)
                .repeat(1, n_f, 1, 1, 1)
                .view(-1, *inital_feature_map.shape[1:])
            )
            _rec = Perception_model.create_img(_feat, _rec_xy, _rec_app)
            _rgb_idx = _frame_idx + args.given_frames
            _gt = _gt_src.index_select(1, _rgb_idx).reshape(
                -1, *_gt_src.shape[2:]
            )
            _rec01 = to_piqa_range(_rec)
            _gt01 = to_piqa_range(_gt)
            _late_w = None
            if args.temporal_late_weight != 1.0 and n_f > 1:
                _late_w = torch.linspace(
                    1.0,
                    args.temporal_late_weight,
                    n_f,
                    device=predicted_states.device,
                    dtype=_rec.dtype,
                )
                _late_w = _late_w / _late_w.mean()
            def _mean_over_frames(per_sample):
                flat = per_sample.reshape(-1)
                if flat.numel() != predicted_states.shape[0] * n_f:
                    return flat.mean()
                per_frame = flat.view(predicted_states.shape[0], n_f)
                if _late_w is None:
                    return per_frame.mean()
                return (per_frame * _late_w).mean()
            if args.dyn_perceptual_weight > 0:
                _lp = _get_dyn_lpips_fn()
                if _lp is not None:
                    if args.metric_loss_backend == "piqa":
                        loss_pix = _mean_over_frames(_lp(_rec01, _gt01).reshape(-1))
                    else:
                        loss_pix = _mean_over_frames(_lp(_rec, _gt).reshape(-1))
                    loss = loss + args.dyn_perceptual_weight * loss_pix
            if args.ssim_loss_weight > 0:
                _ssim = _get_dyn_ssim_fn()
                if _ssim is not None:
                    loss_ssim = _mean_over_frames(
                        (1.0 - _ssim(_rec01, _gt01)).reshape(-1)
                    )
                    loss = loss + args.ssim_loss_weight * loss_ssim
            if args.rgb_loss_weight > 0:
                if args.rgb_loss_type == "mse":
                    _rgb_each = (_rec - _gt).square().mean(dim=(1, 2, 3))
                else:
                    _rgb_each = torch.sqrt((_rec - _gt).square() + 1e-6).mean(
                        dim=(1, 2, 3)
                    )
                loss_rgb = _mean_over_frames(_rgb_each)
                loss = loss + args.rgb_loss_weight * loss_rgb
            if args.decoder_recon_weight > 0:
                _gt_state = pos.index_select(1, _rgb_idx)
                _gt_xy = _gt_state[..., :2].reshape(-1, K, 2)
                _gt_app = _gt_state[..., 2:].reshape(
                    -1, K, _gt_state.shape[-1] - 2
                )
                _oracle = Perception_model.create_img(_feat, _gt_xy, _gt_app)
                loss_decoder_recon = torch.sqrt(
                    (_oracle - _gt).square() + 1e-6
                ).mean()
                loss = loss + args.decoder_recon_weight * loss_decoder_recon

    # if args.autoenc_loss:
    #     state_dim = input_state.shape[-1]  # ?????????????????????
    #     auto_enc = model.decoder(model.decoder(input_state.view(-1,K,state_dim)))

    if batch_idx == 0 and epoch_idx == 0:
        if rgb is not None:
            print(rgb.shape)
        # print(inital_feature_map.shape)
        print(pos.shape, new_pos.shape)
        # print(featuremap_per_object.shape)
        # print(appr.shape)
        # print(gt_states.shape)
        print(predicted_states.shape)
    # print(pos[0, :, 0])  # B T K S -> T S ???????????
    # print('loss_dynamic:', loss_vel)
    # print(new_pos == pos)

    if generate_img:
        # ?????????????????
        # test_dataloder B = 1 B 16 28 28  ????????????????

        rec_xy = predicted_states[0, -args.horizon :, :, :2]
        if args.n_coefficients > 0:
            rec_app = predicted_states[0, -args.horizon :, :, -args.n_coefficients - 1 :]
        elif args.use_scale:
            rec_app = predicted_states[0, -args.horizon :, :, -1:]
        else:
            rec_app = torch.ones_like(rec_xy[..., :1])
        rec_rgb = Perception_model.create_img(
            inital_feature_map.repeat(args.horizon, 1, 1, 1),
            rec_xy,
            rec_app,
        )
        _render_gt = rgb if rgb is not None else cached_rgb
        if args.static_residual_strength > 0:
            _cond = _render_gt[:, : args.given_frames]
            _obs_idx = args.given_frames - 1
            _obs_state = pos[:, _obs_idx]
            _obs_rec = Perception_model.create_img(
                inital_feature_map,
                _obs_state[..., :2],
                _obs_state[..., 2:],
            )
            rec_rgb = apply_static_residual(
                rec_rgb,
                _cond,
                _obs_rec,
                strength=args.static_residual_strength,
                motion_threshold=args.static_residual_threshold,
                dilation=args.static_residual_dilation,
            )
        gt_rgb = _render_gt[
            0, args.given_frames : args.given_frames + args.horizon
        ]  # H C H W

        return (
            loss,
            loss_pos,
            loss_vel,
            loss_coef,
            loss_exist,
            loss_ori,
            loss_scale,
            rec_rgb,
            gt_rgb,
        )
    else:
        return (
            loss,
            loss_pos,
            loss_vel,
            loss_coef,
            loss_exist,
            loss_ori,
            loss_scale,
            loss_pix,
        )


def evaludate_per_frame(Perception_model, dynamics_model, dataloader, args, des="Test"):
    loss_fn = lpips.LPIPS(net="alex").to(device)  # alex/vgg
    list_psnr = []
    list_local_psnr = []
    list_ssim = []
    list_rgb_mse = []
    list_lpips = []

    list_loss = []
    list_loss_coef = []
    list_loss_vel = []
    list_loss_position = []
    list_loss_exist = []

    rec_rgb_list = []
    gt_rgb_list = []

    dynamics_model.eval()
    MSE = nn.MSELoss()

    with torch.no_grad():
        for i, x in enumerate(tqdm(dataloader, desc=des)):
            rgb = x["rgb_ab"].to(device, non_blocking=True)
            rgb = rgb[:, 0 : args.given_frames + args.horizon]
            # T C H W
            (
                loss,
                loss_pos,
                loss_vel,
                loss_coef,
                loss_exist,
                _loss_ori,
                _loss_scale,
                rec_seqs,
                gt_seqs,
            ) = train_one_batch(
                rgb, Perception_model, dynamics_model, args=args, p=0, generate_img=True
            )  # B horizon C H W
            rec_rgb_list.append(rec_seqs)
            gt_rgb_list.append(gt_seqs)
            list_loss.append(loss.item())
            list_loss_coef.append(loss_coef.item())
            list_loss_vel.append(loss_vel.item())
            list_loss_position.append(loss_pos.item())
            list_loss_exist.append(loss_exist.item())
        rec_rgb = torch.stack(rec_rgb_list, dim=0)
        gt_rgb = torch.stack(gt_rgb_list, dim=0)
        print("rec shape:", rec_rgb.shape)
        for t in tqdm(range(rec_rgb.shape[1]), desc="Per_frame"):  # ??????????????
            gt_seq = gt_rgb[:, t].cpu().detach().numpy().transpose(0, 2, 3, 1)
            pred_seq = rec_rgb[:, t].cpu().detach().numpy().transpose(0, 2, 3, 1)
            list_psnr.append(
                np.mean(
                    [
                        peak_signal_noise_ratio(
                            gt_seq[b], pred_seq[b] + 1e-4, data_range=2
                        )
                        for b in range(gt_rgb.shape[0])
                    ]
                )
            )
            # B H W C
            if args.dataset in ("ballsCF", "blocktowerCF"):
                # L-PSNR needs the CoDy background plates; not defined for
                # OBJ3D.
                background_mask = remove_background(gt_seq, args.dataset)
                list_local_psnr.append(
                    np.mean(
                        [
                            peak_signal_noise_ratio(
                                gt_seq[b] * background_mask[b],
                                pred_seq[b] * background_mask[b] + 1e-4,
                                data_range=2,
                            )
                            for b in range(gt_rgb.shape[0])
                        ]
                    )
                )
            else:
                list_local_psnr.append(float("nan"))
            list_ssim.append(
                np.mean(
                    [
                        structural_similarity(
                            gt_seq[b], pred_seq[b], data_range=2, channel_axis=-1
                        )
                        for b in range(gt_rgb.shape[0])
                    ]
                )
            )
            list_rgb_mse.append(MSE(gt_rgb[:, t], rec_rgb[:, t]).item())
            list_lpips.append(loss_fn(gt_rgb[:, t], rec_rgb[:, t]).mean().item())
        log_metric = {
            "PSNR": list_psnr,
            "Local_PSNR": list_local_psnr,
            "SSIM": list_ssim,
            "LPIPS": list_lpips,
            "MSE": list_rgb_mse,
        }
        metric_result = pd.DataFrame(log_metric)
        log_dir = os.path.dirname(args.weight_path)
        csv_path = os.path.join(log_dir, "metric_result.csv")
        # metric_result.to_csv(csv_path, index=None, encoding='utf8')
        out = [
            np.mean(list_loss),
            np.mean(list_loss_position),
            np.mean(list_loss_vel),
            np.mean(list_loss_coef),
            np.mean(list_loss_exist),
            np.mean(list_rgb_mse) * 1000,
            np.mean(list_psnr),
            np.mean(list_local_psnr),
            np.mean(list_ssim),
            np.mean(list_lpips),
        ]
    return out


def test(Perception_model, dynamics_model, dataloader, args, des="Test", viz=True,
         export_dir=""):
    # Skip alex LPIPS when ranking by PSNR; avoids torch-hub download hangs.
    loss_fn = None
    if not getattr(args, "val_skip_lpips", False):
        loss_fn = lpips.LPIPS(net="alex").to(device)  # alex/vgg
    list_loss = []
    list_loss_coef = []
    list_loss_vel = []
    list_loss_position = []
    list_loss_exist = []
    list_loss_ori = []
    list_loss_scale = []

    mse_pixel_loss = []
    PSNR_list = []
    SSIM_list = []
    LPIPS_list = []
    Local_PSNR_list = []

    dynamics_model.eval()
    MSE = nn.MSELoss()

    with torch.no_grad():
        for i, x in enumerate(tqdm(dataloader, desc=des)):
            rgb = x["rgb_ab"].to(device, non_blocking=True)
            rgb = rgb[:, 0 : args.given_frames + args.horizon]  # B T C H
            (
                loss,
                loss_pos,
                loss_vel,
                loss_coef,
                loss_exist,
                loss_ori,
                loss_scale,
                rec_rgb,
                gt_rgb,
            ) = train_one_batch(
                rgb, Perception_model, dynamics_model, args=args, p=0, generate_img=True
            )
            if args.hflip_self_ensemble:
                rec_flip = train_one_batch(
                    torch.flip(rgb, [-1]),
                    Perception_model,
                    dynamics_model,
                    args=args,
                    p=0,
                    generate_img=True,
                )[7]
                rec_rgb = 0.5 * (rec_rgb + torch.flip(rec_flip, [-1]))

            list_loss.append(loss.item())
            list_loss_coef.append(loss_coef.item())
            list_loss_vel.append(loss_vel.item())
            list_loss_position.append(loss_pos.item())
            list_loss_exist.append(loss_exist.item())
            list_loss_ori.append(loss_ori.item())
            list_loss_scale.append(loss_scale.item())

            mse_pixel = MSE(gt_rgb, rec_rgb) * 1000
            mse_pixel_loss.append(mse_pixel.item())
            if loss_fn is not None:
                LPIPS_list.append(loss_fn(gt_rgb, rec_rgb).mean().item())
            rec_img = rec_rgb.cpu().detach().numpy().transpose(0, 2, 3, 1)
            gt_img = gt_rgb.cpu().detach().numpy().transpose(0, 2, 3, 1)
            PSNR_list.append(
                np.mean(
                    [
                        peak_signal_noise_ratio(gt_img[b], rec_img[b], data_range=2)
                        for b in range(rec_rgb.shape[0])
                    ]
                )
            )  # ???batch ?????PSNR
            SSIM_list.append(
                np.mean(
                    [
                        structural_similarity(
                            gt_img[b],
                            rec_img[b],
                            data_range=2,
                            channel_axis=-1,
                        )
                        for b in range(rec_rgb.shape[0])
                    ]
                )
            )

            if args.dataset in ("ballsCF", "blocktowerCF"):
                background_mask = remove_background(gt_img, args.dataset)
                Local_PSNR_list.append(
                    np.mean(
                        [
                            peak_signal_noise_ratio(
                                gt_img[b] * background_mask[b],
                                rec_img[b] * background_mask[b] + 1e-4,
                                data_range=2,
                            )
                            for b in range(rec_rgb.shape[0])
                        ]
                    )
                )
            else:
                Local_PSNR_list.append(float("nan"))
            if export_dir:
                # DDLP-protocol export: predicted frames only, [0,1] HWC float32.
                os.makedirs(export_dir, exist_ok=True)
                _ex = x["ex"]
                _ex_id = int(_ex[-1].item()) if torch.is_tensor(_ex) else int(_ex[-1])
                _pred01 = (
                    ((rec_rgb + 1) / 2)
                    .clamp(0, 1)
                    .cpu()
                    .numpy()
                    .transpose(0, 2, 3, 1)
                    .astype(np.float32)
                )
                np.save(os.path.join(export_dir, "%d.npy" % _ex_id), _pred01)
            if viz:
                log_dir = os.path.dirname(args.weight_path)
                log_dir_vis = os.path.join(log_dir, "vis_" + str(args.horizon))
                if not os.path.exists(log_dir_vis):
                    os.makedirs(log_dir_vis)
                video_idx_list = x["ex"]
                # B T C H W
                if (i + 1) % 1 == 0 and video_idx_list[-1].item() in [211, 1615]:
                    # print(torch.max(rgb))
                    # print(torch.max(img_gen))

                    visualize(
                        gt_rgb.unsqueeze(0).cpu().detach(),
                        rec_rgb.unsqueeze(0).cpu().detach(),
                        log_dir_vis,
                        batch_idx="video_" + str(video_idx_list[-1].item()),
                    )
    return (
        np.mean(list_loss),
        np.mean(list_loss_position),
        np.mean(list_loss_vel),
        np.mean(list_loss_coef),
        np.mean(list_loss_exist),
        np.mean(list_loss_ori),
        np.mean(list_loss_scale),
        np.mean(mse_pixel_loss),
        np.mean(PSNR_list),
        np.mean(Local_PSNR_list),
        np.mean(SSIM_list),
        float("nan") if not LPIPS_list else np.mean(LPIPS_list),
    )


def validate(Perception_model, dynamics_model, dataloader, args, des="Validation"):
    list_loss = []
    list_loss_coef = []
    list_loss_vel = []
    list_loss_position = []
    list_loss_exist = []

    dynamics_model.eval()

    with torch.no_grad():
        for i, x in enumerate(tqdm(dataloader, desc=des)):
            (
                loss,
                loss_pos,
                loss_vel,
                loss_coef,
                loss_exist,
                _loss_ori,
                _loss_scale,
                _loss_pix,
            ) = train_one_batch(
                _batch_rgb(x),
                Perception_model,
                dynamics_model,
                args=args,
                p=0,
                **_cached_kwargs(x),
            )  # ??????????????????,?????????1yep
            list_loss.append(loss.detach())
            list_loss_coef.append(loss_coef.detach())
            list_loss_vel.append(loss_vel.detach())
            list_loss_position.append(loss_pos.detach())
            list_loss_exist.append(loss_exist.detach())
    return (
        float(torch.stack(list_loss).mean()),
        float(torch.stack(list_loss_position).mean()),
        float(torch.stack(list_loss_vel).mean()),
        float(torch.stack(list_loss_coef).mean()),
        float(torch.stack(list_loss_exist).mean()),
    )


@torch.no_grad()
def validate_pixel_protocol(
    Perception_model, dynamics_model, dataloader, args, horizon=44
):
    """Validation-only DDLP-caliber metrics used for model selection."""
    from piqa import LPIPS as PIQALPIPS, PSNR as PIQAPSNR, SSIM as PIQASSIM

    psnr_fn = PIQAPSNR(reduction="none").to(device)
    ssim_fn = PIQASSIM(reduction="none").to(device)
    lpips_fn = (
        None
        if args.val_skip_lpips
        else PIQALPIPS(network="vgg", reduction="none").to(device)
    )
    psnr_values, ssim_values, lpips_values = [], [], []
    old_horizon = args.horizon
    dynamics_model.eval()
    Perception_model.eval()
    try:
        args.horizon = horizon
        for i, x in enumerate(tqdm(dataloader, desc="Validation pixels")):
            if i >= args.val_pixel_videos:
                break
            result = train_one_batch(
                None,
                Perception_model,
                dynamics_model,
                args=args,
                p=0,
                generate_img=True,
                **_cached_kwargs(x),
            )
            rec, gt = result[-2], result[-1]
            rec = ((rec + 1) / 2).clamp(0, 1)
            gt = ((gt + 1) / 2).clamp(0, 1)
            psnr_values.append(psnr_fn(rec, gt).cpu())
            ssim_values.append(ssim_fn(rec, gt).cpu())
            if lpips_fn is not None:
                lpips_values.append(lpips_fn(rec, gt).cpu())
    finally:
        args.horizon = old_horizon
    metrics = {
        "PSNR": float(torch.cat(psnr_values).mean()),
        "SSIM": float(torch.cat(ssim_values).mean()),
        "LPIPS_vgg": (
            float(torch.cat(lpips_values).mean()) if lpips_values else float("nan")
        ),
    }
    print("Validation pixel metrics:", metrics)
    return metrics


def train(
    Perception_model,
    dynamics_model,
    train_dataloader,
    val_dataloader,
    args,
    val_pixel_dataloader=None,
):
    now_day = datetime.now().date()
    now_time = datetime.now().time()
    # HH-MM-SS (no colons, Windows-sync safe) + optional run_id; dedup if it still exists.
    run_tag = str(now_time)[0:8].replace(":", "-")
    if args.run_id:
        run_tag = run_tag + "_" + args.run_id
    log_dir = args.output_dir or os.path.join(
        "new_output",
        args.dataset,
        "logs_for_" + args.name,
        str(now_day),
        run_tag,
    )
    if args.output_dir:
        os.makedirs(log_dir, exist_ok=True)
    else:
        _base_log_dir = log_dir
        _dup = 1
        while os.path.exists(log_dir):
            _dup += 1
            log_dir = "%s_%d" % (_base_log_dir, _dup)
        os.makedirs(log_dir)
    print("log_dir:", log_dir)
    json_path = os.path.join(log_dir, "log.json")
    with open(json_path, "w") as f:
        json.dump(vars(args), f, indent=2)  # ?????,indent=2??????

    writer = SummaryWriter(log_dir)
    if args.adapter_only:
        for parameter in dynamics_model.parameters():
            parameter.requires_grad_(False)
        for parameter in dynamics_model.long_horizon_calibrator.parameters():
            parameter.requires_grad_(True)
        print(
            "adapter_only: training %d calibrator parameters"
            % sum(
                parameter.numel()
                for parameter in dynamics_model.parameters()
                if parameter.requires_grad
            )
        )
    if args.contact_adapter_only:
        for parameter in dynamics_model.parameters():
            parameter.requires_grad_(False)
        for parameter in dynamics_model.contact_adapter.parameters():
            parameter.requires_grad_(True)
        print(
            "contact_adapter_only: training %d contact parameters"
            % sum(
                parameter.numel()
                for parameter in dynamics_model.parameters()
                if parameter.requires_grad
            )
        )
    if args.temporal_adapter_only:
        for parameter in dynamics_model.parameters():
            parameter.requires_grad_(False)
        for parameter in dynamics_model.temporal_adapter.parameters():
            parameter.requires_grad_(True)
        print(
            "temporal_adapter_only: training %d temporal parameters"
            % sum(
                parameter.numel()
                for parameter in dynamics_model.parameters()
                if parameter.requires_grad
            )
        )
    if args.rollout_corrector_only:
        for parameter in dynamics_model.parameters():
            parameter.requires_grad_(False)
        for parameter in dynamics_model.rollout_corrector.parameters():
            parameter.requires_grad_(True)
        print(
            "rollout_corrector_only: training %d corrector parameters"
            % sum(
                parameter.numel()
                for parameter in dynamics_model.parameters()
                if parameter.requires_grad
            )
        )
    if args.decoder_only:
        enable_decoder_only(Perception_model, dynamics_model)
        print("decoder_only: dynamics frozen")
    _trainable_dynamics = [
        parameter for parameter in dynamics_model.parameters()
        if parameter.requires_grad
    ]
    _unfreeze_dec = args.unfreeze_decoder and (
        args.dyn_perceptual_weight > 0
        or args.ssim_loss_weight > 0
        or args.rgb_loss_weight > 0
        or args.decoder_recon_weight > 0
    )
    if _unfreeze_dec:
        # Mode B: end-to-end fine-tune of the perception decoder with a small
        # lr; encoder / filter bank / everything else stays frozen.
        _dec_params = list(Perception_model.decoder.parameters())
        for _p in _dec_params:
            _p.requires_grad = True
        _groups = [{"params": _dec_params, "lr": args.decoder_lr}]
        if _trainable_dynamics:
            _groups.insert(0, {"params": _trainable_dynamics, "lr": args.lr})
        optim = torch.optim.Adam(_groups)
        print(
            "unfreeze_decoder: %d decoder params, lr=%g"
            % (sum(int(p.numel()) for p in _dec_params), args.decoder_lr)
        )
    else:
        optim = torch.optim.Adam(_trainable_dynamics, lr=args.lr)

    def _save_ckpt(name):
        torch.save(dynamics_model.state_dict(), os.path.join(log_dir, name))
        if _unfreeze_dec:
            torch.save(
                Perception_model.decoder.state_dict(),
                os.path.join(log_dir, name.replace(".pt", ".decoder.pt")),
            )

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    # sched = torch.optim.lr_scheduler.StepLR(optim, gamma=0.5, step_size=10)
    best_val_loss = math.inf
    best_train_loss = math.inf
    best_val_position_loss = math.inf
    best_train_epoch = 0
    best_val_epoch = 0
    best_val_position_epoch = 0
    best_val_psnr = -math.inf
    best_val_ssim = -math.inf
    best_val_lpips = math.inf
    best_val_psnr_epoch = 0
    best_val_ssim_epoch = 0
    best_val_lpips_epoch = 0
    train_epochsize = len(train_dataloader)

    arg_str_list = [
        "{}={}".format(k, v) for k, v in vars(args).items()
    ]  # ???namespace????? key-value
    # print('------------------------  argument namespace ----------------------')
    # print('arg_str_list:', arg_str_list)
    arg_str = "__".join(arg_str_list)
    writer.add_text("hparams", arg_str)  # ??????????????????????
    step_per_epoch = len(train_dataloader)
    p_gt = 0.0  # ?????? or ???????
    train_step = 0
    for epoch in range(args.epoch):
        train_loss = 0.0
        train_loss_position = 0.0
        train_loss_vel = 0.0
        train_loss_coef = 0.0
        train_loss_exist = 0.0
        train_loss_pix = 0.0

        dynamics_model.train()
        # Perception_model.eval()
        print(
            "=== EPOCH ",
            epoch + 1,
            "/",
            args.epoch,
            " ===",
            "lr=",
            optim.param_groups[0]["lr"],
        )

        if args.discount_factor:  # ?????????epoch??????
            df = linear_anneal(epoch, 1e-20, 1.0, 0, args.df_epoch)
        else:
            df = 1.0
        optim.zero_grad(set_to_none=True)
        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):
            if args.train_mode_verbose == 1:
                p_gt = 1.0
            elif args.train_mode_verbose == 2:
                p_gt = linear_anneal(
                    train_step,
                    1.0,
                    0.0,
                    args.warm_epoch * step_per_epoch,
                    args.decay_epoch * step_per_epoch,
                )
            writer.add_scalar("TRAIN/p", p_gt, train_step + 1)
            train_step += 1

            # if args.discount_factor: # ?????????batchsize??????
            #     df = linear_anneal(
            #     epoch*len(train_dataloader)+i, 1e-20, 1.0, 0, 20*len(train_dataloader))
            # else:
            #     df = 1.0
            # ????, ???????discount_factor?? ????? ??????,??????????????????
            (
                loss,
                loss_pos,
                loss_vel,
                loss_coef,
                loss_exist,
                _loss_ori,
                _loss_scale,
                loss_pix,
            ) = train_one_batch(
                _batch_rgb(x),
                Perception_model,
                dynamics_model,
                args=args,
                p=p_gt,
                batch_idx=i,
                epoch_idx=epoch,
                df=df,
                **_cached_kwargs(x),
            )

            group_start = (i // args.grad_accum_steps) * args.grad_accum_steps
            group_size = min(args.grad_accum_steps, train_epochsize - group_start)
            (loss / group_size).backward()
            if (i + 1) % args.grad_accum_steps == 0 or i + 1 == train_epochsize:
                optim.step()
                optim.zero_grad(set_to_none=True)

            train_loss += loss.detach()
            train_loss_coef += loss_coef.detach()
            train_loss_vel += loss_vel.detach()
            train_loss_position += loss_pos.detach()
            train_loss_exist += loss_exist.detach()
            train_loss_pix += loss_pix.detach()

        train_loss = float(train_loss / train_epochsize)
        train_loss_coef = float(train_loss_coef / train_epochsize)
        train_loss_vel = float(train_loss_vel / train_epochsize)
        train_loss_position = float(train_loss_position / train_epochsize)
        train_loss_exist = float(train_loss_exist / train_epochsize)
        train_loss_pix = float(train_loss_pix / train_epochsize)

        (
            val_loss,
            val_loss_position,
            val_loss_vel,
            val_loss_coef,
            val_loss_exist,
        ) = validate(Perception_model, dynamics_model, val_dataloader, args=args)
        print(
            "Epoch %d  ======> Train  loss = %.4f, loss_pos = %.4f, loss_dynamic = %.4f, loss_coef = %.4f, loss_exist = %.4f, loss_pix = %.4f,"
            % (
                epoch + 1,
                train_loss,
                train_loss_position,
                train_loss_vel,
                train_loss_coef,
                train_loss_exist,
                train_loss_pix,
            )
        )
        print(
            "Epoch %d  ======> Valid  loss = %.4f, loss_pos = %.4f, loss_dynamic = %.4f, loss_coef = %.4f, loss_exist = %.4f,"
            % (
                epoch + 1,
                val_loss,
                val_loss_position,
                val_loss_vel,
                val_loss_coef,
                val_loss_exist,
            )
        )

        writer.add_scalar("TRAIN/lr", optim.param_groups[0]["lr"], epoch + 1)

        writer.add_scalar("TRAIN/df", df, epoch + 1)
        writer.add_scalar("TRAIN/loss", train_loss, epoch + 1)
        writer.add_scalar("TRAIN/loss_pos", train_loss_position, epoch + 1)
        writer.add_scalar("TRAIN/loss_coef", train_loss_coef, epoch + 1)
        writer.add_scalar("TRAIN/loss_pix", train_loss_pix, epoch + 1)
        writer.add_scalar("Valid/loss", val_loss, epoch + 1)
        writer.add_scalar("Valid/loss_pos", val_loss_position, epoch + 1)
        writer.add_scalar("Valid/loss_coef", val_loss_coef, epoch + 1)

        sched.step(val_loss)

        _save_ckpt("latest_model.pt")

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_epoch = epoch + 1
            _save_ckpt("best_train_model.pt")
            writer.add_scalar("TRAIN/best_loss", best_train_loss, epoch + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
            _save_ckpt("best_val_model.pt")
            writer.add_scalar("Valid/best_loss", best_val_loss, epoch + 1)

        if val_loss_position < best_val_position_loss:
            best_val_position_loss = val_loss_position
            best_val_position_epoch = epoch + 1
            _save_ckpt("best_val_pos_model.pt")
            writer.add_scalar("Valid/best_loss_pos", best_val_position_loss, epoch + 1)

        if (
            val_pixel_dataloader is not None
            and args.val_pixel_every > 0
            and ((epoch + 1) % args.val_pixel_every == 0 or epoch + 1 == args.epoch)
        ):
            pixel_metrics = validate_pixel_protocol(
                Perception_model, dynamics_model, val_pixel_dataloader, args
            )
            for key, value in pixel_metrics.items():
                writer.add_scalar("ValidPixel/" + key, value, epoch + 1)
            metrics_path = os.path.join(log_dir, "val_pixel_metrics.csv")
            pd.DataFrame([{"epoch": epoch + 1, **pixel_metrics}]).to_csv(
                metrics_path,
                mode="a",
                header=not os.path.exists(metrics_path),
                index=False,
            )
            if pixel_metrics["PSNR"] > best_val_psnr:
                best_val_psnr = pixel_metrics["PSNR"]
                best_val_psnr_epoch = epoch + 1
                _save_ckpt("best_val_psnr_model.pt")
            if pixel_metrics["SSIM"] > best_val_ssim:
                best_val_ssim = pixel_metrics["SSIM"]
                best_val_ssim_epoch = epoch + 1
                _save_ckpt("best_val_ssim_model.pt")
            if pixel_metrics["LPIPS_vgg"] < best_val_lpips:
                best_val_lpips = pixel_metrics["LPIPS_vgg"]
                best_val_lpips_epoch = epoch + 1
                _save_ckpt("best_val_lpips_model.pt")

        print(
            "Epoch %d  ======> Best Train loss = %f @ Epoch %d"
            % (epoch + 1, best_train_loss, best_train_epoch)
        )
        print(
            "Epoch %d  ======> Best Val loss = %f @ Epoch %d"
            % (epoch + 1, best_val_loss, best_val_epoch)
        )
        print(
            "Epoch %d  ======> Best Val loss position = %f @ Epoch %d"
            % (epoch + 1, best_val_position_loss, best_val_position_epoch)
        )
        print("log_dir:", log_dir)
    log_result = {
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "best_val_loss_pos": float(best_val_position_loss),
        "initial_lr": float(args.lr),
        "final_lr": float(optim.param_groups[0]["lr"]),
        "best_val_psnr": None if best_val_psnr == -math.inf else best_val_psnr,
        "best_val_psnr_epoch": best_val_psnr_epoch,
        "best_val_ssim": None if best_val_ssim == -math.inf else best_val_ssim,
        "best_val_ssim_epoch": best_val_ssim_epoch,
        "best_val_lpips": None if best_val_lpips == math.inf else best_val_lpips,
        "best_val_lpips_epoch": best_val_lpips_epoch,
    }
    update_json(os.path.join(log_dir, "log.json"), log_result)
    print("log_dir:", log_dir)
    return log_dir


CKPT_NAMES = [
    "best_val_model.pt",
    "best_val_pos_model.pt",
    "best_val_psnr_model.pt",
    "best_val_ssim_model.pt",
    "best_val_lpips_model.pt",
    "best_train_model.pt",
    "latest_model.pt",
]


def eval_one_checkpoint(pt_path, des="Test"):
    _load_dynamics_state(pt_path)
    # Mode-B runs save a fine-tuned decoder next to each dynamics checkpoint;
    # load it so eval matches training. Absent for normal runs -> no-op.
    _dec_path = pt_path.replace(".pt", ".decoder.pt")
    if not args.decoder_checkpoint and os.path.isfile(_dec_path):
        Perception_model.decoder.load_state_dict(
            torch.load(_dec_path, map_location=device)
        )
        print("Loaded fine-tuned decoder:", _dec_path)
    _export = (
        args.export_pred_dir
        if (
            os.path.basename(pt_path)
            == (args.eval_only_checkpoint or "best_val_model.pt")
        )
        else ""
    )
    (
        test_loss,
        test_loss_pos,
        test_loss_vel,
        test_loss_coef,
        test_loss_exist,
        test_loss_ori,
        test_loss_scale,
        test_mse,
        test_psnr,
        test_local_psnr,
        test_ssim,
        test_lpips,
    ) = test(
        Perception_model,
        dynamics_model,
        val_eval_dataloader if args.eval_split == "val" else test_dataloader,
        args=args,
        des="%s %s" % (args.eval_split, des),
        viz=False,
        export_dir=_export,
    )
    pose_err = (test_loss_pos * 2 + test_loss_coef * 4 + test_loss_exist) / 7
    return {
        "ckpt": os.path.basename(pt_path),
        "loss": float(test_loss),
        "loss_pos": float(test_loss_pos),
        "loss_vel": float(test_loss_vel),
        "loss_coef": float(test_loss_coef),
        "loss_exist": float(test_loss_exist),
        "loss_ori": float(test_loss_ori),
        "loss_scale": float(test_loss_scale),
        "pose_error": float(pose_err),
        "MSE": float(test_mse),
        "PSNR": float(test_psnr),
        "L-PSNR": float(test_local_psnr),
        "SSIM": float(test_ssim),
        "LPIPS": float(test_lpips),
        "LPIPS_x100": float(test_lpips) * 100.0,
    }


def eval_all_checkpoints(ckpt_dir):
    """Evaluate every saved .pt in ckpt_dir and write metric_table.csv."""
    rows = []
    seen = set()
    if args.eval_only_checkpoint:
        names = [args.eval_only_checkpoint]
    else:
        names = list(CKPT_NAMES)
        for fn in sorted(os.listdir(ckpt_dir)):
            if fn.endswith(".decoder.pt"):
                continue  # mode-B decoder snapshots, not dynamics checkpoints
            if fn.endswith(".pt") and fn not in names:
                names.append(fn)
    for name in names:
        pt_path = os.path.join(ckpt_dir, name)
        if not os.path.isfile(pt_path) or pt_path in seen:
            continue
        seen.add(pt_path)
        print("======= Evaluate %s =======" % name)
        rows.append(eval_one_checkpoint(pt_path, des="Test %s" % name))
    if not rows:
        print("No .pt found in", ckpt_dir)
        return None
    df = pd.DataFrame(rows)
    csv_path = os.path.join(ckpt_dir, "metric_table.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print("======= metric table =======")
    print(df.to_string(index=False, float_format=lambda v: "%.4f" % v))
    print("saved:", csv_path)
    best_row = next((r for r in rows if r["ckpt"] == "best_val_model.pt"), rows[0])
    update_json(
        os.path.join(ckpt_dir, "log.json"),
        {
            "Perception_model_weight_path": Perception_weight_path,
            "Dynamics_model_weight_path": os.path.join(ckpt_dir, best_row["ckpt"]),
            "test_loss": best_row["loss"],
            "test_loss_pos": best_row["loss_pos"],
            "test_loss_coef": best_row["loss_coef"],
            "test_loss_exist": best_row["loss_exist"],
            "test_loss_ori": best_row["loss_ori"],
            "test_loss_scale": best_row["loss_scale"],
            "test_pixel_mse": best_row["MSE"],
            "test_psnr": best_row["PSNR"],
            "test_local_psnr": best_row["L-PSNR"],
            "test_ssim": best_row["SSIM"],
            "test_lpips": best_row["LPIPS"],
            "test_pose_error": best_row["pose_error"],
            "num_para": para_num,
            "metric_table": csv_path,
        },
    )
    return df


if args.epoch != 0:
    print("=======Train===========")
    _train_cache = _perception_cache_path(_train_split_name)
    _val_cache = _perception_cache_path(_val_split_name)
    _tf, _tp, _trgb = build_or_load_perception_cache(
        train_dataloader.dataset, _train_cache, batch_size=args.batchsize
    )
    _vf, _vp, _vrgb = build_or_load_perception_cache(
        val_dataloader.dataset, _val_cache, batch_size=8
    )
    _tf, _tp, _trgb = _filter_cache_by_count(
        _tf, _tp, _trgb, _train_split_name, args.train_counts)
    _vf, _vp, _vrgb = _filter_cache_by_count(
        _vf, _vp, _vrgb, _val_split_name, args.val_counts)
    # Features already live in RAM; workers only add pickle overhead.
    train_dataloader = DataLoader(
        CachedPerceptionDataset(
            _tf, _tp, window=_need_frames, random_window=args.random_window,
            rgb=_trgb, windows_per_video=args.windows_per_video,
            collision_sample_prob=args.collision_sample_prob,
            window_max_frame=args.train_window_max_frame,
        ),
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        CachedPerceptionDataset(
            _vf, _vp, window=_need_frames, random_window=False, rgb=_vrgb
        ),
        batch_size=8,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    val_pixel_dataloader = None
    if args.val_pixel_every > 0:
        if _vrgb is None or _vp.shape[1] < args.given_frames + 44:
            raise RuntimeError(
                "validation pixel selection requires an RGB cache with at least "
                "given_frames+44 frames"
            )
        val_pixel_dataloader = DataLoader(
            CachedPerceptionDataset(
                _vf,
                _vp,
                window=args.given_frames + 44,
                random_window=False,
                rgb=_vrgb,
            ),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    if args.weight_path != "":
        pt_path = os.path.join(args.weight_path)
        print("Loading pre_training weight file from %s" % pt_path)
        _load_dynamics_state(pt_path)
    log_dir = train(
        Perception_model,
        dynamics_model,
        train_dataloader,
        val_dataloader,
        args,
        val_pixel_dataloader=val_pixel_dataloader,
    )
    if args.skip_test_after_train:
        print("skip_test_after_train: validation-selected checkpoints saved in", log_dir)
    else:
        print("=======Test all checkpoints===========")
        print("parameter = ", para_num)
        eval_all_checkpoints(log_dir)

else:
    print("=======test===========")
    if os.path.isdir(args.weight_path):
        ckpt_dir = args.weight_path
    else:
        ckpt_dir = os.path.dirname(os.path.abspath(args.weight_path))
    print("Evaluate checkpoints in", ckpt_dir)
    eval_all_checkpoints(ckpt_dir)

# para_num = count_parameters(dynamics_model)
# print('parameter = ', para_num)

"""
#%% ?????????????????
# pt_path = os.path.join(args.weight_path)
# print('Loading pre_training weight file from %s' % pt_path)
# dynamics_model.load_state_dict(torch.load(pt_path, map_location=device))
# Perception_model.eval()
# dynamics_model.eval()
# video_dataloader = DataLoader(
#     blocktowerCF_Video(
#         mode='test', resolution=112, sampling_mode='full', env=args.dataset),
#     batch_size=1, num_workers=1, pin_memory=True)

# with torch.no_grad():
#     for i, x in enumerate(tqdm(video_dataloader, desc='Test rec_image')):

#         # 1 T C H W -> T C H W -> T 16 28 28
#         rgb_data = x['rgb_ab'].squeeze().to(device)  # T C H W
#         given_data = rgb_data[0:args.given_frames]

#         # given_state = torch.cat([Perception_model.get_keypoints_only(given_data)[
#         #     0], Perception_model.get_keypoints_only(given_data)[1]], dim=-1).unsqueeze(0)
#         # given_state = creat_state_squence(
#         #     given_state, n=args.num_velocity)
#         # static_feature = Perception_model.get_features_only(
#         #     given_data[0:1])  # ????????????

#         # # ????????
#         # horizon = args.horizon
#         # rec_state_with_speed = Inference_model.prediction(
#         #     given_state, horizon).squeeze()  # [horizon K 11] ?????????

#         # rec_pos = rec_state_with_speed[..., 0:2]
#         # rec_coef = rec_state_with_speed[..., -5:]  # T K 5

#         rgb = given_data
#         T, C, H, W = rgb.shape
#         inital_feature_map = Perception_model.get_features_only(
#             rgb[0:1])  # B 16 28 28
#         state_1, state_2 = Perception_model.get_keypoints_only(
#             rgb.contiguous().view(-1, C, H, W))
#         pos = torch.cat([state_1, state_2], dim=-1)  # B T K S
#         K = pos.shape[-2]
#         pos = pos.view(1, T, K, -1)
#         new_pos = creat_state_squence(pos, n=args.num_velocity)
#         input_state = new_pos[:, 0:args.given_frames-args.num_velocity]

#         predict_list = []
#         for i in range(input_state.shape[1]-1):
#             cur_obj = input_state[:, i]
#             delta_state = dynamics_model(inpt=cur_obj, args=args, env_features=inital_feature_map,
#                                          pos_dim=args.pos_emb, first_timestep=(i == 0))
#             next_state = delta_state + cur_obj
#             predict_list.append(next_state)

#         decoder_input = input_state[:, -1]
#         for t in range(args.horizon):
#             cur_obj = decoder_input
#             delta_state = dynamics_model(inpt=cur_obj, args=args, env_features=inital_feature_map,
#                                          pos_dim=args.pos_emb, first_timestep=(t == 0 and input_state.shape[1] == 1))
#             next_state = delta_state + cur_obj
#             predict_list.append(next_state)
#             decoder_input = next_state

#         predicted_states = torch.stack(predict_list, dim=1).squeeze()  # T S
#         rec_pos = predicted_states[..., 0:2]
#         rec_coef = predicted_states[..., -args.n_coefficients-1:]  # T K 5

#         if x['ex'][0].item() == 3196:
#             Y = 1
#             ## ?????????????????????????
#             # rec_coef[:, 3, -1] = 0.1
#             # rec_coef[:, 0, -1] = 0.1
#             # rec_coef[:, 2, -1] = 0.1
#         rec_rgb = Perception_model.create_img(inital_feature_map.repeat(
#             args.horizon, 1, 1, 1), rec_pos, rec_coef)

#         gt_rbg = rgb_data[args.given_frames:args.given_frames+args.horizon]

#         gt_img = gt_rbg.cpu().detach().numpy().transpose(0, 2, 3, 1)  # T H W C
#         rec_img = rec_rgb.cpu().detach().numpy().transpose(0, 2, 3, 1)  # T H W C

#         if str(x['ex'][0].item()) == '3196':

#             vis_img(gt_img, rec_img, '',
#                     batch_idx='video_'+str(x['ex'][0].item()))
#             # inpt_imgs = given_data.cpu().detach().numpy().transpose(0, 2, 3, 1)
#             # inpt_imgs = (inpt_imgs + 1) / 2  # [-1,1] -> [0,1]
#             # tau = inpt_imgs.shape[0]
#             # fig, ax = plt.subplots(1, tau, figsize=(12, 4))
#             # for k in range(tau):
#             #     ax[k].imshow(inpt_imgs[k])
#             #     ax[k].axis("off")
#             #     ax[k].set_title('t='+str(k+1))
#             # plt.tight_layout()
#             # plt.savefig(os.path.join('', 'video_' +
#             #             str(x['ex'][0].item()) + '_inputs.jpg'))
#             break
"""

# %% ????????????????FLOPs
# with torch.no_grad():
#     for i, x in enumerate(tqdm(test_dataloader)):
#         rgb = x['rgb_ab'].to(device)
#         rgb = rgb[:, 0:args.given_frames + args.horizon]
#         print('input shape:', rgb.shape)
#         break
# Perception_model.eval()

# # 1. ??????????? --- ????????
# B, T, C, H, W = rgb.shape  # 1 26 3 112 112
# inital_feature_map = Perception_model.get_features_only(
#     rgb[:, 0])  # B 16 28 28
# state_1, state_2 = Perception_model.get_keypoints_only(
#     rgb.contiguous().view(-1, C, H, W))
# pos = torch.cat([state_1, state_2], dim=-1)  # B T K S
# K = pos.shape[-2]
# pos = pos.view(B, T, K, -1)
# new_pos = creat_state_squence(pos, n=args.num_velocity)
# input_state = new_pos[:, 0:args.given_frames-args.num_velocity]
# target_state = new_pos[:, args.given_frames-args.num_velocity:]
# cur_obj = input_state[:, 0]  # B T K S -> B K S
# print('input state shape:', cur_obj.shape)


# print("{} | {} | {}".format("Model", "Params(M)", "FLOPs(G)"))
# print("---|---|---")
# total_ops, total_params = profile(
#     Perception_model, (rgb[:, 0], rgb[:, 0]))  # ?????
# print("{} | {:.2f} | {:.2f}".format(args.name, total_params / 1e6, total_ops / 1e9,
#                                     ))

# print("{} | {} | {}".format("Model", "Params(M)", "FLOPs(G)"))
# print("---|---|---")
# total_ops, total_params = profile(
#     dynamics_model, (cur_obj, args, inital_feature_map,
#                      args.pos_emb, True))
# print("{} | {:.2f} | {:.2f}".format(args.name, total_params / 1e6, total_ops / 1e9,
#                                     ))
