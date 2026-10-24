#!/usr/bin/env python
"""Unified train/evaluate entry point for the public release."""

from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPPORTED = ("ballsCF", "blocktowerCF", "simb", "obj3d")


def load_config(dataset):
    path = os.path.join(REPO_ROOT, "configs", dataset + ".json")
    with open(path, "r") as handle:
        return json.load(handle)


def replace_flag(command, flag, value):
    command = list(command)
    if flag in command:
        index = command.index(flag)
        command[index + 1] = str(value)
    else:
        command.extend([flag, str(value)])
    return command


def remove_flag(command, flag):
    command = list(command)
    while flag in command:
        index = command.index(flag)
        del command[index]
        if index < len(command) and not str(command[index]).startswith("--"):
            del command[index]
    return command


def checkpoint_paths(config, root):
    dataset = config["dataset"]
    perception = os.path.join(
        root, dataset, "perception", config["perception"]["checkpoint"])
    dynamics_dir = os.path.join(root, dataset, "dynamics")
    dynamics = os.path.join(dynamics_dir, config["dynamics"]["checkpoint"])
    return perception, dynamics, dynamics_dir


def stage_command(config, stage, paths, options, evaluate=False):
    section = config[stage]
    command = [sys.executable, os.path.join(REPO_ROOT, section["script"])]
    command.extend(section["args"])
    command = replace_flag(command, "--gpu", options.gpu)
    command = replace_flag(command, "--seed", options.seed)

    data_root = os.path.abspath(options.data_root)
    if config["data_kind"] == "obj3d":
        command = replace_flag(command, "--obj3d_root", data_root)
    elif config["data_kind"] == "simb" and stage == "dynamics":
        command = replace_flag(command, "--data_root", data_root)

    perception_path, dynamics_path, dynamics_dir = paths
    if stage == "perception":
        command = replace_flag(
            command, "--output_dir", os.path.dirname(perception_path))
    else:
        command = replace_flag(
            command, "--Perception_model_weight_path", perception_path)
        command = replace_flag(command, "--output_dir", dynamics_dir)
        if options.max_videos and section["script"] == "main.py":
            command = replace_flag(command, "--max_videos", options.max_videos)

    if options.epochs is not None:
        command = replace_flag(command, "--epoch", options.epochs)

    if evaluate:
        command = replace_flag(command, "--epoch", 0)
        command = replace_flag(command, "--weight_path", dynamics_path)
        if config["dataset"] == "obj3d":
            command = replace_flag(command, "--horizon", 44)
            command = remove_flag(command, "--random_window")
            command = remove_flag(command, "--windows_per_video")
        if section["script"] == "main.py":
            command = replace_flag(
                command, "--eval_only_checkpoint",
                os.path.basename(dynamics_path))
    return command


def run_command(command, dry_run):
    print("$ " + " ".join(command))
    if not dry_run:
        subprocess.check_call(command, cwd=REPO_ROOT)


def require_file(path, label):
    if not os.path.isfile(path):
        raise SystemExit(
            "%s not found: %s\nSee checkpoints/README.md or run --mode train."
            % (label, path))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or evaluate one supported NeuroCom dataset.")
    parser.add_argument("--dataset", required=True, choices=SUPPORTED)
    parser.add_argument(
        "--mode", required=True, choices=("train", "eval", "reproduce"))
    parser.add_argument(
        "--stage", default="all", choices=("perception", "dynamics", "all"),
        help="limit training to one stage")
    parser.add_argument(
        "--data-root", default="", help="downloaded dataset root")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="override epochs for smoke tests")
    parser.add_argument(
        "--max-videos", type=int, default=0,
        help="stage-2 subset size for smoke tests")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print commands without importing torch or reading data")
    return parser.parse_args()


def main():
    options = parse_args()
    config = load_config(options.dataset)
    if not options.data_root:
        options.data_root = os.path.join(REPO_ROOT, "data", options.dataset)

    output_root = os.path.abspath(options.output_root)
    checkpoint_root = os.path.abspath(options.checkpoint_root)
    training_paths = checkpoint_paths(config, output_root)
    release_paths = checkpoint_paths(config, checkpoint_root)

    if options.mode in ("train", "reproduce"):
        if options.stage in ("perception", "all"):
            run_command(
                stage_command(
                    config, "perception", training_paths, options),
                options.dry_run)
        elif not options.dry_run:
            require_file(training_paths[0], "trained perception checkpoint")

        if options.stage in ("dynamics", "all"):
            if not options.dry_run:
                require_file(training_paths[0], "trained perception checkpoint")
            run_command(
                stage_command(config, "dynamics", training_paths, options),
                options.dry_run)

    if options.mode == "eval":
        if not options.dry_run:
            require_file(release_paths[0], "perception checkpoint")
            require_file(release_paths[1], "dynamics checkpoint")
        run_command(
            stage_command(
                config, "dynamics", release_paths, options, evaluate=True),
            options.dry_run)
    elif options.mode == "reproduce" and options.stage != "perception":
        if not options.dry_run:
            require_file(training_paths[1], "trained dynamics checkpoint")
        run_command(
            stage_command(
                config, "dynamics", training_paths, options, evaluate=True),
            options.dry_run)


if __name__ == "__main__":
    main()
