#!/usr/bin/env python
"""Validate user-provided datasets and create local path metadata.

This tool never downloads or copies raw datasets.
"""

from __future__ import print_function

import argparse
import json
import os


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPPORTED = ("ballsCF", "blocktowerCF", "simb", "obj3d")


def first_directory(candidates):
    for path in candidates:
        if os.path.isdir(path):
            return os.path.abspath(path)
    return ""


def replace_link(destination, source):
    if os.path.islink(destination):
        if os.path.realpath(destination) == os.path.realpath(source):
            return
        os.unlink(destination)
    elif os.path.exists(destination):
        raise SystemExit(
            "Refusing to replace existing path: %s" % destination)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    os.symlink(source, destination)


def prepare_cophy(dataset, source, check_only):
    scenario = first_directory([
        os.path.join(source, dataset, "4"),
        os.path.join(source, dataset),
        source,
    ])
    if not scenario:
        raise SystemExit("CoPhy scenario directory not found under %s" % source)
    if not check_only:
        path = os.path.join(REPO_ROOT, "Datasets", "dataset_dir.json")
        payload = {}
        if os.path.isfile(path):
            with open(path, "r") as handle:
                payload = json.load(handle)
        payload.setdefault("CoDy", {})[dataset] = scenario
        payload.setdefault("Cophy", {})[dataset] = scenario
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return scenario


def prepare_linked(dataset, source, required, check_only):
    missing = [name for name in required
               if not os.path.exists(os.path.join(source, name))]
    if missing:
        raise SystemExit(
            "%s is missing required entries: %s"
            % (source, ", ".join(missing)))
    destination = os.path.join(REPO_ROOT, "data", dataset)
    if not check_only:
        replace_link(destination, source)
        if dataset == "simb":
            replace_link(os.path.join(REPO_ROOT, "Datasets", "simb"), source)
    return destination


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=SUPPORTED)
    parser.add_argument("--source", required=True,
                        help="path to the already downloaded dataset")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source = os.path.abspath(os.path.expanduser(args.source))
    if not os.path.isdir(source):
        raise SystemExit("Dataset directory not found: %s" % source)
    if args.dataset in ("ballsCF", "blocktowerCF"):
        resolved = prepare_cophy(
            args.dataset, source, args.check_only)
    elif args.dataset == "obj3d":
        resolved = prepare_linked(
            args.dataset, source, ("train", "val", "test"), args.check_only)
    else:
        resolved = prepare_linked(
            args.dataset, source, ("train.hkl", "test.hkl"),
            args.check_only)
    print("%s dataset is ready: %s" % (args.dataset, resolved))


if __name__ == "__main__":
    main()
