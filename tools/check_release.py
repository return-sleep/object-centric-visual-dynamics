#!/usr/bin/env python
"""Fail if the public tree contains private or oversized artifacts."""

from __future__ import print_function

import os
import re
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}
FORBIDDEN_PARTS = {"NeuroComputing投稿", ".cursor", "new_output", "output"}
FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".nn", ".npy", ".npz", ".hkl", ".mp4", ".zip", ".tgz"
}
SENSITIVE = [
    re.compile(r"/remote-home/"),
    re.compile(r"dataset_rc_n4_p1"),
    re.compile(r"xiaopeng\.com", re.IGNORECASE),
    re.compile(r"activate_gsgn"),
    re.compile(r"Editorial\s+Manager", re.IGNORECASE),
    re.compile(r"password\s*[:=]", re.IGNORECASE),
]
TRAFFIC_CODE = re.compile(r"traffic", re.IGNORECASE)
MAX_BYTES = 20 * 1024 * 1024


def iter_files():
    for directory, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            yield os.path.join(directory, filename)


def main():
    failures = []
    for path in iter_files():
        relative = os.path.relpath(path, ROOT)
        parts = set(relative.split(os.sep))
        if parts & FORBIDDEN_PARTS:
            failures.append("%s: forbidden path" % relative)
        if any(relative.lower().endswith(suffix)
               for suffix in FORBIDDEN_SUFFIXES):
            failures.append("%s: binary artifact" % relative)
        if os.path.getsize(path) > MAX_BYTES:
            failures.append("%s: larger than 20 MiB" % relative)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except (UnicodeDecodeError, OSError):
            continue
        if relative == os.path.join("tools", "check_release.py"):
            continue
        for pattern in SENSITIVE:
            if pattern.search(text):
                failures.append(
                    "%s: sensitive pattern %s" % (relative, pattern.pattern))
        if relative.endswith(".py") and TRAFFIC_CODE.search(text):
            failures.append("%s: excluded Traffic code" % relative)
    if failures:
        print("\n".join(sorted(set(failures))))
        return 1
    print("release check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
