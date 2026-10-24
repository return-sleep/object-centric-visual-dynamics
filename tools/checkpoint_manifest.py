#!/usr/bin/env python
"""Generate or verify SHA256 manifests for model checkpoint files."""
import argparse
import hashlib
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='checkpoints')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.verify:
        rows = args.verify.read_text().splitlines()
        if not rows:
            parser.error('Manifest is empty')
        seen = set()
        for row in rows:
            fields = row.split('  ', 1)
            if len(fields) != 2:
                parser.error('Invalid manifest row')
            expected, name = fields
            path = (root / name).resolve()
            if root not in path.parents or name in seen:
                parser.error('Unsafe or duplicate manifest path: ' + name)
            seen.add(name)
            if not path.is_file() or digest(path) != expected:
                parser.error('Missing file or hash mismatch: ' + name)
        print('Verified %d checkpoints' % len(seen))
    else:
        paths = sorted(p for p in root.rglob('*')
                       if p.is_file() and p.suffix in ('.pt', '.pth', '.nn'))
        if not paths:
            parser.error('No checkpoint files found')
        for path in paths:
            print('%s  %s' % (digest(path), path.relative_to(root).as_posix()))


if __name__ == '__main__':
    main()
