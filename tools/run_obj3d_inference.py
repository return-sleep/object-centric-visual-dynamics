#!/usr/bin/env python
"""Generate OBJ3D predictions with fixed model and stochastic-rollout averaging."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--checkpoint-root', default=str(ROOT / 'checkpoints/obj3d'))
    p.add_argument('--output-root', default=str(ROOT / 'runs/obj3d-inference'))
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--limit-videos', type=int, default=0)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    cfg = json.loads((ROOT / 'configs/obj3d_inference.json').read_text())
    ckpt = Path(a.checkpoint_root).resolve()
    out = Path(a.output_root).resolve()
    data = Path(a.data_root).resolve()
    jobs = [(name, 1024, []) for name in cfg['members'] if name != 'stochastic']
    jobs += [('stoch_s%d' % seed, seed, ['--rollout_noise_std', str(cfg['rollout_noise_std'])])
             for seed in cfg['stochastic_seeds']]
    for name, seed, extra in jobs:
        member = 'stochastic' if name.startswith('stoch_s') else name
        weight = ckpt / cfg['checkpoint_members'][member] / 'best_val_psnr_model.pt'
        perception = ckpt / 'perception/best_val_obj3d_4kpts.nn'
        if not a.dry_run:
            for file in (weight, perception, perception.parent / 'log.json'):
                if not file.is_file():
                    p.error('Missing checkpoint/config: %s' % file)
        cmd = [sys.executable, str(ROOT / 'main.py'), '--dataset', 'obj3d',
               '--epoch', '0', '--batchsize', '1', '--given_frames', '6',
               '--horizon', '44', '--n_keypoints', '6', '--n_coefficients', '8',
               '--num_velocity', str(cfg['num_velocity'][member]), '--num_gn', '2',
               '--Inference_model', 'predictor_module_with_appr', '--consider_appr',
               '--perception_auto_config', '--Perception_model_weight_path', str(perception),
               '--weight_path', str(weight), '--eval_only_checkpoint', weight.name,
               '--obj3d_root', str(data), '--export_pred_dir', str(out / name),
               '--val_skip_lpips', '--gpu', str(a.gpu), '--seed', str(seed)]
        if member == 'r9':
            cmd += ['--long_horizon_calibrator', '--calibrator_scope', 'full', '--calibrator_scale', '0.05']
        if member == 'residual':
            for field in ('static_residual_strength', 'static_residual_threshold', 'static_residual_dilation'):
                cmd += ['--' + field, str(cfg[field])]
        if a.limit_videos:
            cmd += ['--max_videos', str(a.limit_videos)]
        cmd += extra
        print(' '.join(cmd), flush=True)
        if not a.dry_run:
            (out / 'logs').mkdir(parents=True, exist_ok=True)
            with (out / 'logs' / (name + '.log')).open('w') as log:
                subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=True)
    if a.dry_run:
        return
    import numpy as np
    dirs = [out / name for name, _, _ in jobs]
    ids = sorted(int(f.stem) for f in dirs[0].glob('*.npy'))
    if not ids or any(sorted(int(f.stem) for f in d.glob('*.npy')) != ids for d in dirs):
        raise RuntimeError('Members have missing or inconsistent prediction IDs')
    expected = sorted(int(d.name) for d in (data / 'test').iterdir() if d.is_dir() and d.name.isdigit())
    if a.limit_videos:
        expected = expected[:a.limit_videos]
    if ids != expected:
        raise RuntimeError('Prediction IDs do not match the requested test split')
    final = out / 'predictions'; final.mkdir(exist_ok=True)
    for ep in ids:
        arrays = []
        for member in cfg['members']:
            if member == 'stochastic':
                samples = [np.load(out / ('stoch_s%d' % s) / ('%d.npy' % ep)) for s in cfg['stochastic_seeds']]
                arr = np.mean(np.stack(samples), axis=0)
            else:
                arr = np.load(out / member / ('%d.npy' % ep))
            arrays.append(arr.astype(np.float32))
        stack = np.stack(arrays)
        if stack.shape[1:] != (44, 128, 128, 3) or not np.isfinite(stack).all():
            raise RuntimeError('Invalid prediction array for episode %d' % ep)
        blend = np.empty_like(stack[0])
        for (start, end), weights in zip(cfg['buckets'], cfg['weights']):
            blend[start:end] = np.einsum('m,mthwc->thwc', np.asarray(weights, dtype=np.float64), stack[:, start:end])
        np.save(final / ('%d.npy' % ep), np.clip(blend, 0, 1).astype(np.float32))
    (out / 'inference_config.json').write_text(json.dumps(cfg, indent=2))
    print('Generated %d videos: %s' % (len(ids), final))


if __name__ == '__main__':
    main()
