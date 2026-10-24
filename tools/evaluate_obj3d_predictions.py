#!/usr/bin/env python
"""Evaluate OBJ3D prediction arrays using frame-wise PIQA metrics."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch

def load_obj3d_gt(obj3d_root, ep_id, given_frames, horizon, split="test"):
    """GT predicted frames of one OBJ3D test episode: [horizon, 3, H, W] in [0, 1]."""
    from PIL import Image

    frames = []
    for i in range(given_frames + 1, given_frames + horizon + 1):  # 1-indexed pngs
        path = os.path.join(obj3d_root, split, str(ep_id), "test_%d.png" % i)
        img = Image.open(path).convert("RGB")
        frames.append(torch.from_numpy(np.array(img)).float() / 255.0)
    return torch.stack(frames, 0).permute(0, 3, 1, 2).contiguous()

def eval_obj3d(args, device):
    """Exact DDLP caliber: piqa PSNR/SSIM/LPIPS-vgg on [0,1], mean +/- std."""
    from piqa import PSNR, SSIM, LPIPS

    psnr_fn = PSNR(reduction="none").to(device)
    ssim_fn = SSIM(reduction="none").to(device)
    lpips_fn = None
    if not args.skip_lpips:
        lpips_fn = LPIPS(network="vgg", reduction="none").to(device)

    test_dir = os.path.join(args.obj3d_root, args.split)
    pred_files = sorted(
        (f for f in os.listdir(args.pred_dir) if f.endswith(".npy")),
        key=lambda f: int(os.path.splitext(f)[0]),
    )
    assert len(pred_files) > 0, "no npy predictions in %s" % args.pred_dir

    psnr_all, ssim_all, lpips_all = [], [], []
    for f in pred_files:
        ep_id = int(os.path.splitext(f)[0])
        assert os.path.isdir(os.path.join(test_dir, str(ep_id))), \
            "no GT episode for pred %s" % f
        gt = load_obj3d_gt(
            args.obj3d_root, ep_id, args.given_frames, args.horizon, args.split
        )
        pred = torch.from_numpy(np.load(os.path.join(args.pred_dir, f))).float()
        pred = pred.permute(0, 3, 1, 2).clamp(0, 1).contiguous()  # T C H W in [0, 1]
        assert pred.shape == gt.shape, \
            "ep %d: pred %s vs gt %s" % (ep_id, pred.shape, gt.shape)

        gt, pred = gt.to(device), pred.to(device)
        with torch.no_grad():
            psnr_all.append(psnr_fn(pred, gt).cpu())
            ssim_all.append(ssim_fn(pred, gt).cpu())
            if lpips_fn is not None:
                lpips_all.append(lpips_fn(pred, gt).cpu())

    psnr_all = torch.cat(psnr_all)
    ssim_all = torch.cat(ssim_all)
    row = {
        "method": args.method,
        "dataset": args.dataset,
        "n_videos": len(pred_files),
        "PSNR": float(psnr_all.mean()),
        "PSNR_std": float(psnr_all.std()),
        "SSIM": float(ssim_all.mean()),
        "SSIM_std": float(ssim_all.std()),
    }
    if args.horizon >= 44:
        per_video_psnr = psnr_all.reshape(len(pred_files), args.horizon)
        per_video_ssim = ssim_all.reshape(len(pred_files), args.horizon)
        row["PSNR_1_14"] = float(per_video_psnr[:, :14].mean())
        row["PSNR_15_29"] = float(per_video_psnr[:, 14:29].mean())
        row["PSNR_30_44"] = float(per_video_psnr[:, 29:44].mean())
        row["SSIM_1_14"] = float(per_video_ssim[:, :14].mean())
        row["SSIM_15_29"] = float(per_video_ssim[:, 14:29].mean())
        row["SSIM_30_44"] = float(per_video_ssim[:, 29:44].mean())
    if lpips_all:
        lpips_all = torch.cat(lpips_all)
        row["LPIPS_vgg"] = float(lpips_all.mean())
        row["LPIPS_vgg_std"] = float(lpips_all.std())
        if args.horizon >= 44:
            per_video_lpips = lpips_all.reshape(len(pred_files), args.horizon)
            row["LPIPS_1_14"] = float(per_video_lpips[:, :14].mean())
            row["LPIPS_15_29"] = float(per_video_lpips[:, 14:29].mean())
            row["LPIPS_30_44"] = float(per_video_lpips[:, 29:44].mean())
    else:
        row["LPIPS_vgg"] = float("nan")
        row["LPIPS_vgg_std"] = float("nan")
    return row

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pred-dir', required=True)
    p.add_argument('--data-root', dest='obj3d_root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit-videos', type=int, default=0)
    a = p.parse_args()
    a.dataset = 'obj3d'; a.method = 'object-centric-visual-dynamics'
    a.given_frames = 6; a.horizon = 44; a.split = 'test'; a.skip_lpips = False
    expected = sorted(int(d.name) for d in (Path(a.obj3d_root) / 'test').iterdir()
                      if d.is_dir() and d.name.isdigit())
    if a.limit_videos:
        expected = expected[:a.limit_videos]
    actual = sorted(int(f.stem) for f in Path(a.pred_dir).glob('*.npy'))
    if not expected or actual != expected:
        p.error('Predictions must cover exactly the requested test episodes')
    result = eval_obj3d(a, torch.device(a.device))
    result['episode_ids'] = actual
    result['protocol'] = '6 conditioning frames, 44 predictions; mean over frames and videos; PIQA PSNR/SSIM/LPIPS-VGG, RGB [0,1]'
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
