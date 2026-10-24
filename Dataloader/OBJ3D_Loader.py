"""
OBJ3D (G-SWM) PNG-sequence loader for stage-1 derendering training.

Data layout:
    {root}/{train,val,test}/{ep_id}/test_1.png ... test_100.png
    - ep_id are integer folder names
    - frames are 1-indexed, native 128x128 RGBA

Interface matches what train_derendering.py expects:
    {'ex': ep_id, 'rgb_ab': FloatTensor [2, C, H, W] in [-1, 1]}
sampling_mode='full' returns the whole episode [T, C, H, W] instead
(reserved for the stage-2 adaptation in main.py).
"""

import glob
import os
import os.path as osp
from random import randint, random

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _frame_number(path):
    return int(osp.splitext(osp.basename(path))[0].partition('_')[-1])


class OBJ3D_Video(Dataset):
    def __init__(self, mode='train', resolution=64, sampling_mode='rand',
                 root='../Datasets/OBJ3D', frame_gap=0, max_frames=None,
                 temporal_neighbor=False, early_boost=0.0, early_max=15,
                 shift_aug=0, gap_range=None):
        """
        :param mode: 'train', 'val' or 'test' split
        :param resolution: output image size (square), native is 128
        :param sampling_mode: 'rand' random source frame, 'fix' source at frame 0,
            'full' load the entire episode
        :param root: OBJ3D root containing train/val/test subfolders
        :param frame_gap: source-target distance in frames; 0 = int(0.2 * T),
            matching the CoPhy convention (0.2 * video_length)
        :param max_frames: in 'full' mode, keep only the first max_frames frames
            (stage-2 needs given_frames + horizon, not all 100)
        :param temporal_neighbor: also return 'rgb_adj', the frame right after
            the target (stage-1 temporal losses T1/T2)
        :param early_boost: with this probability, sample the source frame from
            [0, early_max] instead of the whole episode (oversample early frames
            where balls are entering / half-visible at borders)
        :param early_max: upper frame index for the early-frame window
        :param shift_aug: max pixel shift for border-clip augmentation. The whole
            frame tuple is padded (replicate) by this amount and randomly
            translated, so balls get artificially clipped at borders and their
            centers can fall outside the frame. Teaches the encoder/decoder to
            represent half-visible objects. Combine with kpt_margin >=
            2*shift_aug/resolution so keypoints can sit at out-of-frame centers.
        :param gap_range: (lo, hi) to sample the source-target distance per item
            instead of using the fixed frame_gap. With the fixed gap of 20 an
            early source frame always pairs with a target >= 20, so early-frame
            states never receive a reconstruction gradient; a random gap also
            matches the gap 1..44 range seen during stage-2 evaluation.
        """
        super(OBJ3D_Video, self).__init__()
        assert sampling_mode in ['rand', 'fix', 'full']
        assert mode in ['train', 'val', 'test']
        self.root = osp.join(root, mode)
        assert osp.isdir(self.root), 'OBJ3D split not found: %s' % self.root
        self.mode = mode
        self.resolution = resolution
        self.sampling_mode = sampling_mode
        self.frame_gap = frame_gap
        self.max_frames = max_frames
        self.temporal_neighbor = temporal_neighbor
        self.early_boost = early_boost
        self.early_max = early_max
        self.shift_aug = shift_aug
        self.gap_range = gap_range

        folders = []
        for name in os.listdir(self.root):
            try:
                folders.append(int(name))
            except ValueError:
                continue
        folders.sort()

        self.episodes = []
        for f in folders:
            paths = glob.glob(osp.join(self.root, str(f), 'test_*.png'))
            paths.sort(key=_frame_number)
            if len(paths) >= 2:
                self.episodes.append((f, paths))
        self.video_length = max(len(paths) for _, paths in self.episodes)

    def __len__(self):
        return len(self.episodes)

    def _read_frame(self, path):
        img = Image.open(path).convert('RGB')
        if img.size != (self.resolution, self.resolution):
            img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        return transforms.ToTensor()(img) * 2 - 1  # [-1, 1]

    def __getitem__(self, item):
        ep_id, paths = self.episodes[item]
        T = len(paths)
        out = {'ex': ep_id}

        if self.sampling_mode == 'full':
            if self.max_frames:
                paths = paths[:self.max_frames]
            rgb = torch.stack([self._read_frame(p) for p in paths], dim=0)
            out['rgb_ab'] = rgb
            return out

        if self.gap_range is not None and self.sampling_mode == 'rand':
            lo, hi_g = self.gap_range
            gap = randint(int(lo), int(min(hi_g, T - 2)))
        else:
            gap = self.frame_gap if self.frame_gap > 0 else int(0.2 * T)
        gap = min(gap, T - 1)
        if self.sampling_mode == 'rand':
            hi = T - 1 - gap
            if self.early_boost > 0 and random() < self.early_boost:
                t = randint(0, min(self.early_max, hi))
            else:
                t = randint(0, hi)
        else:
            t = 0
        rgb = torch.stack([self._read_frame(paths[t]),
                           self._read_frame(paths[t + gap])], dim=0)
        out['rgb_ab'] = rgb  # [2, C, H, W]
        if self.temporal_neighbor:
            out['rgb_adj'] = self._read_frame(paths[min(t + gap + 1, T - 1)])
        if self.shift_aug > 0:
            import torch.nn.functional as F
            S = self.shift_aug
            dx, dy = randint(-S, S), randint(-S, S)
            keys = ['rgb_ab'] + (['rgb_adj'] if 'rgb_adj' in out else [])
            for k in keys:
                v = F.pad(out[k], (S, S, S, S), mode='replicate')
                H, W = v.shape[-2:]
                out[k] = v[..., S + dy:S + dy + H - 2 * S,
                           S + dx:S + dx + W - 2 * S]
        return out
