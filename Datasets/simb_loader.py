import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image
import cv2
import torch
import hickle
import pickle
import numpy as np
from glob import glob
import cv2
import torch
import random
import numpy as np
from glob import glob
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from yacs.config import CfgNode


def get_config(dataset):
    if dataset != 'simb':
        raise ValueError("This loader supports only the 'simb' dataset.")
    _C = CfgNode()
    _C.INPUT = CfgNode()
    _C.INPUT.PRELOAD_TO_MEMORY = True

    _C.INPUT.IMAGE_MEAN = [123.49497621, 71.4224038, 57.6804548]
    _C.INPUT.IMAGE_STD = [45.16602904, 23.43329777, 15.98162961]
    _C.INPUT.IMAGE_CHANNEL = 3
    _C.RPIN = CfgNode()
    _C.RPIN.ARCH = ''
    _C.RPIN.BACKBONE = ''
    _C.RPIN.HORIZONTAL_FLIP = True
    _C.RPIN.VERTICAL_FLIP = True
    # prediction setting
    _C.RPIN.INPUT_SIZE = 4
    _C.RPIN.PRED_SIZE_TRAIN = 20
    _C.RPIN.PRED_SIZE_TEST = 40
    # input for mixed dataset
    _C.RPIN.IMAGE_EXT = '.jpg'
    _C.RPIN.INPUT_HEIGHT = 64
    _C.RPIN.INPUT_WIDTH = 64
    # training setting
    _C.RPIN.MAX_NUM_OBJS = 3
    _C.RPIN.POSITION_LOSS_WEIGHT = 1
    _C.RPIN.MASK_LOSS_WEIGHT = 0.0
    _C.RPIN.MASK_SIZE = 14
    _C.RPIN.VAE = False
    return _C


def xyxy2xywh(boxes):
    """
    输入:[xmin,ymin,xmax,ymax]
    返回中心坐标以及h,w
    """
    assert boxes.ndim == 2
    w = boxes[:, 2] - boxes[:, 0] + 1.0
    h = boxes[:, 3] - boxes[:, 1] + 1.0
    xc = boxes[:, 0] + 0.5 * (w - 1.0)
    yc = boxes[:, 1] + 0.5 * (h - 1.0)
    return np.vstack([xc, yc, w, h]).transpose()


class Phys(Dataset):
    def __init__(self, data_root, split, C, image_ext='.jpg'):
        self.data_root = data_root
        self.split = split
        self.image_ext = image_ext
        self.C = C
        # 1. define property of input and rollout parameters
        # print(C)
        self.input_size = C.RPIN.INPUT_SIZE  # number of input images
        self.pred_size = eval(
            f'C.RPIN.PRED_SIZE_{"TRAIN" if split == "train" else "TEST"}')
        self.seq_size = self.input_size + self.pred_size
        # 2. define model configs
        self.input_height, self.input_width = C.RPIN.INPUT_HEIGHT, C.RPIN.INPUT_WIDTH
        self.video_list, self.anno_list = None, None
        self.video_info = None

    def __len__(self):
        return self.video_info.shape[0]

    def __getitem__(self, idx):
        # 第几个视频的第几帧 作为第一帧，总共是seq_len-total_len+1
        vid_idx, img_idx = self.video_info[idx, 0], self.video_info[idx, 1]
        # 第几个视频的名称和对应的标签名
        video_name, anno_name = self.video_list[vid_idx], self.anno_list[vid_idx]
        if self.C.RPIN.VAE:
            data, data_t = self._parse_image(video_name, vid_idx, img_idx)
        else:
            data, video_idx = self._parse_image(video_name, vid_idx, img_idx)
            data_t = data.copy()

        boxes, gt_masks = self._parse_label(anno_name, vid_idx, img_idx)

        # image flip augmentation
        if random.random() > 0.5 and self.split == 'train' and self.C.RPIN.HORIZONTAL_FLIP:
            boxes[..., [0, 2]] = self.input_width - boxes[..., [2, 0]]
            data = np.ascontiguousarray(data[..., ::-1])
            gt_masks = np.ascontiguousarray(gt_masks[..., ::-1])

        if random.random() > 0.5 and self.split == 'train' and self.C.RPIN.VERTICAL_FLIP:
            boxes[..., [1, 3]] = self.input_height - boxes[..., [3, 1]]
            # 函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
            data = np.ascontiguousarray(data[..., ::-1, :])
            gt_masks = np.ascontiguousarray(gt_masks[..., ::-1])

        # when the number of objects is fewer than the max number of objects
        num_objs = boxes.shape[1]
        g_idx = []
        for i in range(self.C.RPIN.MAX_NUM_OBJS):
            for j in range(self.C.RPIN.MAX_NUM_OBJS):
                if j == i:
                    continue
                g_idx.append([i, j, (i < num_objs) * (j < num_objs)])
        g_idx = np.array(g_idx)

        valid = np.ones(self.C.RPIN.MAX_NUM_OBJS)
        valid[num_objs:] = 0
        boxes = np.concatenate([boxes] + [boxes[:, :1]
                               for _ in range(self.C.RPIN.MAX_NUM_OBJS - num_objs)], axis=1)
        gt_masks = np.concatenate([gt_masks] + [gt_masks[:, :1]
                                  for _ in range(self.C.RPIN.MAX_NUM_OBJS - num_objs)], axis=1)

        # rois
        rois = boxes[:self.input_size].copy()
        # gt boxes
        # gt_boxes = boxes[self.input_size:].copy()
        gt_boxes = boxes.copy()
        gt_boxes = xyxy2xywh(gt_boxes.reshape(-1, 4)
                             ).reshape((-1, self.C.RPIN.MAX_NUM_OBJS, 4))
        gt_boxes[..., 0::2] /= self.input_width
        gt_boxes[..., 1::2] /= self.input_height
        gt_boxes = gt_boxes.reshape(self.seq_size, -1, 4)  # H K 4

        # a fake variable used to make interface consistent
        labels = torch.zeros(1)
        data = torch.from_numpy(data.astype(np.float32))
        data_t = torch.from_numpy(data_t.astype(np.float32))  # data_t和data是相同的
        rois = torch.from_numpy(rois.astype(np.float32))  # 没有归一化的bbox
        gt_boxes = torch.from_numpy(gt_boxes.astype(np.float32))
        gt_masks = torch.from_numpy(gt_masks.astype(np.float32))
        valid = torch.from_numpy(valid.astype(np.float32))  # 有效的标注?
        out = {'rgb_ab': data, 'ex': video_idx, 'rois': rois,
               'gt_boxes': gt_boxes, 'gt_masks': gt_masks}
        # return data, data_t, rois, gt_boxes, gt_masks, valid, g_idx, labels
        return out

    def _parse_image(self, video_name, vid_idx, img_idx):
        raise NotImplementedError

    def _parse_label(self, anno_name, vid_idx, img_idx):
        raise NotImplementedError


class SimB(Phys):
    def __init__(self, data_root, split, C, image_ext='.jpg'):
        super().__init__(data_root, split, C, image_ext)

        # Video paths, for example '<data_root>/train/000/'.
        self.video_list = sorted(glob(f'{self.data_root}/{self.split}/*/'))
        # Matching annotations, for example '<data_root>/train/000.pkl'.
        self.anno_list = [v[:-1] + '.pkl' for v in self.video_list]

        if C.INPUT.PRELOAD_TO_MEMORY:
            print('loading data from hickle file...')
            data = hickle.load(f'{self.data_root}/{self.split}.hkl')
            self.total_img = np.transpose(data['X'], (0, 1, 4, 2, 3))
            self.total_box = np.zeros((data['y'].shape[:3] + (5,)))
            for anno_idx, anno_name in enumerate(self.anno_list):
                #print(f'loading progress: {anno_idx}/{len(self.anno_list)}')
                with open(anno_name, 'rb') as f:
                    boxes = pickle.load(f)
                self.total_box[anno_idx] = boxes

        self.video_info = np.zeros((0, 2), dtype=np.int32)
        for idx, video_name in enumerate(self.video_list if not C.INPUT.PRELOAD_TO_MEMORY else self.total_box):
            #print(f'loading progress: {idx}/{len(self.video_list)}')
            if C.INPUT.PRELOAD_TO_MEMORY:
                num_sw = self.total_box[idx].shape[0] - self.seq_size + 1
            else:
                # 图片总数
                num_im = len(glob(f'{video_name}/*{image_ext}'))
                # 步长为1的滑动窗口! 震惊
                num_sw = num_im - self.seq_size + 1  # number of sliding windows

            if num_sw <= 0:
                continue
            video_info_t = np.zeros((num_sw, 2), dtype=np.int32)
            # 第几个视频
            video_info_t[:, 0] = idx  # video index
            # 该视频的第几个子序列
            video_info_t[:, 1] = np.arange(num_sw)  # sliding window index
            self.video_info = np.vstack((self.video_info, video_info_t))

    def _parse_image(self, video_name, vid_idx, img_idx):
        if self.C.INPUT.PRELOAD_TO_MEMORY:
            # data = self.total_img[vid_idx,
            #                       img_idx:img_idx + self.input_size].copy()
            data = self.total_img[vid_idx,
                                  img_idx:img_idx + self.seq_size].copy()
            data = data * 2 - 1
            video_idx = str(vid_idx) + '_'+str(img_idx)
        else:
            image_list = sorted(glob(f'{video_name}/*{self.image_ext}'))
            # image_list = image_list[img_idx:img_idx + self.input_size]
            image_list = image_list[img_idx:img_idx + self.seq_size]
            # data = np.zeros(
            #     (self.input_size, 3, self.input_height, self.input_width))
            if self.image_ext == '.jpg':
                data = np.array([
                    cv2.imread(image_name) for image_name in image_list
                ], dtype=float).transpose((0, 3, 1, 2))  # T C H W
                data = data[:, ::-1]
                data = data / 255.0 * 2 - 1
                video_idx = image_list[0].split('.')[0].split(
                    '/')[-2]+'_'+image_list[0].split('.')[0].split('/')[-1]
            else:
                for idx, image_name in enumerate(image_list):
                    with open(image_name, 'rb') as f:
                        data[idx, ...] = (pickle.load(
                            f) / 255).transpose(2, 0, 1)

            # for c in range(3):
            #     data[:, c] -= C.INPUT.IMAGE_MEAN[c]
            #     data[:, c] /= C.INPUT.IMAGE_STD[c]

        return data, video_idx

    def _parse_label(self, anno_name, vid_idx, img_idx):
        if self.C.INPUT.PRELOAD_TO_MEMORY:
            boxes = self.total_box[vid_idx,
                                   img_idx:img_idx + self.seq_size, :, 1:].copy()
        else:
            with open(anno_name, 'rb') as f:
                boxes = pickle.load(f)[img_idx:img_idx + self.seq_size, :, 1:]
        gt_masks = np.zeros(
            (self.pred_size, boxes.shape[1], self.C.RPIN.MASK_SIZE, self.C.RPIN.MASK_SIZE))
        return boxes, gt_masks


# dataset = 'simb'
# split = 'train'
# batchsize = 1
# C = get_config(dataset)
# C.RPIN.HORIZONTAL_FLIP = False
# C.RPIN.VERTICAL_FLIP = False
# train_loader = DataLoader(SimB('Datasets/'+dataset, split='train', C=C, image_ext='.jpg'),
#                           shuffle=True, batch_size=batchsize, num_workers=4)
# idx = 0
# for i, data in enumerate(train_loader):
#     # rgb = data[0]
#     rgb = data['rgb_ab'][-1]  # B T C H W
#     ex = data['ex'][-1]
#     bbox = data['gt_boxes'][-1]  # T K 4

#     video_idx = ex.split('_')[0]
#     video_idx = '%05d' % int(video_idx)
#     img_idx = ex.split('_')[1]
#     img_idx = '%05d' % int(img_idx)
#     img_path = os.path.join('Datasets/simb', split,
#                             str(video_idx), str(img_idx)+'_debug.png')
#     # img=cv2.imread(img_path)
#     xc = bbox[0, :, 0] * 64
#     yc = bbox[0, :, 1] * 64
#     w = bbox[0, :, 2]
#     h = bbox[0, :, 3]

#     source = rgb[0].detach().cpu().numpy().transpose((1, 2, 0))
#     color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#     c_bank = ['red','green', 'blue',    'yellow']
#     source = (source + 1) / 2
#     fig = plt.figure(figsize=(7, 7))
#     ax1 = fig.add_subplot(211)
#     ax2 = fig.add_subplot(212)
#     ax1.imshow(source.squeeze())
#     ax2.imshow(source.squeeze())
#     for k in range(w.shape[0]):
#         ax2.scatter(xc[k], yc[k], c=c_bank[k], s=10)
#     plt.tight_layout()
#     plt.savefig('video_'+str(idx)+'_keypointNet_result'+'.png')

#     # xc =  bbox[0,:,0]
#     # yc =  bbox[0,:,1]
#     # w = bbox[0,:,2]
#     # h = bbox[0,:,3]
#     # for k in range(w.shape[0]):
#     #     a = (int((xc[k]-w[k]/2)*64), int((yc[k]-h[k]/2)*64))
#     #     b = (int((xc[k]+w[k]/2)*64), int((yc[k]+h[k]/2)*64))
#     #     print(a,b)
#     #     cv2.rectangle(img, a, b, color[k], 1)
#     # cv2.imwrite(str(idx)+'.jpg',img)
#     print(rgb.shape)
#     print(bbox.shape)
#     print(ex)
#     print(img_path)
#     idx += 1
#     if idx == 10:
#         break

# dataset = 'simb'
# C = get_config(dataset)

# simb_dataset = SimB('Datasets/'+dataset, split='train', image_ext='.jpg')
# train_dataloader = DataLoader(simb_dataset, shuffle=True, batch_size=1)
# for i, data in enumerate(train_dataloader):
#     video_data = data[0]  # B inpt+pred C H W
#     video_data = data[0].squeeze()[::3]  # B, 24, 3, 96,196
#     video_data = (video_data+1)/2  # 8 3 96 196
#     video_list = [video_data[i] for i in range(video_data.shape[0])]
#     video_cat = torch.cat(video_list, dim=-1)
#     save_image(video_cat, dataset+'.jpg')
#     break
