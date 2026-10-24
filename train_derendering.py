
#!/usr/bin/env python
# coding: utf-8
"""
reader: Huilin XU
date: 2022/7/23 
description :训练keypointNet,提取图像中的目标位置
训练命令:python train_derendering.py --epoch 300 --dataset blocktowerCF --n_coefficients 8
"""
from random import randint
from torchvision.utils import save_image
from datetime import datetime
import os
import math
from pickle import TRUE
from re import T
# from re import L
# from this import d
import torch
import argparse

from Datasets.simb_loader import SimB, get_config
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import random
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import json
from tensorboardX import SummaryWriter
from src.KeypointNet.Derendering import Derendering
from src.KeypointNet.Derendering_with_rotate import Derendering_with_rotate
from src.KeypointNet.Derendering_newest import Derendering_newest
import matplotlib.pyplot as plt
import cv2
torch.autograd.set_detect_anomaly = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def update_json(dir, new_data):
    with open(dir, 'r') as f:
        old_data = json.load(f)
        old_data.update(new_data)
    with open(dir, 'w') as f:
        json.dump(old_data, f, indent=2)


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=0, type=int,
                    help="Number of Epoch for training. Can be set to 0 for evaluation")
parser.add_argument('--n_keypoints', default=4, type=int,
                    help="Number of keypoints to use")
parser.add_argument('--dataset', default="blocktowerCF", type=str,
                    choices=['ballsCF', 'blocktowerCF', 'simb', 'obj3d'],
                    help="Dataset to use")
parser.add_argument('--batchsize', default=32, type=int)
parser.add_argument('--lr', default=0.0001, type=float, help="Learning Rate")
parser.add_argument('--n_coefficients', default=4, type=int,
                    help="Number of rotation coefficients. 0 = no rotation filters")
parser.add_argument('--use_scale', action='store_true',
                    help="When n_coefficients=0, still use the 1-d scale/exist factor (pos+scale). "
                         "Ignored when n_coefficients>0 (scale is already used).")
parser.add_argument('--gpu', default=0, type=int,
                    help="CUDA device index. Replaces the old hardcoded cuda:2.")
parser.add_argument('--num_filter', default=32, type=int,
                    help="Number of filter in perception module")
parser.add_argument('--mode', default="fixed", type=str,  # dilatation filter bank 是指定的还是通过梯度下降学习�????
                    help="'fixed': use fixed dilatation filter bank. 'learned': learn the filters via gradient descent")
parser.add_argument('--seed', default=1024, type=int, help="Random seed")
parser.add_argument('--std', default=0.1, type=float,
                    help="std of guassian-like heatmap")
parser.add_argument('--filter_size', default=5, type=int,
                    help="size of rotate kerenl for distort heatmap")
parser.add_argument('--use_pixelshuffle', action='store_true',
                    help="decoder: replace bilinear upsampling with learned PixelShuffle (new arch; old checkpoints incompatible)")
parser.add_argument('--shared_object_decoder', action='store_true',
                    help="sum shared per-object basis maps before decoding, "
                    "making rendering permutation invariant to slot order")
parser.add_argument('--is_depth', action='store_true', help="是否考虑深度信息来变换热力图")
parser.add_argument('--condensation_loss',
                    action='store_true', help="是否考虑额外的热力图损失")
parser.add_argument('--separate_loss', action='store_true', help="是否考虑额外的稀疏损�????")
#parser.add_argument('--video_path', default="/Volumes/Samsung_T5/CoPhy_Dataset/CoPhy_112/blocktowerCF/4/", type=str, help="Path to video dataset")
# parser.add_argument('--video_path', default="/Datasets/Cophy/ballsCF/4",
#                     type=str, help="Path to video dataset")
parser.add_argument('--name', default='derendering',
                    type=str, help="Name for weight saving, derendering/derendering_with_rotate ")
parser.add_argument('--pre_train', action='store_true')
parser.add_argument('--weight_path', type=str, default='')
parser.add_argument('--resolution', default=64, type=int,
                    help="OBJ3D image resolution (64 or 128; native is 128)")
parser.add_argument('--obj3d_root', default='../Datasets/OBJ3D', type=str,
                    help="Root of the OBJ3D dataset containing train/val/test")
parser.add_argument('--frame_gap', default=0, type=int,
                    help="Source-target frame distance for OBJ3D pairs. "
                         "0 = int(0.2 * T)")
parser.add_argument('--mse_weight', default=10000., type=float,
                    help="Weight of the pixel MSE term in the reconstruction loss")
parser.add_argument('--gdl_weight', default=0.1, type=float,
                    help="Weight of the Sobel gradient (GDL) term in the reconstruction loss")
parser.add_argument('--perceptual_loss', action='store_true',
                    help="Add LPIPS perceptual loss on the reconstruction")
parser.add_argument('--perceptual_net', default='alex', choices=['alex', 'vgg'],
                    help="LPIPS backbone. alex = historical default; vgg matches the "
                    "DDLP eval metric (piqa LPIPS-vgg)")
parser.add_argument('--perceptual_weight', default=1.0, type=float,
                    help="Weight of the perceptual loss term")
parser.add_argument('--kpt_margin', default=0.0, type=float,
                    help="allow keypoints to range [-1-m, 1+m] so half-entered "
                    "objects keep a correct centroid (B1). 0 = historical "
                    "behavior; >0 changes the coordinate system -> retrain "
                    "stage-1 and rebuild stage-2 caches")
parser.add_argument('--temporal_weight', default=0.0, type=float,
                    help="T1: hinge loss on adjacent-frame appearance-coefficient "
                    "change, relu(|d_app| - temporal_tau). obj3d only; 0 = off")
parser.add_argument('--temporal_tau', default=0.05, type=float,
                    help="free margin of the T1 hinge: coefficient changes below "
                    "this magnitude are not penalized (measured natural "
                    "adjacent-frame change is ~0.02-0.05)")
parser.add_argument('--early_boost', default=0.0, type=float,
                    help="probability of sampling the source frame from frames "
                    "[0, 15] instead of the whole episode (oversample early "
                    "frames with half-entered border balls). obj3d only; 0 = off")
parser.add_argument('--shift_aug', default=0, type=int,
                    help="border-clip augmentation: replicate-pad frames by this "
                    "many pixels and randomly translate, so balls get clipped at "
                    "borders with centers outside the frame. Combine with "
                    "--kpt_margin >= 2*shift_aug/resolution. obj3d only; 0 = off")
parser.add_argument('--saliency_weight', default=0.0, type=float,
                    help="weight of the saliency-weighted MSE term (same scale "
                    "as mse_weight). Pixels deviating from a blurred background "
                    "estimate get upweighted ~1+saliency_gamma x, so dropping "
                    "small objects (balls are ~1%% of pixels) becomes expensive. "
                    "0 = off")
parser.add_argument('--saliency_gamma', default=9.0, type=float,
                    help="concentration of the saliency weight map")
parser.add_argument('--saliency_blur', default=15, type=int,
                    help="kernel size of the blur background estimate")
parser.add_argument('--gap_range', default='', type=str,
                    help="'lo,hi' to sample the source-target frame distance per "
                    "sample instead of the fixed frame_gap. Lets early frames be "
                    "reconstruction targets (the fixed gap of 20 never allows it) "
                    "and matches the gap range seen at stage-2 eval. obj3d only")
parser.add_argument('--run_tag', default='', type=str,
                    help="suffix appended to the HH:MM log_dir name, so several "
                    "runs launched in the same minute don't share a directory")
parser.add_argument('--output_dir', default='', type=str,
                    help="deterministic training output directory; empty keeps "
                         "the historical timestamp layout")
parser.add_argument('--cover_weight', default=0.0, type=float,
                    help="weight of the keypoint-coverage loss (MINT-style "
                    "detection supervision): every salient pixel must be covered "
                    "by some keypoint heatmap. Pairs with condensation_loss as "
                    "recall/precision. 0 = off")
parser.add_argument('--kpt_temporal_weight', default=0.0, type=float,
                    help="hinge on how far a keypoint slot may move between "
                    "adjacent frames, relu(|dp| - kpt_temporal_tau). Stage-1 "
                    "trains on frame pairs, so nothing stops slot k from "
                    "teleporting between objects; the resulting keypoint "
                    "sequence has no object permanence and no dynamics model "
                    "can extrapolate it. 0 = off")
parser.add_argument('--kpt_temporal_tau', default=0.06, type=float,
                    help="free margin of the keypoint hinge, in normalized "
                    "coordinates")
parser.add_argument('--probe_weight', default=0.0, type=float,
                    help="T2: weight of the predictability-probe loss. A small "
                    "MLP predicts the adjacent frame's coefficients from the "
                    "target frame's; its error backprops into the encoder, "
                    "penalizing unpredictable state components. obj3d only")
args = parser.parse_args()
print(args)
use_cuda = True
device = torch.device(
    f"cuda:{args.gpu}" if torch.cuda.is_available() and use_cuda else "cpu"
)

MSE = nn.MSELoss()
BATCHSIZE = 32

if args.dataset == 'simb':
    img_channel = 3
    img_width = 64
    img_height = 64
elif args.dataset == 'obj3d':
    img_width = args.resolution
    img_height = args.resolution
    img_channel = 3
else:
    img_width = 112
    img_height = 112
    img_channel = 3
print('img_width', img_width, 'img_height',
      img_height, 'img_channel', img_channel)

# 保持的路�????
SAVE_PATH = 'src/KeypointNet'


def evaluate():  # 两张图像的重�????
    print(args)
    if args.dataset == 'obj3d':
        dataloader = DataLoader(
            OBJ3D_Video(mode='test', resolution=args.resolution,
                        sampling_mode='fix', root=args.obj3d_root,
                        frame_gap=args.frame_gap),
            batch_size=8, num_workers=4, pin_memory=True)
    elif args.dataset == 'simb':
        dataloader = test_dataloader
    else:
        dataloader = DataLoader(
            datasets[args.dataset](mode='test', resolution=112,
                                   sampling_mode='fix'),
            batch_size=8, num_workers=4, pin_memory=True)

    # Path to the saved weights
    state_dict = torch.load(os.path.join(
        args.weight_path), map_location=device)

    model = Derendering(n_keypoints=args.n_keypoints,
                        mode=args.mode,
                        n_coefficients=args.n_coefficients,
                        use_scale=args.use_scale,
                        device=device, img_width=img_width, img_height=img_height,
                        img_channel=img_channel, num_filter=args.num_filter,
                        std=args.std, filter_size=args.filter_size).to(device)

    model.load_state_dict(state_dict)
    return validate(model, dataloader, viz=True)


def vizu(source, target, target_hat, target_keypoints, log_dir, idx):
    """
    Functions:可视化原图像、目标图像、提取的关键点和重建目标图像的结�???? 2*2
    Input: [B C H W]
    """
    # source_img target_img rec_target keypoints
    B, C, H, W = source.shape

    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    source = source.detach().cpu().numpy()[-1].transpose((1, 2, 0))  # 64 64 1
    target = target.detach().cpu().numpy()[-1].transpose((1, 2, 0))
    pred = target_hat.detach().cpu().numpy()[-1].transpose((1, 2, 0))

    source = (source + 1) / 2
    target = (target + 1) / 2
    pred = (pred + 1) / 2

    ax1.imshow(source.squeeze())
    ax2.imshow(target.squeeze())
    ax3.imshow(target.squeeze())
    ax4.imshow(pred.squeeze())

    keypoints_target = target_keypoints.detach().cpu().numpy()[-1]
    K = keypoints_target.shape[0]
    # X, Y = (keypoints_target[:, 1] + 1) / 2 * \
    #     112, (keypoints_target[:, 0] + 1) / 2 * 112  # X,Y�????2D坐标 [0,112]
    X, Y = (keypoints_target[:, 1] + 1) / 2 * \
        W, (keypoints_target[:, 0] + 1) / \
        2 * H  # X,Y�????2D坐标 [0,112]
    c_bank = ['red', 'green',  'blue', 'yellow']
    #c_bank = ['green',  'blue', 'red',  'yellow']
    for k in range(K):
        ax3.scatter(X[k], Y[k], c=c_bank[k], s=15)
    ax4.scatter(X, Y, c='white', s=15)

    ax1.axis("off")  # 取消坐标�????
    ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")

    ax1.set_title("Source image")
    ax2.set_title("Target image")
    ax3.set_title("Keypoints")
    ax4.set_title("Reconstruction")

    plt.tight_layout()
    plt.show()

    log_dir = os.path.join(log_dir, 'vis')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    plt.savefig(os.path.join(
        log_dir, 'video_'+str(idx[-1])+'_keypointNet_result'+'.png'))
    # plt.savefig(os.path.join(
    #     log_dir, 'video_'+str(idx[-1].item())+'_keypointNet_result'+'.png'))


def validate(model, dataloader, viz=False, log_dir='', h=1, desc='Valdation'):
    """
    验证集可视化结果, print Val_Loss & compute PSNR
    只考虑像素级别的损失，一些额外的损失不计�????
    """
    model.eval()
    list_loss = []
    list_mse_loss = []
    list_psnr = []
    with torch.no_grad():
        # for i, x in enumerate(tqdm(dataloader, desc="Validation")):
        for i, x in enumerate(tqdm(dataloader, desc=desc)):

            if args.dataset == 'simb':
                idx = x['ex']
                rgb = x['rgb_ab'].to(device)
                video_len = rgb.shape[1]
                t = randint(0, int(video_len*0.2))
                r = t+int(0.2*video_len)
                source = rgb[:, t]  # B  C H W
                target = rgb[:, r]  # B  C H W
            else:

                rgb = x['rgb_ab'].to(device)
                #rgb = x['rgb_cd'].to(device)
                source, target = rgb[:, 0], rgb[:, h]
                idx = x['ex']

            # B, DELTA_T, C, H, W = target.shape
            # for t in range(DELTA_T):
            #     out = model(source, target[:, t])  # B T C H W
            #     costs += get_loss(target[:, t],
            #                       out["target"], img_width, img_channel)
            # out_list = []
            # out_keypoint_list = []
            # B, DELTA_T, C, H, W = target.shape

            # for t in range(DELTA_T):
            #     tgt_hat = model(source, target[:, t])  # B DELTA_T C H W
            #     out_list.append(tgt_hat["target"])
            #     out_keypoint_list.append(tgt_hat["target"])

            # target = target.contiguous().view(-1, C, H, W)  # B*DELTA_T C H W
            # if DELTA_T == 1:
            #     out = out_list[0]
            #     out_keypoint = out_keypoint_list[0]
            #     costs = get_loss(target, out, img_width, img_channel)

            # else:
            #     out = torch.stack(out_list, dim=1).view(-1,
            #                                             C, H, W)  # B*DELTA_T C H
            #     out_keypoint = torch.cat(out_list, dim=0)
            #     costs = get_loss(target,
            #                      out, img_width, img_channel)
            # target = source # 自己重建
            out = model(source, target)
            costs = get_loss(target, out["target"],
                             img_width, img_channel, img_height)
            # [B C H W ] -> [B H W C ] numpy计算metric
            cd = target.cpu().numpy().transpose(0, 2, 3, 1)
            predictions = out["target"].cpu().numpy().transpose(0, 2, 3, 1)
            #predictions = out.cpu().numpy().transpose(0, 2, 3, 1)
            costs.append(  # 像素�????-1~1 所�???? data_range = 2 PSNR
                np.mean([peak_signal_noise_ratio(cd[b], predictions[b], data_range=2) for b in range(cd.shape[0])]))
            loss = costs[0]
            # if args.condensation_loss:
            #     loss += detect_loss(out["target_heatmap"])
            # if args.separate_loss:
            #     loss += dist_loss(out["target_keypoints"], separation_loss_sigma=1e-1)
            list_loss.append(loss.cpu().detach().numpy())
            list_mse_loss.append(costs[1].cpu().detach().numpy())
            list_psnr.append(costs[-1])

            if viz and ((i+1) % 10 == 0 or i == 0):  # 每个batch都保持吗??
                # source_img target_img rec_target keypoints
                vizu(source, target, out["target"],
                     out["target_keypoints"], log_dir, idx)
                # vizu(source, target, out,
                #      out_keypoint, log_dir, idx)

    return np.mean(list_loss), np.mean(list_mse_loss), np.mean(list_psnr)


def compute_grad(rgb, width, channel, height):
    """
    Functions: sobel算子卷积: 横向和纵�????---边缘检�????

    G = (rgb*filter_1)^2+(rgb*filter_2)^2
    """
    rgb = rgb.view(-1, channel, height, width)
    a = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]]).expand(1, channel, 3, 3).to(device)
    b = torch.Tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]]).expand(1, channel, 3, 3).to(device)
    sobel = (F.conv2d(rgb, a, padding=1) ** 2 +
             F.conv2d(rgb, b, padding=1) ** 2).squeeze(1)

    return sobel


def get_loss(target, target_hat, width, channel, height):
    """
    Functions:计算重建损失+边缘检测的损失
    Input : 原始图像与重建图�????
    Output: [loss, loss_rgb*1000, loss_grad* 0.1]
    loss = mse_weight * loss_rgb + gdl_weight * loss_grad

    """
    loss_rgb = MSE(target_hat, target)  # 已经是标量了
    loss_grad = MSE(compute_grad(target, width, channel, height),
                    compute_grad(target_hat, width, channel, height))  # sobel卷积的结�????

    loss = args.mse_weight * loss_rgb + args.gdl_weight * loss_grad  # 额外计算了梯�????

    return [loss, loss_rgb*1000, loss_grad]


def _masked_hinge(values, tau, mask):
    """Mean relu(values - tau) over True entries of mask; 0 if mask is empty."""
    pen = F.relu(values - tau)
    if mask is None:
        return pen.mean()
    weight = mask.float()
    denom = weight.sum().clamp(min=1.0)
    return (pen * weight).sum() / denom


_lpips_fn = None


def get_lpips_fn():
    """Lazily build the LPIPS perceptual loss; inputs are in [-1, 1]."""
    global _lpips_fn
    if _lpips_fn is None:
        try:
            import lpips
            _lpips_fn = lpips.LPIPS(net=args.perceptual_net, verbose=False).to(device)
            for p in _lpips_fn.parameters():
                p.requires_grad_(False)
        except Exception as e:
            print('WARNING: lpips unavailable (%s); perceptual loss disabled' % e)
            _lpips_fn = False
    return _lpips_fn if _lpips_fn is not False else None


def detect_loss(weight_matrix, gama=0.1):
    """
    输入 图像对应的热力图 B K H W  default gama=0.01, 幅度范围0-1
    """
    weight_shape = weight_matrix.shape  # B K H W
    #assert len(weight_shape) == 5, weight_shape
    if len(weight_shape) == 5:
        weight_matrix1 = weight_matrix.view(weight_matrix.size(0),
                                            weight_matrix.size(1), weight_matrix.size(2), -1)
    else:
        weight_matrix1 = weight_matrix.view(weight_matrix.size(0),
                                            weight_matrix.size(1), -1)
    heatmap_max = torch.max(weight_matrix1, dim=-1)[0]
    heatmap_mean = torch.mean(weight_matrix1, dim=-1)
    max_min = torch.min(heatmap_max-heatmap_mean, dim=-1)[0]  # 优化最差的那个 B
    #incite = -torch.sum(max_min)
    incite = -torch.mean(max_min)  # 防止T和B造成的影�????
    return gama*incite


def dist_loss(key_location, separation_loss_scale=2e-2, separation_loss_sigma=2e-3):
    """
    输入 关键点的 坐标 B K 2  default gama=0.01
    """
    #   cfg.separation_loss_scale = 2e-2
    #   cfg.separation_loss_sigma = 2e-3
    B, K, _ = key_location.shape
    x = key_location[..., 0]  # B K
    y = key_location[..., 1]  # B K
    d = ((x[:, :, None] - x[:,  None, :]) ** 2.0 +
         (y[:, :, None] - y[:,  None, :]) ** 2.0)  # T K K 计算成对的几何距�???? # B K K
    loss_matrix = torch.exp(-d / (2.0 * separation_loss_sigma ** 2.0))  # B K K
    # Mean across batch. k k  对角线都�????1
    loss_matrix = torch.mean(loss_matrix, dim=0)
    loss = torch.sum(loss_matrix)  # Sum matrix elements
    # print(loss)
    # Subtract sum of values on diagonal, which are always 1:
    loss = loss - K  # 减去对角线的�????
    return loss


if args.dataset == 'simb':
    C = get_config('simb')
    args.given_frames = 4
    args.horizon = 40
    print(C.RPIN.INPUT_SIZE, C.RPIN.PRED_SIZE_TRAIN, C.RPIN.PRED_SIZE_TEST)
    C.RPIN.INPUT_SIZE = args.given_frames
    C.RPIN.PRED_SIZE_TRAIN = args.horizon
    C.RPIN.PRED_SIZE_TEST = args.horizon
    print(C.RPIN.INPUT_SIZE, C.RPIN.PRED_SIZE_TRAIN, C.RPIN.PRED_SIZE_TEST)
    train_dataloader = DataLoader(SimB('Datasets/'+args.dataset, split='train', C=C, image_ext='.jpg'),
                                  shuffle=True, batch_size=args.batchsize, num_workers=4)
    test_dataloader = DataLoader(SimB('Datasets/'+args.dataset, split='test', C=C, image_ext='.jpg'),
                                 shuffle=False, batch_size=args.batchsize//2, num_workers=4)
    val_dataloader = test_dataloader
elif args.dataset == 'obj3d':
    from Dataloader.OBJ3D_Loader import OBJ3D_Video

    _want_adj = (args.temporal_weight > 0 or args.probe_weight > 0
                 or args.kpt_temporal_weight > 0)
    _gap_range = None
    if args.gap_range:
        _lo, _hi = args.gap_range.split(',')
        _gap_range = (int(_lo), int(_hi))
    train_dataloader = DataLoader(
        OBJ3D_Video(mode='train', resolution=args.resolution,
                    sampling_mode='rand', root=args.obj3d_root,
                    frame_gap=args.frame_gap, temporal_neighbor=_want_adj,
                    early_boost=args.early_boost, shift_aug=args.shift_aug,
                    gap_range=_gap_range),
        batch_size=args.batchsize, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(
        OBJ3D_Video(mode='val', resolution=args.resolution,
                    sampling_mode='fix', root=args.obj3d_root,
                    frame_gap=args.frame_gap),
        batch_size=args.batchsize//2, num_workers=4, pin_memory=True, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(
        OBJ3D_Video(mode='test', resolution=args.resolution,
                    sampling_mode='fix', root=args.obj3d_root,
                    frame_gap=args.frame_gap),
        batch_size=args.batchsize//2, num_workers=4, pin_memory=True, shuffle=False)

else:
    from Dataloader.general_Video_Loaders import blocktowerCF_Video, ballsCF_Video
    datasets = {"blocktowerCF": blocktowerCF_Video,
                "ballsCF": ballsCF_Video}

    train_dataloader = DataLoader(
        datasets[args.dataset](mode='train_large', resolution=112,
                               sampling_mode="rand", env=args.dataset),
        batch_size=args.batchsize, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
    # fix 代表 一个视频中读取两张图片,相隔0.15*video_len，第一张图�????0.15*video_len处，下采样训练keypointNet
    val_dataloader = DataLoader(
        datasets[args.dataset](mode='val', resolution=112,
                               sampling_mode='fix', env=args.dataset, load_cd=False, load_ab=True),
        batch_size=args.batchsize//2, num_workers=4, pin_memory=True, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(
        datasets[args.dataset](mode='test', resolution=112,
                               sampling_mode='fix', env=args.dataset, load_cd=True, load_ab=True),
        batch_size=1, num_workers=1, pin_memory=True)  # num_worker 有效提高加载数据集的速度

if args.name == 'derendering':
    model = Derendering(n_keypoints=args.n_keypoints,
                        mode=args.mode,
                        n_coefficients=args.n_coefficients,
                        use_scale=args.use_scale,
                        device=device, img_width=img_width, img_height=img_height, img_channel=img_channel, num_filter=args.num_filter,
                        std=args.std, filter_size=args.filter_size,
                        use_pixelshuffle=args.use_pixelshuffle,
                        kpt_margin=args.kpt_margin,
                        shared_object_decoder=args.shared_object_decoder).to(device)
elif args.name == 'derendering_with_rotate':
    model = Derendering_with_rotate(n_keypoints=args.n_keypoints, is_depth=args.is_depth,
                                    n_coefficients=args.n_coefficients,
                                    device=device, img_width=img_width, img_channel=img_channel, std=args.std, filter_size=args.filter_size).to(device)

else:
    model = Derendering_newest(n_keypoints=args.n_keypoints, is_depth=args.is_depth,
                               n_coefficients=args.n_coefficients,
                               device=device, img_width=img_width, img_channel=img_channel, std=args.std, filter_size=args.filter_size).to(device)


def main():
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    now_day = datetime.now().date()
    now_time = datetime.now().time()
    # log_dir = os.path.join(SAVE_PATH, args.dataset,
    #                        str(now_day)+'-' + str(now_time)[0:5])
    if args.n_coefficients > 0:
        variant = "pos_scale_rot"
    elif args.use_scale:
        variant = "pos_scale"
    else:
        variant = "pos"
    # log_dir is minute-resolution, so concurrent launches within the same
    # minute would share it and overwrite each other's checkpoints; --run_tag
    # appends a suffix to keep them apart.
    _stamp = str(now_time)[0:5]
    if args.run_tag:
        _stamp = '%s_%s' % (_stamp, args.run_tag)
    log_dir = args.output_dir or os.path.join(
        SAVE_PATH, args.dataset, variant, str(now_day), _stamp)

    os.makedirs(log_dir, exist_ok=True)
    print('log_dir:', log_dir)
    json_path = os.path.join(log_dir, 'log.json')
    with open(json_path, 'w') as f:
        json.dump(vars(args), f, indent=2)  # 一行，,indent=2：多�????

    writer = SummaryWriter(log_dir)

    # rand 代表 一个视频中读取两张图片,相隔0.15*video_len，第一张图片位�????0~0.15*video_len之间

    if args.pre_train:
        pt_path = os.path.join(args.weight_path)
        print('Loading pre_training weight file from %s' % pt_path)
        model.load_state_dict(torch.load(pt_path, map_location=device))

    # T2 predictability probe: predicts the adjacent frame's appearance
    # coefficients from the target frame's. Its residual = the unpredictable
    # (jitter) component; backpropping it into the encoder pressures the
    # states to be predictable without forbidding genuine (learnable) change.
    probe = None
    if args.probe_weight > 0:
        _app_dim = args.n_keypoints * (args.n_coefficients + 1)
        probe = nn.Sequential(
            nn.Linear(_app_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, _app_dim),
        ).to(device)
        print('predictability probe on, dim=%d' % _app_dim)

    trained_params = list(model.parameters())
    if probe is not None:
        trained_params += list(probe.parameters())
    optim = torch.optim.Adam(trained_params, lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='max', patience=5, factor=0.5, verbose=True, min_lr=1e-8)  # 以psnr为评价指�????

    best_val_loss = math.inf
    best_val_mse_loss = math.inf
    best_train_loss = math.inf
    best_val_psnr = 0.
    best_train_epoch = 0
    best_val_epoch = 0

    for epoch in range(args.epoch):

        list_train_loss = []
        list_train_mse_loss = []
        list_separate_loss = []
        list_condensation_loss = []
        list_perceptual_loss = []
        list_temporal_loss = []
        list_probe_loss = []
        list_saliency_loss = []
        list_cover_loss = []
        list_kpt_temporal_loss = []
        model.train()
        print("=== EPOCH ", epoch + 1, "/", args.epoch, " ===",
              'lr=', optim.param_groups[0]['lr'])
        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):

            if args.dataset == 'simb':
                rgb = x['rgb_ab'].to(device)
                video_len = rgb.shape[1]
                t = randint(0, int(video_len*0.15))
                r = t+int(0.15*video_len)
                source = rgb[:, t]  # B  C H W
                target = rgb[:, r].unsqueeze(1)  # B  C H W
            else:

                rgb = x['rgb_ab'].to(device)
                # 随机选取�????,距离0.15*video_len
                source, target = rgb[:, 0], rgb[:, 1:]

            out_list = []
            heatmap_list = []
            coord_list = []
            param_list = []
            B, DELTA_T, C, H, W = target.shape
            for t in range(DELTA_T):
                tgt_hat = model(source, target[:, t])
                out_list.append(tgt_hat["target"])
                heatmap_list.append(tgt_hat["target_heatmap"])
                coord_list.append(tgt_hat["target_keypoints"])
                param_list.append(tgt_hat["target_parameter"])
            target_seq = target

            out = torch.stack(out_list, dim=1).reshape(-1, C, H, W)
            target = target_seq.reshape(-1, C, H, W)
            heatmap = torch.stack(heatmap_list, dim=1)  # [B, T, K, h, w]
            coord = torch.stack(coord_list, dim=1)      # [B, T, K, 2]
            params = torch.stack(param_list, dim=1)     # [B, T, K, C+1]
            cost = get_loss(target, out, img_width, img_channel, img_height)
            condensation_loss = detect_loss(
                F.tanh(heatmap.reshape(-1, *heatmap.shape[2:])), gama=5)
            separate_loss = dist_loss(
                coord.reshape(-1, args.n_keypoints, coord.shape[-1]),
                separation_loss_sigma=1e-1)
            if i == 0 and epoch == 0:
                print(target.shape, out.shape, B, DELTA_T)
            # cost /= DELTA_T
            # costs += get_loss(target[:, t],
            #                       out["target"], img_width, img_channel)
            # out = model(source, target)
            # cost = get_loss(target, out["target"], img_width, img_channel)

            optim.zero_grad()
            # cost = [loss, loss_rgb, loss_grad]
            # cost[0].backward()
            loss = cost[0]
            if args.condensation_loss:
                loss += condensation_loss
                # cost.append(condensation_loss)
            if args.separate_loss:
                loss += separate_loss
            perceptual = torch.zeros((), device=device)
            if args.perceptual_loss:
                lpips_fn = get_lpips_fn()
                if lpips_fn is not None:
                    perceptual = lpips_fn(out, target).mean()
                    loss = loss + args.perceptual_weight * perceptual

            # saliency-weighted MSE: upweight pixels that deviate from a
            # blurred background estimate (balls on a smooth floor), so the
            # loss no longer tolerates dropping small/half-visible objects.
            saliency = torch.zeros((), device=device)
            if args.saliency_weight > 0:
                k = args.saliency_blur
                blur = F.avg_pool2d(
                    F.pad(target, (k // 2, k // 2, k // 2, k // 2),
                          mode='reflect'), k, stride=1)
                sal = (target - blur).abs().mean(1, keepdim=True)
                wmap = 1.0 + args.saliency_gamma * sal / (sal.mean() + 1e-6)
                wmap = wmap / wmap.mean()
                saliency = ((out - target) ** 2 * wmap).mean()
                loss = loss + args.saliency_weight * saliency

            # Coverage: every salient pixel must sit under some keypoint heatmap.
            cover = torch.zeros((), device=device)
            if args.cover_weight > 0:
                # spatial-softmax per keypoint, then take the best slot at
                # each pixel. tanh(sum(.)) saturates to 1 with K=25 and
                # gives a dead -1 loss; the max of a distribution does not.
                hm_flat = heatmap.reshape(-1, heatmap.shape[2],
                                          heatmap.shape[3], heatmap.shape[4])
                hm_prob = F.softmax(hm_flat.flatten(2), dim=-1).reshape_as(hm_flat)
                hm = hm_prob.max(dim=1, keepdim=True).values
                k = args.saliency_blur
                blur = F.avg_pool2d(
                    F.pad(target, (k // 2, k // 2, k // 2, k // 2),
                          mode='reflect'), k, stride=1)
                sal = (target - blur).abs().mean(1, keepdim=True)
                sal = sal / (sal.mean() + 1e-6)
                sal_ds = F.adaptive_avg_pool2d(sal, hm.shape[-2:])
                cover = -(sal_ds * hm).sum() / (sal_ds.sum() + 1e-6)
                loss = loss + args.cover_weight * cover

            # temporal / probe / keypoint trajectory
            temporal = torch.zeros((), device=device)
            probe_loss = torch.zeros((), device=device)
            kpt_temporal = torch.zeros((), device=device)
            want_traj = args.kpt_temporal_weight > 0
            if want_traj and ('rgb_adj' in x or DELTA_T >= 2):
                if DELTA_T >= 2:
                    kp0, _ = model.get_keypoints_only(source)
                    traj = torch.cat([kp0.unsqueeze(1), coord], dim=1)
                else:
                    adj = x['rgb_adj'].to(device)
                    kp_adj, params_adj = model.get_keypoints_only(adj)
                    traj = torch.stack(
                        [tgt_hat["target_keypoints"], kp_adj], dim=1)
                    if args.temporal_weight > 0:
                        d_app = (params_adj - tgt_hat["target_parameter"]).abs()
                        temporal = F.relu(d_app - args.temporal_tau).mean()
                        loss = loss + args.temporal_weight * temporal
                    if args.probe_weight > 0 and probe is not None:
                        params_tgt = tgt_hat["target_parameter"]
                        pred_adj = probe(params_tgt.reshape(B, -1))
                        probe_loss = F.mse_loss(
                            pred_adj, params_adj.reshape(B, -1))
                        loss = loss + args.probe_weight * probe_loss

                if args.kpt_temporal_weight > 0 and traj.shape[1] >= 2:
                    step = (traj[:, 1:, :, :2] - traj[:, :-1, :, :2]).norm(dim=-1)
                    kpt_temporal = _masked_hinge(
                        step, args.kpt_temporal_tau, None)
                    loss = loss + args.kpt_temporal_weight * kpt_temporal

            elif (args.temporal_weight > 0 or args.probe_weight > 0) \
                    and 'rgb_adj' in x:
                adj = x['rgb_adj'].to(device)
                _, params_adj = model.get_keypoints_only(adj)
                params_tgt = tgt_hat["target_parameter"]
                if args.temporal_weight > 0:
                    d_app = (params_adj - params_tgt).abs()
                    temporal = F.relu(d_app - args.temporal_tau).mean()
                    loss = loss + args.temporal_weight * temporal
                if args.probe_weight > 0 and probe is not None:
                    pred_adj = probe(params_tgt.reshape(B, -1))
                    probe_loss = F.mse_loss(
                        pred_adj, params_adj.reshape(B, -1))
                    loss = loss + args.probe_weight * probe_loss

            # cost[0].backward() # 靠靠靠靠�????
            loss.backward()
            optim.step()
            list_train_loss.append(cost[0].item())
            list_train_mse_loss.append(cost[1].item())
            list_separate_loss.append(separate_loss.item())  # 没有item会导致溢出的
            list_condensation_loss.append(condensation_loss.item())
            list_perceptual_loss.append(perceptual.item())
            list_temporal_loss.append(temporal.item())
            list_probe_loss.append(probe_loss.item())
            list_saliency_loss.append(saliency.item())
            list_cover_loss.append(cover.item())
            list_kpt_temporal_loss.append(kpt_temporal.item())
        # if epoch == 100:
        #     optim = torch.optim.Adam(trained_params, lr=args.lr)

        train_loss, train_mse_loss = np.mean(
            list_train_loss), np.mean(list_train_mse_loss)
        train_separate_loss, train_condensation_loss = np.mean(
            list_separate_loss), np.mean(list_condensation_loss)
        train_perceptual_loss = np.mean(list_perceptual_loss)
        train_temporal_loss = np.mean(list_temporal_loss)
        train_probe_loss = np.mean(list_probe_loss)
        val_loss, val_mse_loss, val_psnr = validate(
            model, val_dataloader, log_dir=log_dir)
        ## sched.step(val_loss)
        sched.step(val_psnr)

        writer.add_scalar('Train/lr', optim.param_groups[0]['lr'], epoch+1)
        writer.add_scalar('Train/loss', train_loss, epoch+1)
        writer.add_scalar('Train/mse_loss', train_mse_loss, epoch+1)
        writer.add_scalar('Train/separate_loss', train_separate_loss, epoch+1)
        writer.add_scalar('Train/condensation_loss',
                          train_condensation_loss, epoch+1)
        writer.add_scalar('Train/perceptual_loss', train_perceptual_loss, epoch+1)
        writer.add_scalar('Train/temporal_loss', train_temporal_loss, epoch+1)
        writer.add_scalar('Train/probe_loss', train_probe_loss, epoch+1)
        writer.add_scalar('Train/saliency_loss', np.mean(list_saliency_loss), epoch+1)
        writer.add_scalar('Train/cover_loss', np.mean(list_cover_loss), epoch+1)
        writer.add_scalar('Train/kpt_temporal_loss',
                          np.mean(list_kpt_temporal_loss), epoch+1)
        writer.add_scalar('Valid/loss', val_loss, epoch+1)
        writer.add_scalar('Valid/mse_loss', val_mse_loss, epoch+1)
        writer.add_scalar('Valid/psnr', val_psnr, epoch+1)
        print('Train loss = %f, Train mse_loss = %f, Train separate_loss = %f,Train condensation_loss = %f, cover = %f, kpt_step = %f' %
              (train_loss, train_mse_loss, train_separate_loss, train_condensation_loss,
               np.mean(list_cover_loss), np.mean(list_kpt_temporal_loss)))
        print('Val loss = %f, Val mse_loss = %f, Val PSNR= %f' %
              (val_loss, val_mse_loss, val_psnr))
        #torch.save(model.state_dict(), SAVE_PATH)
        torch.save(model.state_dict(), os.path.join(
            log_dir, args.dataset+'_4kpts.nn'))

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_epoch = epoch+1
            torch.save(model.state_dict(), os.path.join(
                log_dir, 'best_train_'+args.dataset+'_4kpts.nn'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch+1
            torch.save(model.state_dict(), os.path.join(
                log_dir, 'best_val_'+args.dataset+'_4kpts.nn'))

        if val_mse_loss < best_val_mse_loss:
            best_val_mse_loss = val_mse_loss

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr

        print('Best Train loss = %f @ Epoch %d' %
              (best_train_loss, best_train_epoch))
        print('Best Val loss = %f @ Epoch %d' %
              (best_val_loss, best_val_epoch))
        print(log_dir)
    val_loss, val_mse_loss, val_psnr = validate(
        model, val_dataloader, viz=False, log_dir=log_dir)

    print(' Val loss = %f, Val mse_loss = %f,Val PSNR = %f' %
          (val_loss, val_mse_loss, val_psnr))
    para_num = count_parameters(model)
    log_result = {'best_train_loss': float(best_train_loss),
                  'best_val_loss': float(best_val_loss), 'best_val_mse_loss': float(best_val_mse_loss),
                  'best_val_psnr': best_val_psnr, 'model_para_num': float(para_num)}

    update_json(json_path, log_result)
    print(args.name, ': parameter = ', para_num)
    print(json_path)


if __name__ == '__main__':
    para_num = count_parameters(model)
    print('parameter = ', para_num)
    if args.epoch == 0:
        pt_path = os.path.join(args.weight_path)
        print('Loading pre_training weight file from %s' % pt_path)
        model.load_state_dict(torch.load(pt_path, map_location=device))
        log_dir = os.path.dirname(args.weight_path)

        # val_loss, val_mse_loss, val_psnr = validate(
        #     model, val_dataloader, viz=False, log_dir=log_dir)
        # print(' Val loss = %f, Val mse_loss = %f,Val PSNR = %f' %
        #       (val_loss, val_mse_loss, val_psnr))
        test_loss, test_mse_loss, test_psnr = validate(
            model, test_dataloader, viz=False, log_dir=log_dir, desc='Test')
        print(' test loss = %f, test mse_loss = %f,test PSNR = %f' %
              (test_loss, test_mse_loss, test_psnr))
        log_result = {'val_loss': float(test_loss), '_val_mse_loss': float(test_mse_loss),
                  'val_psnr': test_psnr}
        json_path = os.path.join(log_dir, 'log.json')
        #update_json(json_path, log_result)
        # #检查连续性！ 时序上的连续�???? 参考空间的旋转�????
        # vis_dataloader = DataLoader(
        # datasets[args.dataset](mode='test', resolution=112,
        #                     sampling_mode='full',load_state=True,load_cd=True),
        # batch_size=1, num_workers=1, pin_memory=True,shuffle=False) # 一个实例查看参数的连续�????
        # one_batch_data = next(iter(vis_dataloader))
        # rgb = one_batch_data['rgb_ab'].to(device)
        # source, target = rgb[:, 0:1], rgb[:, 1:] # 1 1 C H W  1 149 C H W
        # target = torch.cat([source,target],dim=1)[:,0:25] # 1 150 C H W
        # print(target.shape)
        # B, T, C, H, W = target.shape
        # target = target.contiguous().view(-1, C, H, W)  # B*T C H W
        # target_keypoints_location, target_parameter = model.get_keypoints_only(target)

        # target_keypoints_location = target_keypoints_location.contiguous().view(
        #     B, T, -1, 2)  # B T K 2
        # target_parameter = target_parameter.contiguous().view(
        #     B, T, -1, args.n_coefficients + 1)  # B T K 2
        # target_pos = torch.cat([target_keypoints_location, target_parameter],dim=-1) # B T K S
        # vis_pos = target_pos[0, 0:2]# 第一个batch 的前5个时刻的 第一个目标的参数变化
        # gt_pose = (one_batch_data["pose_2D_ab"][0, 0:2]-56.)/56.
        # print(one_batch_data["pose_2D_ab"].shape)
        # print(one_batch_data["pose_2D_cd"].shape)
        # source_img = source.squeeze().detach(
        # ).cpu().numpy().transpose((1, 2, 0)) # H W C
        # plt.imsave('filename.jpeg', (source_img+1)/2)

        #out = model(source, target)
        # vizu(target, target, target,
        #      target_keypoints_location, log_dir='', idx=one_batch_data["ex"][0])
        # print(one_batch_data["ex"][0])
        # print(gt_pose)
        # print(vis_pos[...,0:2])

        """
        # test_dataloader = DataLoader(
        #     datasets[args.dataset](mode='test', resolution=112,
        #                            sampling_mode='full'),
        #     batch_size=1, num_workers=1, pin_memory=True)
        # json_path = os.path.join(log_dir, 'log.json')
        # list_psnr = []
        # list_mse = []
        # list_horizon = np.arange(1, 29)
        # viz = False
        # for h in range(0, 28):
        #     if h == 27:
        #         viz = True
        #     val_loss, val_mse_loss, val_psnr = validate(
        #         model, test_dataloader, viz=viz, log_dir=log_dir, h=h+1)  # 重建每一�????

        #     list_psnr.append(val_psnr)
        #     list_mse.append(val_mse_loss)
        #     # 测试集和验证集是一致的
        #     # log_result = {'test_loss': float(val_loss),
        #     #               'test_mse_loss': float(val_mse_loss), 'test_psnr': float(val_psnr)}
        #     # print(log_result)
        # print(np.mean(list_psnr), np.mean(list_mse))
        # print(val_psnr, val_mse_loss)  # 重建28帧的结果
        # fig = plt.figure(figsize=(16, 7))
        # ax1 = fig.add_subplot(121)
        # ax2 = fig.add_subplot(122)
        # ax1.plot(list_horizon, list_mse)
        # ax2.plot(list_horizon, list_psnr)
        # ax1.set_title("TEST_rec:MSE")
        # ax2.set_title("TEST_rec:PSNR")
        # ax1.set_xlabel('t/frame')
        # ax1.set_ylabel('MSE/e-3')
        # ax2.set_xlabel('t/frame')
        # ax2.set_ylabel('PSNR')
        # ax1.grid(ls='--')
        # ax2.grid(ls='--')
        # plt.tight_layout()
        # plt.savefig(os.path.join(
        #     log_dir, 'rec_result.png'))

        # update_json(json_path, log_result)
        # evaluate()
        """
    else:
        main()


# 测试一�???? 设计的额外的损失函数
# pt_path = os.path.join(args.weight_path)
# print('Loading pre_training weight file from %s' % pt_path)
# model.load_state_dict(torch.load(pt_path, map_location=device))
# one_batch_data = next(iter(test_dataloader))
# rgb = one_batch_data['rgb_ab'].to(device)
# source, target = rgb[:, 0], rgb[:, 1]
# print(source.shape)
# out = model(source, target)
# #target_keypoints_location, target_parameter = model.get_keypoints_only(target) #  B K 2
# costs = get_loss(target, out["target"], img_width, img_channel)
# costs.append(detect_loss(out["target_heatmap"]))
# print(costs[0].item(),costs[1].item()*10, costs[-2].item()*0.1,costs[-1].item())
# loc_separate_loss = dist_loss(out["target_keypoints"],separation_loss_sigma=1e-1) # 目前预训练的 1左右，若sigma = 0.1的话
# print(loc_separate_loss.item())
# pt_path = os.path.join(args.weight_path)
# print('Loading pre_training weight file from %s' % pt_path)
# model.load_state_dict(torch.load(pt_path, map_location=device))
# log_dir = os.path.dirname(args.weight_path)
# model.eval()
# with torch.no_grad():
#     # for i, x in enumerate(tqdm(dataloader, desc="Validation")):
#     for i, x in enumerate(vis_dataloader):
#         rgb = x['rgb_ab'].to(device)
#         source, target = rgb[:, 0], rgb[:, 1:]
#         idx = x['ex']
#         break
# B, T, C, H, W = target.shape
# target = target.contiguous().view(-1, C, H, W)  # B*T C H W
# target_keypoints_location, target_parameter = model.get_keypoints_only(target)
# target_keypoints_location = target_keypoints_location.contiguous().view(
#     B, T, -1, 2)  # B T K 2
# target_parameter = target_parameter.contiguous().view(
#     B, T, -1, args.n_coefficients + 1)  # B T K 2
# src_features = model.get_features_only(source)
# target_keypoints_location, target_parameter = model.get_keypoints_only(target)
# tgt_distort_heatmaps = model.get_keypoints_map(
#     src_features, target_keypoints_location, target_parameter)
# rec_tgt_imgs = model.create_img(
#     src_features, target_keypoints_location, target_parameter)
# print(source.shape)
# print(target.shape)
# print(src_features.shape)
# print(target_keypoints_location.shape)  # 4 T-1 4 2
# print(target_parameter.shape)
# target_pos = torch.cat([target_keypoints_location, target_parameter], dim=-1)
# print(target_pos[-1, 0:25:5].permute(1, 0, 2))
# print(tgt_distort_heatmaps.shape)
# video_idx = -1
# tgt = (target[video_idx] + 1) / 2
# tgt_heatmaps = tgt_distort_heatmaps[video_idx]
# save_image(tgt, 'targt_img.png')
# vis_heatmaps_list = [tgt_heatmaps[i] for i in range(tgt_heatmaps.shape[0])]
# vis_heatmaps = torch.cat(vis_heatmaps_list, dim=-1).unsqueeze(0)
# print(vis_heatmaps.shape)
# save_image(vis_heatmaps, 'targt_heatmap.png')

# rec_tgt = (rec_tgt_imgs[video_idx] + 1) / 2
# save_image(rec_tgt, 'rec_tgt.png')
# out = model(source, target)
# rec_gt_tgt = (out["target"][video_idx] + 1) / 2
# save_image(rec_gt_tgt, 'rec_gt_tgt.png')
# print(target_keypoints_location[0])
# print(target_parameter[0])
# 试一试球的索�????


# with torch.no_grad():
#     # for i, x in enumerate(tqdm(dataloader, desc="Validation")):
#     for i, x in enumerate(tqdm(test_dataloader)):  # B C H W

#             rgb = x['rgb_ab'].to(device)
#             #rgb = x['rgb_cd'].to(device)
#             source, target = rgb[:, 0], rgb[:, 1]
#             print('Input shape:', source.shape)
#             idx = x['ex']
#         break
# print("{} | {} | {}".format("Model", "Params(M)", "FLOPs(G)"))
# print("---|---|---")
# total_ops, total_params = profile(
#     model, (source, source))  # 考虑�????
# print("{} | {:.2f} | {:.2f}".format(args.name, total_params / 1e6, total_ops / 1e9,
#                                     ))
