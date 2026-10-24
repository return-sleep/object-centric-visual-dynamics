#!/usr/bin/env python
# coding: utf-8
"""
author: Huilin XU
date: 2023/2/28
description : 在KTH上测试我们的动态模型
test: python train_my_model.py --epoch 0 --weight_path 
python train_my_model.py --epoch 0 --given_frames 3 --num_gn 2 --Inference_model predictor_module_with_appr --consider_appr
"""

import pandas as pd
import os
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
from torch.utils.data import DataLoader
from utils import *
from src.baseline_model.FID_metric import calculate_fid
from src.KeypointNet.Derendering import Derendering
from src.baseline_model.GN_modules import predictor_module, predictor_module_with_appr, predictor_module_apprmap
from src.inference_module.record_dynamic import *
from Datasets.simb_loader import SimB, get_config
# from train_derendering import validate as Perception_validate
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings("ignore")


def visualize(cd, cd_hat, log_dir, batch_idx):
    import matplotlib.pyplot as plt
    plt.style.use("seaborn")
    # [b horzion C H W] 每个batch 的第一个视频预测结果
    cd = (cd[-1].numpy().transpose(0, 2, 3, 1) + 1) / 2  # [-1,1] -> [0,1]
    cd_hat = (cd_hat[-1].numpy().transpose(0, 2, 3, 1) + 1) / 2
    fig, ax = plt.subplots(2, 8, figsize=(12, 4))

    # 预测空间分成8分
    idx = np.linspace(0, cd.shape[0] - 1, 8).astype(int)
    # print(idx)
    for k in range(8):
        ax[0][k].imshow(cd[idx[k]])
        ax[1][k].imshow(cd_hat[idx[k]])
        ax[0][k].axis("off")
        ax[1][k].axis("off")
        ax[0][k].set_title('t='+str(idx[k]))
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, batch_idx + '_compare.jpg'))


def creat_state_squence(ab, n=2):
    """
    创建新的状态(引入前n个时刻的速度)
    """
    B, T, K, S = ab.shape
    num_coef = S - 2
    if n == 0:
        new_ab = ab
    else:
        new_ab = torch.zeros(B, T-n, K, S+n*2).to(ab.device)
        new_ab[..., :2] = ab[:, n:, :, :2]  # 坐标
        new_ab[..., -num_coef:] = ab[:, n:, :, -num_coef:]
        # if S == 7:
        #     new_ab[..., -5:] = ab[:, n:, :, -5:]  # 5个外观系数 [B T-2 K 2+2+2+5]

        for i in range(n):
            if i == 0:
                new_ab[..., 2:4] = ab[:, n:, :, :2] - ab[:, n-1:-1, :, :2]
            else:
                new_ab[..., 2*(i+1):2*(i+2)] = ab[:, n-i:-i, :,
                                                  :2] - ab[:, n-1-i:-1-i, :, :2]
    return new_ab


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=300, type=int,
                        help="Number of Epoch for training. Can be set to 0 for evaluation")
    parser.add_argument('--batchsize', default=32, type=int,
                        help="Batchsize. Can be set to 1 for evaluation")
    parser.add_argument('--lr', default=5e-4, type=float, help="Learning rate")
    parser.add_argument('--dataset', default="simb", choices=["simb"],
                        help="Trajectory dataset (Simb in this release)")
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--given_frames', type=int, default=4,
                        help='Given previous n frames to predict states of future frames, must > 1')
    parser.add_argument('--horizon', type=int, default=20,
                        help='predict states of horizon-step future frames')
    parser.add_argument('--num_velocity', default=0, type=int)
    parser.add_argument('--name', default='my_model_with_env', type=str,
                        help="Name for weight saving, consider feature map around object positions")
    parser.add_argument('--Perception_model', type=str,
                        default='derendering', help='or derendering_with_rotate')
    parser.add_argument('--Perception_model_weight_path', type=str,
                        default='')
    parser.add_argument('--n_coefficients', default=4, type=int,
                        help="Number of coefficients to use")
    parser.add_argument('--n_keypoints', default=3, type=int,
                        help="Number of keypoints to use")
    parser.add_argument('--std', default=0.1, type=float,
                        help="std of guassian-like heatmap")
    parser.add_argument('--Inference_model', type=str,
                        default='predictor_module_with_appr', help='or else predictor_module_with_appr, predictor_module_apprmap')
    parser.add_argument('--consider_appr', action='store_true',
                        help='if consider feature map around each object')
    parser.add_argument('--weight_path', type=str,
                        default='')
    parser.add_argument('--is_rnn', action='store_true',
                        help='if use rnn why predict next state vector')
    parser.add_argument('--pos_emb', default=0, type=int,
                        help="dim of pos embedding like nerf")
    parser.add_argument('--num_gn', default=2, type=int)
    parser.add_argument('--num_layer', default=2, type=int)
    parser.add_argument('--embed_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)

    parser.add_argument('--noise_std', type=float,
                        default=0.0, help='add noise in training phase, 0.0003')
    parser.add_argument('--train_mode_verbose', default=0, type=int,
                        help='0: free running mode, 1: teacher forcing, 2: Curriculum Learning')
    parser.add_argument('--warm_epoch', default=10, type=int)
    parser.add_argument('--decay_epoch', default=50, type=int)
    parser.add_argument('--df_epoch', default=20, type=int)
    parser.add_argument('--autoenc_loss', action='store_true')
    parser.add_argument('--loss_type', default='absoluate', type=str,
                        help="predict states or states displacement, absoluate/relative/merge")
    parser.add_argument('--pos_weight', default=1.0, type=float)
    parser.add_argument('--discount_factor', action='store_true')
    parser.add_argument('--gpu', default=0, type=int,
                        help='CUDA device index')
    parser.add_argument('--data_root', default='Datasets/simb', type=str,
                        help='Simb directory containing train.hkl/test.hkl')
    parser.add_argument('--output_dir', default='', type=str,
                        help='deterministic training output directory')
    return parser.parse_args()


args = get_args()
print(args)

assert args.num_velocity < args.given_frames

if args.seed:  # 神经网络初始化固定
    random.seed(args.seed)
    torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(args.seed)  # 为所有GPU设置随机种子；
    torch.backends.cudnn.deterministic = True

MSE = nn.MSELoss()
use_cuda = True
device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available()
                      and use_cuda else "cpu")
# 准备数据集
img_channel = 3
if args.dataset == 'simb':
    img_width = 64
    img_height = 64
else:
    img_height = 96
    img_width = 192
print('img_width', img_width, 'img_height',
      img_height, 'img_channel', img_channel)
C = get_config(args.dataset)
print(C.RPIN.INPUT_SIZE, C.RPIN.PRED_SIZE_TRAIN, C.RPIN.PRED_SIZE_TEST)
C.RPIN.INPUT_SIZE = args.given_frames
C.RPIN.PRED_SIZE_TRAIN = args.horizon # 训练集的序列长度
C.RPIN.PRED_SIZE_TEST = args.horizon # 验证集的序列长度
# C.RPIN.PRED_SIZE_TEST = args.horizon *2
print(C.RPIN.INPUT_SIZE, C.RPIN.PRED_SIZE_TRAIN, C.RPIN.PRED_SIZE_TEST)
train_dataloader = DataLoader(SimB(args.data_root, split='train', C=C, image_ext='.jpg'),
                              shuffle=True, batch_size=args.batchsize, num_workers=4)
test_dataloader = DataLoader(SimB(args.data_root, split='test', C=C, image_ext='.jpg'),
                             shuffle=False, batch_size=args.batchsize, num_workers=4)
val_dataloader = test_dataloader

# 预训练的感知模块 (3个 x,y,u)
Perception_model = Derendering(n_keypoints=args.n_keypoints, mode='fixed', n_coefficients=args.n_coefficients,
                               img_channel=img_channel, img_width=img_width, device=device).to(device)
Perception_weight_path = (args.Perception_model_weight_path or
                          "src/KeypointNet/simb/2023-03-08/14-43/best_val_simb_4kpts.nn")
args.Perception_model_weight_path = Perception_weight_path
print('Load perception weight_path from:', Perception_weight_path)
Perception_model.load_state_dict(torch.load(
    Perception_weight_path, map_location=device))

# 定义动态模块
K = args.n_keypoints
S = 2 + 1 + args.n_coefficients + 2 * args.num_velocity
if args.consider_appr:
    appr_dim = 16
else:
    appr_dim = 0

Inference_model_dict = {'predictor_module': predictor_module, 'predictor_module_with_appr':
                        predictor_module_with_appr, 'predictor_module_apprmap': predictor_module_apprmap}
dynamics_model = Inference_model_dict[args.Inference_model](
    input_dim=S * (args.pos_emb*2+1), appr_dim=appr_dim, output_dim=S, num_gn=args.num_gn,
    embed_size=args.embed_size, hidden_size=args.hidden_size, is_rnn=args.is_rnn, num_layer=args.num_layer).to(device)
para_num = count_parameters(dynamics_model)
print('dynamics_model parameter = ', para_num)


def train_one_batch(rgb, Perception_model, model, args, p=0, batch_idx=1, epoch_idx=1, generate_img=False, df=1.0):

    Perception_model.eval()

    # 1. 提取各目标表征 --- 数据集准备
    B, T, C, H, W = rgb.shape
    inital_feature_map = Perception_model.get_features_only(
        rgb[:, 0])  # B 16 28 28
    state_1, state_2 = Perception_model.get_keypoints_only(
        rgb.contiguous().view(-1, C, H, W))
    pos = torch.cat([state_1, state_2], dim=-1)  # B*T K S
    K = pos.shape[-2]
    pos = pos.view(B, T, K, -1)
    new_pos = creat_state_squence(pos, n=args.num_velocity)
    input_state = new_pos[:, 0:args.given_frames-args.num_velocity]
    target_state = new_pos[:, args.given_frames-args.num_velocity:]  # B T K S

    # 2. 目标状态预测
    predict_list = []
    delta_list = []
    # Encoder 部分 历史帧
    for i in range(input_state.shape[1]-1):
        cur_obj = input_state[:, i]
        delta_state = model(inpt=cur_obj, args=args, env_features=inital_feature_map,
                            pos_dim=args.pos_emb, first_timestep=(i == 0))
        next_state = delta_state + cur_obj
        predict_list.append(next_state)
        delta_list.append(delta_state)
    # Decoder 部分 未来帧
    decoder_input = input_state[:, -1]
    use_teacher_forcing = np.random.binomial(1, p, args.horizon)
    for t in range(target_state.shape[1]):
        cur_obj = decoder_input
        delta_state = model(inpt=cur_obj, args=args, env_features=inital_feature_map,
                            pos_dim=args.pos_emb, first_timestep=(t == 0 and input_state.shape[1] == 1))
        next_state = delta_state + cur_obj
        predict_list.append(next_state)
        delta_list.append(delta_state)
        if use_teacher_forcing[t]:
            decoder_input = target_state[:, t] + torch.randn_like(
                target_state[:, t], device=target_state.device) * args.noise_std * p * math.sqrt(t)  # p=0 不考虑噪声，p=1完全考虑
        else:
            decoder_input = next_state

    gt_states = new_pos[:, 1:]
    predicted_states = torch.stack(predict_list, dim=1)

    loss, loss_pos, loss_vel, loss_coef, loss_exist = get_loss_with_df(
        predicted_states, gt_states, pos_weight=args.pos_weight, df=df, n_coef=args.n_coefficients+1)  # 说明可以通过啊这

    predicted_delta_states = torch.stack(delta_list, dim=1)
    gt_delta_state = new_pos[:, 1:]-new_pos[:, 0:-1]
    delta_loss = MSE(gt_delta_state, predicted_delta_states) * 1e3

    if batch_idx == 0 and epoch_idx == 0:
        print(rgb.shape)
        print(pos.shape, new_pos.shape)
        print(predicted_states.shape)
        # print(pos[0, 0, 0])  # 展示一下 

    if generate_img:  # 这里的代码只适合test_batchsize = 1
        # 给定起始帧，预测未来
        # test_dataloder B = 1 B 16 28 28  去除预测中的速度部分
        rec_states = torch.cat(
            [predicted_states[..., 0:2], predicted_states[..., -args.n_coefficients-1:]], dim=-1)
        # print("rec state:", rec_states.shape, 'inital_feature_map shape:',
        #       inital_feature_map.shape)  # B T C H W
        rec_rgb_list = []
        gt_rgb = rgb[:, args.given_frames:args.given_frames +
                     args.horizon]
        for i in range(B):
            rec_img = Perception_model.create_img(
                inital_feature_map[i].repeat(args.horizon, 1, 1, 1),
                rec_states[i, -args.horizon:, :, :2], rec_states[i, -args.horizon:, :, 2:])  # B H K 2
            rec_rgb_list.append(rec_img)
            # H C H W
        rec_rgb = torch.stack(rec_rgb_list)  # B T C H W
        return loss, loss_pos, loss_vel, loss_coef, loss_exist, rec_rgb, gt_rgb
    else:
        return loss, loss_pos, loss_vel, loss_coef, loss_exist


def evaludate_per_frame(Perception_model, dynamics_model, dataloader, args, des='Test'):
    loss_fn = lpips.LPIPS(net='alex').to(device)  # alex/vgg
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
        for i, data in enumerate(tqdm(dataloader, desc=des)):
            # x, y = data
            # rgb = torch.cat([x, y], dim=1).to(device)
            rgb = data['rgb_ab'].to(device)
            rgb = rgb[:, 0:args.given_frames + args.horizon]
            # T C H W
            loss, loss_pos, loss_vel, loss_coef, loss_exist, rec_seqs, gt_seqs = train_one_batch(
                rgb, Perception_model, dynamics_model, args=args, p=0, generate_img=True)  # B horizon C H W
            rec_rgb_list.append(rec_seqs)
            gt_rgb_list.append(gt_seqs)
            list_loss.append(loss.item())
            list_loss_coef.append(loss_coef.item())
            list_loss_vel.append(loss_vel.item())
            list_loss_position.append(loss_pos.item())
            list_loss_exist.append(loss_exist.item())
        rec_rgb = torch.cat(rec_rgb_list, dim=0)
        gt_rgb = torch.cat(gt_rgb_list, dim=0)
        print('rec shape:', rec_rgb.shape)
        for t in tqdm(range(rec_rgb.shape[1]), desc='Per_frame'):  # 在预测方向逐帧计算
            gt_seq = gt_rgb[:, t].cpu().detach().numpy().transpose(0, 2, 3, 1)
            pred_seq = rec_rgb[:, t].cpu(
            ).detach().numpy().transpose(0, 2, 3, 1)
            list_psnr.append(
                np.mean(
                    [peak_signal_noise_ratio(gt_seq[b], pred_seq[b] + 1e-4, data_range=2) for b in range(gt_rgb.shape[0])]))
            # B H W C
            # background_mask = remove_background(
            #     gt_seq, args.dataset)
            # list_local_psnr.append(
            #     np.mean(
            #         [peak_signal_noise_ratio(gt_seq[b]*background_mask[b], pred_seq[b]*background_mask[b] + 1e-4, data_range=2) for b in range(gt_rgb.shape[0])]))
            list_ssim.append(
                np.mean(
                    [structural_similarity(gt_seq[b], pred_seq[b], data_range=2, multichannel=True) for b in range(gt_rgb.shape[0])]))
            list_rgb_mse.append(MSE(gt_rgb[:, t], rec_rgb[:, t]).item())
            list_lpips.append(
                loss_fn(gt_rgb[:, t], rec_rgb[:, t]).mean().item())
        log_metric = {'PSNR': list_psnr, 'Local_PSNR': list_local_psnr,
                      'SSIM': list_ssim, 'LPIPS': list_lpips, 'MSE': list_rgb_mse}
        metric_result = pd.DataFrame(log_metric)
        log_dir = os.path.dirname(args.weight_path)
        csv_path = os.path.join(log_dir, 'metric_result.csv')
        metric_result.to_csv(csv_path, index=None, encoding='utf8')
        out = [np.mean(list_loss), np.mean(list_loss_position), np.mean(list_loss_vel), np.mean(list_loss_coef), np.mean(
            list_loss_exist), np.mean(list_rgb_mse)*1000, np.mean(list_psnr), np.mean(list_local_psnr), np.mean(list_ssim), np.mean(list_lpips)]
    return out


def test(Perception_model, dynamics_model, dataloader, args, des='Test', viz=True):

    loss_fn = lpips.LPIPS(net='alex').to(device)  # alex/vgg
    list_loss = []
    list_loss_coef = []
    list_loss_vel = []
    list_loss_position = []
    list_loss_exist = []

    mse_pixel_loss = []
    PSNR_list = []
    SSIM_list = []
    LPIPS_list = []
    Local_PSNR_list = []

    dynamics_model.eval()
    MSE = nn.MSELoss()

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc=des)):
            # x, y = data
            # rgb = torch.cat([x, y], dim=1).to(device)
            rgb = data['rgb_ab'].to(device)
            rgb = rgb[:, 0:args.given_frames + args.horizon]
            loss, loss_pos, loss_vel, loss_coef, loss_exist, rec_rgb, gt_rgb = train_one_batch(
                rgb, Perception_model, dynamics_model, args=args, p=0, generate_img=True)

            B, T, C, H, W = rec_rgb.shape
            list_loss.append(loss.item())
            list_loss_coef.append(loss_coef.item())
            list_loss_vel.append(loss_vel.item())
            list_loss_position.append(loss_pos.item())
            list_loss_exist.append(loss_exist.item())

            mse_pixel = MSE(gt_rgb, rec_rgb) * 1000
            mse_pixel_loss.append(mse_pixel.item())
            #print('rec img shape = ', rec_rgb.shape)
            LPIPS_list.append(loss_fn(gt_rgb.reshape(-1, C, H, W),
                              rec_rgb.reshape(-1, C, H, W)).mean().item())
            rec_img = rec_rgb.reshape(-1, C, H,
                                      W).cpu().detach().numpy().transpose(0, 2, 3, 1)
            gt_img = gt_rgb.reshape(-1, C, H,
                                    W).cpu().detach().numpy().transpose(0, 2, 3, 1)
            PSNR_list.append(np.mean([peak_signal_noise_ratio(
                gt_img[b], rec_img[b], data_range=2) for b in range(rec_rgb.shape[0])]))  # 一个batch 的平均PSNR
            SSIM_list.append(np.mean([structural_similarity(
                gt_img[b], rec_img[b], data_range=2, multichannel=True) for b in range(rec_rgb.shape[0])]))

            # background_mask = remove_background(gt_img, args.dataset)
            # Local_PSNR_list.append(
            #     np.mean(
            #         [peak_signal_noise_ratio(gt_img[b]*background_mask[b], rec_img[b]*background_mask[b] + 1e-4, data_range=2) for b in range(rec_rgb.shape[0])]))
            if viz:
                log_dir = os.path.dirname(args.weight_path)
                log_dir_vis = os.path.join(log_dir, 'vis_'+str(args.horizon))
                if not os.path.exists(log_dir_vis):
                    os.makedirs(log_dir_vis)
                video_idx_list = data['ex']
                # B T C H W
                if (i+1) % 1 == 0:
                    # print(torch.max(rgb))
                    # print(torch.max(img_gen))

                    visualize(gt_rgb.cpu().detach(),
                              rec_rgb.cpu().detach(), log_dir_vis, batch_idx='video_'+str(video_idx_list[-1]))
    return np.mean(list_loss), np.mean(list_loss_position), np.mean(list_loss_vel), np.mean(list_loss_coef), np.mean(list_loss_exist), np.mean(mse_pixel_loss), np.mean(PSNR_list), np.mean(Local_PSNR_list), np.mean(SSIM_list), np.mean(LPIPS_list)


def validate(Perception_model, dynamics_model, dataloader, args, des='Validation'):

    list_loss = []
    list_loss_coef = []
    list_loss_vel = []
    list_loss_position = []
    list_loss_exist = []

    dynamics_model.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc=des)):
            # x, y = data
            # rgb = torch.cat([x, y], dim=1).to(device)
            rgb = data['rgb_ab'].to(device)
            rgb = rgb[:, 0:args.given_frames + args.horizon]

            loss, loss_pos, loss_vel, loss_coef, loss_exist = train_one_batch(
                rgb, Perception_model, dynamics_model, args=args, p=0)  # 验证集不考虑折旧因子,因为最后就是1yep
            list_loss.append(loss.item())
            list_loss_coef.append(loss_coef.item())
            list_loss_vel.append(loss_vel.item())
            list_loss_position.append(loss_pos.item())
            list_loss_exist.append(loss_exist.item())
    return np.mean(list_loss), np.mean(list_loss_position), np.mean(list_loss_vel), np.mean(list_loss_coef), np.mean(list_loss_exist)


def train(Perception_model, dynamics_model, train_dataloader, val_dataloader, args):

    now_day = datetime.now().date()
    now_time = datetime.now().time()
    log_dir = args.output_dir or os.path.join(
        'output', args.dataset, 'logs_for_'+args.name,
        str(now_day), str(now_time)[0:5])
    os.makedirs(log_dir, exist_ok=True)
    print('log_dir:', log_dir)
    json_path = os.path.join(log_dir, 'log.json')
    with open(json_path, 'w') as f:
        json.dump(vars(args), f, indent=2)  # 一行，,indent=2：多行

    writer = SummaryWriter(log_dir)
    optim = torch.optim.Adam(dynamics_model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=5, verbose=True,
                                                       min_lr=1e-8)  # 验证 and 训练损失 默认是10
    #sched = torch.optim.lr_scheduler.StepLR(optim, gamma=0.5, step_size=10)
    best_val_loss = math.inf
    best_train_loss = math.inf
    best_val_position_loss = math.inf
    best_train_epoch = 0
    best_val_epoch = 0
    best_val_position_epoch = 0
    train_epochsize = len(train_dataloader)

    arg_str_list = ['{}={}'.format(k, v) for k, v in vars(
        args).items()]  # 读取namespace的字典 key-value
    # print('------------------------  argument namespace ----------------------')
    arg_str = '__'.join(arg_str_list)
    writer.add_text('hparams', arg_str)  # 记录超参数设计，比较容易对比

    for epoch in range(args.epoch):

        train_loss = 0.
        train_loss_position = 0.
        train_loss_vel = 0.
        train_loss_coef = 0.
        train_loss_exist = 0.

        dynamics_model.train()
        # Perception_model.eval()
        print("=== EPOCH ", epoch + 1, "/", args.epoch, " ===",
              'lr=', optim.param_groups[0]['lr'])
        p_gt = 0.  # 教师指导 or 自由训练
        if args.train_mode_verbose == 1:
            p_gt = 1.
        elif args.train_mode_verbose == 2:
            p_gt = linear_anneal(
                epoch, 1.0, 0.0, args.warm_epoch, args.decay_epoch+args.warm_epoch)
        if args.discount_factor:  # 随着迭代的epoch数增加
            df = linear_anneal(
                epoch, 1e-20, 1.0, 0, args.df_epoch)
        else:
            df = 1.0

        for i, data in enumerate(tqdm(train_dataloader, desc="Training")):

            # x, y = data
            # rgb = torch.cat([x, y], dim=1).to(device)
            rgb = data['rgb_ab'].to(device)
            rgb = rgb[:, 0:args.given_frames + args.horizon]

            loss, loss_pos, loss_vel, loss_coef, loss_exist = train_one_batch(
                rgb, Perception_model, dynamics_model, args=args, p=p_gt, batch_idx=i, epoch_idx=epoch, df=df)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss += loss.item()
            train_loss_coef += loss_coef.item()
            train_loss_vel += loss_vel.item()
            train_loss_position += loss_pos.item()
            train_loss_exist += loss_exist.item()

        train_loss /= train_epochsize
        train_loss_coef /= train_epochsize
        train_loss_vel /= train_epochsize
        train_loss_position /= train_epochsize
        train_loss_exist /= train_epochsize

        val_loss, val_loss_position, val_loss_vel,  val_loss_coef, val_loss_exist = validate(
            Perception_model, dynamics_model, val_dataloader, args=args)
        print('Epoch %d  ======> Train  loss = %.4f, loss_pos = %.4f, loss_dynamic = %.4f, loss_coef = %.4f, loss_exist = %.4f,' %
              (epoch+1, train_loss, train_loss_position, train_loss_vel, train_loss_coef, train_loss_exist))
        print('Epoch %d  ======> Valid  loss = %.4f, loss_pos = %.4f, loss_dynamic = %.4f, loss_coef = %.4f, loss_exist = %.4f,' %
              (epoch+1, val_loss, val_loss_position, val_loss_vel, val_loss_coef, val_loss_exist))

        writer.add_scalar('TRAIN/lr', optim.param_groups[0]['lr'], epoch+1)
        writer.add_scalar('TRAIN/p', p_gt, epoch+1)
        writer.add_scalar('TRAIN/df', df, epoch+1)
        writer.add_scalar('TRAIN/loss', train_loss, epoch+1)
        writer.add_scalar('TRAIN/loss_pos', train_loss_position, epoch+1)
        writer.add_scalar('TRAIN/loss_coef', train_loss_coef, epoch+1)
        writer.add_scalar('Valid/loss', val_loss, epoch+1)
        writer.add_scalar('Valid/loss_pos', val_loss_position, epoch+1)
        writer.add_scalar('Valid/loss_coef', val_loss_coef, epoch+1)

        sched.step(val_loss)

        torch.save(dynamics_model.state_dict(), os.path.join(
            log_dir, 'latest_model.pt'))

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_epoch = epoch+1
            torch.save(dynamics_model.state_dict(), os.path.join(
                log_dir, 'best_train_model.pt'))
            writer.add_scalar('TRAIN/best_loss', best_train_loss, epoch+1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch+1
            torch.save(dynamics_model.state_dict(), os.path.join(
                log_dir, 'best_val_model.pt'))
            writer.add_scalar('Valid/best_loss', best_val_loss, epoch+1)

        if val_loss_position < best_val_position_loss:
            best_val_position_loss = val_loss_position
            best_val_position_epoch = epoch + 1
            torch.save(dynamics_model.state_dict(), os.path.join(
                log_dir, 'best_val_pos_model.pt'))
            writer.add_scalar('Valid/best_loss_pos',
                              best_val_position_loss, epoch+1)

        print('Epoch %d  ======> Best Train loss = %f @ Epoch %d' %
              (epoch+1, best_train_loss, best_train_epoch))
        print('Epoch %d  ======> Best Val loss = %f @ Epoch %d' %
              (epoch+1, best_val_loss, best_val_epoch))
        print('Epoch %d  ======> Best Val loss position = %f @ Epoch %d' %
              (epoch+1, best_val_position_loss, best_val_position_epoch))
        print('log_dir:', log_dir)
    log_result = {'best_train_loss': float(best_train_loss), 'best_val_loss': float(best_val_loss),
                  'best_val_loss_pos': float(best_val_position_loss), 'initial_lr': float(args.lr), 'final_lr': float(optim.param_groups[0]['lr'])}
    update_json(os.path.join(
        log_dir, 'log.json'), log_result)
    print('log_dir:', log_dir)
    return log_dir


if args.epoch != 0:
    print('=======Train===========')
    if args.weight_path != '':
        pt_path = os.path.join(args.weight_path)
        print('Loading pre_training weight file from %s' % pt_path)
        dynamics_model.load_state_dict(
            torch.load(pt_path, map_location=device))
    log_dir = train(Perception_model, dynamics_model,
                    train_dataloader, val_dataloader, args)

    # 默认使用7个参数均匀的最佳权重，这样在test dataset的泛化性会好一点
    pt_path = os.path.join(log_dir, 'best_val_model.pt')
    #pt_path = os.path.join(log_dir, 'best_val_model.pt')
    dynamics_model.load_state_dict(torch.load(pt_path, map_location=device))
    test_loss, test_loss_pos, test_loss_vel, test_loss_coef, test_loss_exist, test_mse, test_psnr, test_local_psnr, test_ssim, test_lpips = test(
        Perception_model, dynamics_model, test_dataloader, args, des='Test', viz=False)
    para_num = count_parameters(dynamics_model)
    print('parameter = ', para_num)

    print('pose vector prediction error:',
          (test_loss_pos*2+test_loss_coef*4+test_loss_exist)/7)
    test_pose_error = (test_loss_pos*2+test_loss_coef*4+test_loss_exist)/7
    new_data = {'Perception_model_weight_path': Perception_weight_path, 'Dynamics_model_weight_path': pt_path, 'test_loss': float(test_loss), 'test_loss_pos': float(test_loss_pos), 'test_loss_coef': float(test_loss_coef), 'test_loss_exist': float(test_loss_exist), 'test_pixel_mse': float(test_mse), 'test_psnr': float(
        test_psnr), 'test_local_psnr': float(
        test_local_psnr), 'test_ssim': float(test_ssim), 'test_lpips': float(test_lpips), 'test_pose_error': float(test_pose_error), 'num_para': para_num}
    print(new_data)
    update_json(os.path.join(os.path.dirname(
        pt_path), 'log.json'), new_data)

else:
    print('=======test===========')
    pt_path = os.path.join(args.weight_path)
    print('Loading pre_training weight file from %s' % pt_path)
    dynamics_model.load_state_dict(torch.load(pt_path, map_location=device))

    val_loss, val_loss_position, val_loss_vel,  val_loss_coef, val_loss_exist = validate(
        Perception_model, dynamics_model, val_dataloader, args=args)
    print('Valid  loss = %.4f, loss_pos = %.4f, loss_dynamic = %.4f, loss_coef = %.4f, loss_exist = %.4f,' %
          (val_loss, val_loss_position, val_loss_vel, val_loss_coef, val_loss_exist))

    # test_loss, test_loss_pos, test_loss_vel, test_loss_coef, test_loss_exist, test_mse, test_psnr, test_local_psnr, test_ssim, test_lpips = test(
    #     Perception_model, dynamics_model, test_dataloader, args, des='Test', viz=False)
    # new_data = {'Perception_model_weight_path': Perception_weight_path, 'Dynamics_model_weight_path': pt_path, 'test_loss': float(test_loss), 'test_loss_pos': float(test_loss_pos), 'test_loss_coef': float(test_loss_coef), 'test_loss_exsit': float(test_loss_exist), 'test_pixel_mse': float(test_mse), 'test_psnr': float(
    #     test_psnr), 'test_local_psnr': float(
    #     test_local_psnr), 'test_ssim': float(test_ssim), 'test_lpips': float(test_lpips), 'num_para': para_num}

    # print(new_data)
    # #update_json(os.path.join(os.path.dirname(pt_path), 'log.json'), new_data)

    # print('pose vector prediction error:',
    #       (test_loss_pos*2+test_loss_coef*4+test_loss_exist)/7)

    # eval_dataloader = DataLoader(
    #     datasets[args.dataset](mode='test', resolution=112,
    #                            sampling_mode='full', env=args.dataset),
    #     batch_size=188, num_workers=1, pin_memory=True, drop_last=True)
    # out = evaludate_per_frame(
    #     Perception_model, dynamics_model, test_dataloader, args, des='Evaluate')
    # print('Test  loss = %.4f, loss_pos = %.4f, loss_dynamic = %.4f, loss_coef = %.4f, loss_exist = %.4f'
    #       % (out[0], out[1], out[2], out[3], out[4]))
    # print('Test  pixel_mse = %.4f, PSNR = %.4f, L-PSNR = %.4f, SSIM = %.4f, LPIPS = %.4f'
    #       % (out[5], out[6], out[7], out[8], out[9]))
