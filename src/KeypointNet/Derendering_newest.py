"""
Author:xuhuil
date:2023/1/16
细节： 考虑了参考帧的高斯图, 归一化由坐标和旋转参数获得高斯图, 使得最近的网格值=1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
import warnings
# Debug leftover: enabling anomaly detection at import time made every autograd
# node capture a full Python stack trace (~40% of train time on CPFS).
# torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore")


def create_affine_transform_matrix(z_theta, z_scale, z_where, device, inv=False):
    """
    z_theta: N # 输入的,bottleneck 预测的 系数
    z_scale: N x 2 or N   # 1
    z_where: N x 2 # 已经知道了  z_where = 1 就行
    return: N x 3 x 3
    """
    N = z_theta.size(0)
    R = z_theta.new_zeros(N, 3, 3).to(device)
    S = z_scale.new_zeros(N, 3, 3).to(device)
    T = z_where.new_zeros(N, 3, 3).to(device)

    R[:, 2, 2] = 1.
    S[:, 2, 2] = 1.
    T[:, 0, 0] = 1.
    T[:, 1, 1] = 1.
    T[:, 2, 2] = 1.

    if inv:
        z_theta = -z_theta
        z_scale = z_scale.reciprocal()
        z_where = -z_where

    # 旋转矩阵
    cos_z_theta = z_theta.cos()
    sin_z_theta = z_theta.sin()
    R[:, 0, 0] = cos_z_theta
    R[:, 0, 1] = -sin_z_theta
    R[:, 1, 0] = sin_z_theta
    R[:, 1, 1] = cos_z_theta

    if len(z_scale.size()) == 1:
        S[:, 0, 0] = z_scale
        S[:, 1, 1] = z_scale
    else:
        S[:, 0, 0] = z_scale[:, 0]
        S[:, 1, 1] = z_scale[:, 1]

    T[:, 0, 2] = z_where[:, 0]  # 平移
    T[:, 1, 2] = z_where[:, 1]

    if inv:
        return S.bmm(R).bmm(T)
    else:
        # 缩放 -> 旋转 -> 平移
        # return S.bmm(T).bmm(R)
        return T.bmm(R).bmm(S)  # batchsize 乘法 第一维N 后面两维相乘


def spatial_transform(x, out_size, z_theta, z_scale, z_where, inv=False):
    """
    x: N x C x H_in x W_in           B K H W  这里的N等于B
    out_size: (N, C, H_out, W_out)   B K H W
    z_theta: N                       K
    z_scale: N x 2 or N              K
    z_where: N x 2                   K*2
    return: N x C x H_out x W_out   B K H W
    """
    # input x:[N*4 3 128 128]   out_size=(N*4 3 64 64)
    # A: K x 3 x 3
    A = create_affine_transform_matrix(
        z_theta, z_scale, z_where, inv=inv, device=x.device)  # 仿射转换矩阵

    return F.grid_sample(x, F.affine_grid(A[:, :-1], torch.Size(out_size)))


class Block(nn.Module):
    """
    Functions: conv+BN+Relu
    """

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return torch.relu(x)


class Encoder(nn.Module):
    """
    Functions:
    Input: shape = [2*B, 3, 112, 112] cat[source, target]
    Return:
        x_feat: shape = [2*B, 16, 28, 28], feature map encoding static feature
        pose:  shape = [2*B, K, 28, 28] 2D coordinate of K keypoints
        x_coef: shape = [2*B, K, 28, 28]
    """

    def __init__(self, k, img_channel):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            Block(img_channel, 32, kernel_size=(7, 7), stride=1,
                  padding=3),  # H,W 保持不变  通道数根据 数据而变化 3->1
            Block(32, 32, kernel_size=(3, 3), stride=1),  # H=112 padding=1
            Block(32, 64, kernel_size=(3, 3), stride=2),  # H=56
            Block(64, 64, kernel_size=(3, 3), stride=1),  # H=56
            Block(64, 128, kernel_size=(3, 3), stride=2)  # H=28
        )
        self.net_feat = Block(128, 16, kernel_size=(3, 3), stride=1)  # 28 改

        self.net_pose = Block(128, 128, kernel_size=(3, 3), stride=1)  # 28

        self.net_coef = Block(128, k, kernel_size=(3, 3), stride=1)  # 28

        self.regressor = nn.Conv2d(
            128, k, kernel_size=(1, 1))  # 28 Gaussian mapping??

    def forward(self, x):
        x = self.net(x)  # [B, 128, 28, 28]

        x_feat = self.net_feat(x)  # [B, 16, 28, 28]
        x_pose = self.net_pose(x)  # [B, 128, 28, 28]
        x_coef = self.net_coef(x)  # [B, k, 28, 28]

        pose = F.softplus(self.regressor(x_pose))  # [B, K, 28, 28]
        # softpuls log(1+e^x) 保证是正数
        # condensation_loss 实际上要求 0-1的距离!(sigmoid) 一次尝试而已 spatial-softmax 会不一样的嘤嘤嘤
        # pose = F.sigmoid(self.regressor(x_pose))
        return x_feat, pose, x_coef


class Decoder(nn.Module):
    def __init__(self, k, n_param, img_channel, is_transport=False):
        super(Decoder, self).__init__()
        if n_param == 0:  # 考虑了参考帧的高斯图
            i = k * 2
        else:
            i = k * n_param * 2
        if is_transport:
            i = 0
        # i = 1  # 12/15 添加（尝试）
        self.net = nn.Sequential(
            Block(16 + i, 128, kernel_size=(3, 3), stride=1),  # 6
            Block(128, 128, kernel_size=(3, 3), stride=1),
            Block(128, 64, kernel_size=(3, 3), stride=1),  # 5
            nn.UpsamplingBilinear2d(scale_factor=2),  # 双线性上采样 56*56
            Block(64, 64, kernel_size=(3, 3), stride=1),  # 4
            Block(64, 32, kernel_size=(3, 3), stride=1),  # 3
            nn.UpsamplingBilinear2d(scale_factor=2, ),  # 双线性上采样 112*112
            Block(32, 32, kernel_size=(3, 3), stride=1),  # 2
            Block(32, 32, kernel_size=(7, 7), stride=1, padding=3),  # 1
            nn.Conv2d(32, img_channel, 1)  # 通道数 3->1
        )

    def forward(self, x):
        return torch.tanh(self.net(x))


def spatial_softmax(features):
    # [B,K]+(-1.)=[B, K, -1] --> [B,K,28*28]
    features_reshape = features.reshape(features.shape[:-2] + (-1,))
    output = F.softmax(features_reshape, dim=-1)
    output = output.reshape(features.shape)
    return output


class Derendering_newest(nn.Module):
    def __init__(self, n_keypoints=4, n_coefficients=1, std=0.15, device=torch.device("cpu"), img_width=112, img_channel=3, filter_size=5, is_depth=False):
        super(Derendering_newest, self).__init__()

        self.encoder = Encoder(k=n_keypoints, img_channel=img_channel)
        self.decoder = Decoder(
            k=n_keypoints, n_param=n_coefficients, img_channel=img_channel)

        self.std = std
        self.filter_size = filter_size
        self.k = n_keypoints  # pose 2d position
        self.n_coefficients = n_coefficients  # rotate exist?
        self.extractor = nn.Sequential(
            nn.Linear(img_width * img_width // 16, 2048), nn.ReLU(),  # 图像尺寸而变化
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, n_coefficients + 1)
        )
        self.is_depth = is_depth

    def forward(self, source, target):
        out = {}
        B, C, H, W = source.shape
        # source target
        features, heatmap, coef = self.encoder(
            torch.cat([source, target], dim=0))

        source_features = features[:B]
        keypoints_locations = self.get_keypoint_location(
            heatmap)
        parameters = self.extractor(
            torch.flatten(coef, start_dim=2))
        parameters = torch.sigmoid(parameters)

        target = self.create_img(
            source_features, keypoints_locations, parameters)

        out['source_features'] = source_features
        out["target_keypoints"] = keypoints_locations[B:]
        out["target_parameter"] = parameters[B:]
        out['target_heatmap'] = heatmap[B:]
        out['target'] = target

        return out

    def get_keypoint_location(self, keypoints_features):
        """
        Functions : get keypoint location from K heatmaps
        paper: Unsupervised Learning of Object Landmarks through Conditional Image Generation
        Input: K heatmaps shape = [B K 28 28]
        Return: 2D coordinate for K keypoints  shape = [B K 2]
        """
        # renormalised to a probability distribution via (spatial) Softmax
        # B K 每个heatmap上所有值之和=1 B K H W
        normalized_keypoints = spatial_softmax(keypoints_features)
        S_row = normalized_keypoints.sum(-1)  # N, K, H
        S_col = normalized_keypoints.sum(-2)  # N, K, W

        u_row = S_row.mul(torch.linspace(-1, 1, S_row.size(-1),
                          device=normalized_keypoints.device)).sum(-1)  # ？有点奇怪
        u_col = S_col.mul(torch.linspace(-1, 1, S_col.size(-1),
                          device=normalized_keypoints.device)).sum(-1)
        return torch.stack((u_row, u_col), -1)

    def create_img(self, src_features, keypoints_locations, parameters):

        B = src_features.shape[0]  # B K 2

        src_keypoints_locations = keypoints_locations[:B]
        src_parameters = parameters[:B]
        tgt_keypoints_locations = keypoints_locations[B:]
        tgt_parameters = parameters[B:]

        src_heatmap = self.get_keypoints_map(
            src_features, src_keypoints_locations, src_parameters)
        tgt_heatmap = self.get_keypoints_map(
            src_features, tgt_keypoints_locations, tgt_parameters)

        inpt = torch.cat([src_features, src_heatmap,
                         tgt_heatmap], dim=1)  # 汇合特征图 和 热力图

        target_images = self.decoder(inpt)

        return target_images

    def get_keypoints_map(self, features, keypoints_locations, parameters):

        B, C, H, W = features.shape  # B K 2
        y = torch.linspace(-1.0, 1.0, H, device=keypoints_locations.device)
        x = torch.linspace(-1.0, 1.0, W, device=keypoints_locations.device)
        y = y.reshape(1, 1, H, 1)
        x = x.reshape(1, 1, 1, W)

        mu_x, mu_y = keypoints_locations[...,
                                         0:1], keypoints_locations[..., 1:2]
        mu_x, mu_y = mu_x.unsqueeze(-1), mu_y.unsqueeze(-1)

        if self.n_coefficients != 0:
            # 先卷积 再平移 + 旋转
            center_x, center_y = torch.zeros_like(mu_x), torch.zeros_like(mu_y)
            dist = ((y - center_x) ** 2 + (x - center_y) ** 2) / self.std ** 2
            heatmaps = torch.exp(-dist).unsqueeze(2).view(1,
                                                          B*self.k, H, W)  # 以(0,0)为中心的热力图
            filter = torch.eye(self.filter_size, self.filter_size).unsqueeze(
                0).repeat(B*self.k, 1, 1).unsqueeze(1).to(features.device)
            conv_heatmap = F.conv2d(heatmaps, filter, padding=self.filter_size//2,
                                    groups=B * self.k).view(-1, 1, H, W)
            batchsize = conv_heatmap.shape[0]  # B*K
            out_size = (batchsize, 1, H, W)
            z_scale = torch.ones(batchsize)
            z_where = torch.cat([-mu_y, -mu_x], dim=-1).view(batchsize, -1)
            z_theta = parameters[..., 0:1].view(-1) * math.pi / 2  # 单纯四个角度
            # 改成pi会怎么样
            z_theta = parameters[..., 0:1].view(-1) * math.pi 
            rotate_heatmap = spatial_transform(
                conv_heatmap, out_size, z_theta, z_scale, torch.zeros_like(z_where))

            shift_heatmap = spatial_transform(
                rotate_heatmap, out_size, torch.zeros_like(z_theta), z_scale, z_where)
        else:

            dist = ((y - mu_x) ** 2 + (x - mu_y) ** 2) / self.std ** 2
            heatmaps = torch.exp(-dist)  # 以(0,0)为中心的热力图
            shift_heatmap = heatmaps

        # 最近的网络为1！ over
        distort_heatmap = shift_heatmap.view(B, self.k,  H, W)

        # heatmap_max = torch.max(distort_heatmap.view(B, self.k, -1), dim=-1)[0]
        # # print(heatmap_max.shape) /=的操作会导致梯度后传报错(一定情况下) --> x /= y 修改为 x = x/;y
        # # https://blog.csdn.net/qq_35056292/article/details/116695219
        # distort_heatmap = distort_heatmap / \
        #     heatmap_max.reshape(B, self.k, 1, 1)

        z_exist = parameters[..., -1].view(B, self.k, 1, 1)  # B K 1 1
        if not self.is_depth:  # 是否考虑深度信息(透明度等)
            gussian_map = distort_heatmap * z_exist  # B K H W 2022-12-17-14:17 大小 成正相关
            # B K H W 2022-12-17-14:42 z的坐标(深度图) 成反相关
            #gussian_map = distort_heatmap * torch.exp(-z_exist)
        else:
            z_depth = torch.exp(-z_exist)
            alpha_map = []
            alpha_map.append(torch.ones(
                [B, H, W], device=features.device))  # B 28 28
            for i in range(0, self.k-1):
                alpha_map.append(alpha_map[-1]-distort_heatmap[
                    :, i]*z_depth[:, i])
            alpha_map = torch.stack(alpha_map, dim=1)  # B K  H W
            gussian_map = alpha_map * z_depth * \
                distort_heatmap  # 看单个目标的轨迹，直接将z_depth 此处 设为
        return gussian_map

    def get_features_only(self, x):
        B, C, H, W = x.shape
        # x = self.net(x)
        # x_feat = self.net_feat(x)
        x = self.encoder.net(x)
        x_feat = self.encoder.net_feat(x)
        return x_feat

    def get_keypoints_only(self, x):
        x = self.encoder.net(x)
        x_pose = self.encoder.net_pose(x)
        x_coef = self.encoder.net_coef(x)
        keypoints = F.softplus(
            self.encoder.regressor(x_pose))  # [B, K, 28, 28]
        target_parameter = self.extractor(torch.flatten(x_coef, start_dim=2))
        target_parameter = torch.sigmoid(target_parameter)  # [B, K, C]
        target_keypoints_location = self.get_keypoint_location(
            keypoints)  # [B, K, 2]
        return target_keypoints_location, target_parameter


# B = 8
# K = 3
# C = 1
# model = Derendering_newest(n_keypoints=K,
#                            n_coefficients=C,
#                            device='cpu', img_width=112, img_channel=3)
# source_img = torch.randn(B,  3, 112, 112)
# target_img = torch.randn(B, 3, 112, 112)
# out = model(source_img, target_img)

# MSE = nn.MSELoss()
# print(MSE(target_img, out['target']))

# out = model(target_img, source_img)
# print(MSE(source_img, out['target']))
