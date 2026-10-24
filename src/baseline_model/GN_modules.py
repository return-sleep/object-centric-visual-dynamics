#!/usr/bin/env python
# coding: utf-8
"""
author: Huilin XU
date: 2022/12/11
description : various GN blocks for node vector/ feature map
"""
from src.KeypointNet.Derendering import get_dilatation_filters, gaussian_map
import numpy as np
import torch
import torch.nn as nn
import math
from src.KeypointNet.Derendering_with_rotate import (
    spatial_transform,
    create_affine_transform_matrix,
)
import torch.nn.functional as F


def crop_feature_map(feature_map, pos, crop_size=5):
    """
    Input:
        feature_map: B, C, H, W
        pos : B K 2
    Output:
        crop_feaure: B K C crop_size, crop_size

    """
    B, C, H, W = feature_map.shape
    K = pos.shape[1]
    mu_x = pos[..., 0]
    mu_y = pos[..., 1]
    pos_x, pos_y = np.array(torch.round((mu_y.cpu().detach() + 1) / 2 * H)), np.array(
        torch.round((mu_x.cpu().detach() + 1) / 2 * W)
    )
    crop_img = torch.zeros([B, K, C, crop_size, crop_size]).to(feature_map.device)
    for i in range(B):
        for j in range(K):
            x0 = int(-crop_size // 2 + pos_x[i, j])
            if x0 < H - crop_size - 1:
                x0 = np.max([x0, 0])
                x1 = x0 + crop_size
            else:
                x1 = int(crop_size // 2 + pos_x[i, j])
                x1 = np.min([x1, H - 1])
                x0 = x1 - crop_size

            y0 = int(-crop_size // 2 + pos_y[i, j])
            if y0 < W - crop_size - 1:
                y0 = np.max([y0, 0])
                y1 = y0 + crop_size
            else:
                y1 = int(crop_size // 2 + pos_y[i, j])
                y1 = np.min([y1, W - 1])
                y0 = y1 - crop_size

            crop_img[i, j] = feature_map[i, :, x0:x1, y0:y1]
    return crop_img


def edges_from_adjacency_matrix(A, O, dim=1):
    """
    Input:
        A: (B num_objs num_objs) adjacency_matrix of num_objs nodes
        O: (B num_objs dim_objs)list of object states or  (B num_objs dim_objs H W)
        dim: node is a vector (dim_objs) or a feaure map (dim_objs H W)
    Output:
        #g_idx:  (B num_objs*num_objs 3) triplet (idx_sender, idx_receiver, attr)
        E: edges (B num_objs*num_objs 2*dim_objs) / (B, num_objs*num_objs, 2*dim_objs, H,W)
    """
    B, num_objs, _ = A.shape
    g_idx = torch.zeros([B, num_objs * num_objs, 3], dtype=torch.int64, device=A.device)
    for i in range(num_objs):
        for j in range(num_objs):
            g_idx[:, i * num_objs + j, 0] = i
            g_idx[:, i * num_objs + j, 1] = j
            g_idx[:, i * num_objs + j, 2] = A[:, i, j]  # 是否存在或者是存在后代表的线的长度~~ 边的属�?
            # if i == j : #同一时刻没有self-caution
            #     g_idx[:, i*num_objs+j, 2] = 0
    if dim == 1:
        dim_objs = O.shape[2]
        i1 = g_idx[..., [0]].repeat(1, 1, dim_objs)  # B n*n 1
        i2 = g_idx[..., [1]].repeat(1, 1, dim_objs)
        y1 = torch.gather(O, 1, i1)
        y2 = torch.gather(O, 1, i2)
        r = torch.cat([y1, y2], dim=-1)  # 如果边有属性的话应该额外级�?,发现因果关系时，该边的属�? �? 可学习的
        r = r * g_idx[..., [2]]  # 该条边是否存�?
        edges_exist = g_idx[:, :, [2]]
        # B num_objs*num_objs 2*dim_objects
        E = r.view(B, num_objs * num_objs, -1)
    else:
        dim_objs, psz = O.shape[2], O.shape[3]
        i1 = g_idx[..., [0], None, None].repeat(1, 1, dim_objs, psz, psz)
        y1 = torch.gather(O, 1, i1)
        i2 = g_idx[..., [1], None, None].repeat(1, 1, dim_objs, psz, psz)
        y2 = torch.gather(O, 1, i2)  # b O*(O-1)
        r = torch.cat([y1, y2], dim=2)  # b O*(O-1) 2*dim psz psz
        r = r * g_idx[:, :, [2], None, None]
        edges_exist = g_idx[:, :, [2], None, None]
        E = r.view(B, num_objs * num_objs, -1, psz, psz)
    return E, edges_exist


def positional_encoding(tensor, num_encoding_functions=6, include_input=True):
    """Apply positional encoding to the input.

    Args:
    tensor (torch.Tensor): Input tensor to be positionally encoded. B K S
    encoding_size (optional, int): Number of encoding functions used to compute
        a positional encoding (default: 6).
    include_input (optional, bool): Whether or not to include the input in the
        positional encoding (default: True)

    Returns:
    dim = (2*num_encoding_functions +1) * tensor.dim
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    # Now, encode the input using a set of high-frequency functions and append the
    # resulting values to the encoding.
    for i in range(num_encoding_functions):
        for func in [torch.sin, torch.cos]:
            encoding.append(func(2.0**i * tensor))
    return torch.cat(encoding, dim=-1)


def get_keypoints_map(args, features, keypoints_locations, parameters, is_depth=False):
    """
    输入特征�? �? 目标对应的状态变�?,生成对应的高斯图
    """
    B, C, H, W = features.shape  # B K 2
    k = keypoints_locations.shape[1]
    _key = (H, W, str(keypoints_locations.device), keypoints_locations.dtype)
    _grid = getattr(get_keypoints_map, "_grid", None)
    if _grid is None or get_keypoints_map._grid_key != _key:
        y = torch.linspace(
            -1.0, 1.0, H, device=keypoints_locations.device, dtype=keypoints_locations.dtype
        ).reshape(1, 1, H, 1)
        x = torch.linspace(
            -1.0, 1.0, W, device=keypoints_locations.device, dtype=keypoints_locations.dtype
        ).reshape(1, 1, 1, W)
        get_keypoints_map._grid = (y, x)
        get_keypoints_map._grid_key = _key
    else:
        y, x = get_keypoints_map._grid

    mu_x, mu_y = keypoints_locations[..., 0:1], keypoints_locations[..., 1:2]
    mu_x, mu_y = mu_x.unsqueeze(-1), mu_y.unsqueeze(-1)

    if parameters.shape[-1] > 1:
        # 先卷�? 再平�? + 旋转
        center_x, center_y = torch.zeros_like(mu_x), torch.zeros_like(mu_y)
        dist = ((y - center_x) ** 2 + (x - center_y) ** 2) / args.std**2
        heatmaps = torch.exp(-dist).unsqueeze(2).view(1, B * k, H, W)  # �?(0,0)为中心的热力�?
        filter = (
            torch.eye(args.filter_size, args.filter_size)
            .unsqueeze(0)
            .repeat(B * k, 1, 1)
            .unsqueeze(1)
            .to(features.device)
        )
        conv_heatmap = F.conv2d(
            heatmaps, filter, padding=args.filter_size // 2, groups=B * k
        ).view(-1, 1, H, W)
        batchsize = conv_heatmap.shape[0]  # B*K
        out_size = (batchsize, 1, H, W)
        z_scale = torch.ones(batchsize)
        z_where = torch.cat([-mu_y, -mu_x], dim=-1).view(batchsize, -1)
        z_theta = parameters[..., 0:1].view(-1) * math.pi / 2  # 单纯四个角度
        rotate_heatmap = spatial_transform(
            conv_heatmap, out_size, z_theta, z_scale, torch.zeros_like(z_where)
        )

        shift_heatmap = spatial_transform(
            rotate_heatmap, out_size, torch.zeros_like(z_theta), z_scale, z_where
        )
    else:
        dist = ((y - mu_x) ** 2 + (x - mu_y) ** 2) / args.std**2
        heatmaps = torch.exp(-dist)  # �?(0,0)为中心的热力�?
        shift_heatmap = heatmaps

    # 最近的网络�?1�? over
    distort_heatmap = shift_heatmap.view(B, k, H, W)
    # 目前不需要归一�?
    # heatmap_max = torch.max(distort_heatmap.view(B, k, -1), dim=-1)[0]
    # #print(heatmap_max.shape)
    # distort_heatmap = distort_heatmap /heatmap_max.reshape(B, k, 1, 1)
    if parameters.shape[-1] == 0:
        # z_exist = torch.zeros([B, k, 1, 1], device=keypoints_locations.device)
        z_exist = torch.ones([B, k, 1, 1], device=keypoints_locations.device)
    else:
        z_exist = parameters[..., -1].view(B, k, 1, 1)  # B K 1 1

    if not is_depth:  # 是否考虑深度信息(透明度等)
        gussian_map = distort_heatmap * z_exist  # B K H W 2022-12-17-14:17 大小 成正相关
        # B K H W 2022-12-17-14:42 z的坐�?(深度�?) 成反相关
        # gussian_map = distort_heatmap * torch.exp(-z_exist)
    else:
        z_depth = torch.exp(-z_exist)
        alpha_map = []
        alpha_map.append(torch.ones([B, H, W], device=features.device))  # B 28 28
        for i in range(0, k - 1):
            alpha_map.append(alpha_map[-1] - distort_heatmap[:, i] * z_depth[:, i])
        alpha_map = torch.stack(alpha_map, dim=1)  # B K  H W
        gussian_map = alpha_map * z_depth * distort_heatmap  # 看单个目标的轨迹，直接将z_depth 此处 设为
    return gussian_map


def get_filter_bank(n_keypoints, n_coefficients, device):
    bank = get_dilatation_filters(5, 5, n_coefficients)
    bank = bank.to(device).view(1, 1, n_coefficients, 5, 5)
    bank = bank.repeat(1, n_keypoints, 1, 1, 1)
    return bank


def get_keypoints_map_Filter(
    args, features, keypoints_locations, parameters, is_depth=False
):
    """
    reture: gaussian_map [B K*C H W]
    """
    B, C, H, W = features.shape
    target_keypoints = gaussian_map(keypoints_locations, W, H, std=args.std)
    # print('target_keypoints shape:',target_keypoints.shape)
    bank = get_filter_bank(
        args.n_keypoints, args.n_coefficients, device=features.device
    )
    k = args.n_keypoints
    n_coefficients = args.n_coefficients
    weighted_bank = bank * parameters[:, :, :-1].view(
        B, k, n_coefficients, 1, 1
    )  # [B K C=4 H W]
    weighted_bank = parameters[:, :, -1].view(B, k, 1, 1, 1) * weighted_bank
    weighted_bank = weighted_bank.view(
        B * k * n_coefficients, 1, 5, 5
    )  # [B*K*C 1 5 5] [output_channel, input_channel, H,W]
    # print('weighted_bank shape:',weighted_bank.shape)
    target_keypoints = target_keypoints.unsqueeze(2).repeat(1, 1, n_coefficients, 1, 1)
    target_keypoints = target_keypoints.view(1, B * k * n_coefficients, W, H)
    # # [1 B*K*C 28 28]
    # print('target_keypoints shape:',target_keypoints.shape)
    keypoints_map = F.conv2d(
        target_keypoints, weighted_bank, padding=2, groups=B * k * n_coefficients
    )  # [1 B*K*C 28 28]
    # print(keypoints_map.shape)
    # B K*C 28 28
    keypoints_map = keypoints_map.view(B, -1, W, H)
    return keypoints_map


class MLP(nn.Module):
    """
    Functions: multiple Linear layers
    Input: [*,in_size]
    Output: [*,out_size]
    n_layer: (in-h_size) n_layer * (h_size - h_size) (h_size-out_size)
    """

    def __init__(
        self,
        in_size,
        out_size,
        h_size=128,
        n_layers=2,
        activation=nn.ReLU,
        layernorm=False,
        batchnorm=False,
        n_channel=4,
    ):
        super(MLP, self).__init__()

        layer_list = []

        layer_list.append(nn.Linear(in_size, h_size))
        layer_list.append(activation())

        for _ in range(n_layers):
            layer_list.append(nn.Linear(h_size, h_size))
            layer_list.append(activation())
        # layer_list.append(nn.Dropout(0.5)) ##最大修改的
        layer_list.append(nn.Linear(h_size, out_size))

        if batchnorm:
            layer_list.append(nn.BatchNorm1d(n_channel))
        if layernorm:
            layer_list.append(nn.LayerNorm(out_size))  # LN来稳定训�? 每个MLP后面接一个LN

        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x)


class GCN(nn.Module):
    """
    Functions: contextualize node state by graph network
    Params:
        n_layer: n+2 linear layers in MLP
    Input: [B K S=in_size]
    Output: [B K out_size]
    """

    def __init__(
        self,
        in_size,
        out_size,
        n_layers=2,
        hidden_size=128,
        activation=nn.ReLU,
        layernorm=False,
        batchnorm=False,
    ):
        super(GCN, self).__init__()
        # in_size, out_size, hidden_size,: Ds Dp DE
        # relation-centric function : columns represent the interaction terms
        self.f = MLP(
            2 * in_size,
            hidden_size,
            hidden_size,
            n_layers,
            activation,
            batchnorm=batchnorm,
            layernorm=layernorm,
            n_channel=hidden_size,
        )
        # object-centric
        self.g = MLP(
            2 * hidden_size + in_size,
            out_size,
            hidden_size,
            n_layers,
            activation,
            batchnorm=batchnorm,
            layernorm=layernorm,
            n_channel=4,
        )

    def forward(self, x):
        B, K, S = x.shape

        x1 = x.unsqueeze(1).expand(B, K, K, S)  # B K K S receiver
        # B K K S sender interaction = K(K-1)
        x2 = x.unsqueeze(2).expand(B, K, K, S)

        x12 = torch.cat([x1, x2], dim=-1)  # B K K S*2(sender+receiver)
        interactions = self.f(x12.reshape(B * K * K, 2 * S)).view(
            B, K, K, -1
        )  # B K K hiden_size
        # Broadcast the KxK self-loop mask; no need to repeat over B / hidden.
        E = interactions * (
            1.0 - torch.eye(K, device=x.device, dtype=interactions.dtype)
        ).view(1, K, K, 1)  # 去掉self-causation

        E = E.sum(2)  # B K hidden_size  aggragation function: sum
        E_sum = E.sum(1, keepdim=True).expand(-1, K, -1)  # 每个节点对应的X(外部力或者重力加速度相同)
        # B K out_size linear layer: [anynumber H_in]
        out = self.g(torch.cat([x, E, E_sum], dim=-1))

        return out


class Interaction_Network_1D(nn.Module):
    """
    缺点�? 1D的GN
        没有考虑每条边的属�? 三元组中只考虑了sender和receiver
        不能实时地反应边的性质(是否存在、类别、弹簧长度等)
        全局变量 直接 �? 每个节点 级联 输入
    Input :  nodes representaion (B K D) & adjacency_matrix (B K K )
    Output: updated nodes (B K D)
    """

    def __init__(self, dim_objs, dim_out=None, hidden_size=128, n_layer=2):
        super(Interaction_Network_1D, self).__init__()
        if dim_out == None:
            dim_out = dim_objs
        self.dim_objs = dim_objs
        self_dynamics = [
            MLP(
                in_size=dim_objs,
                out_size=dim_objs,
                h_size=hidden_size,
                n_layers=n_layer,
            )
        ]
        rel_dynamics = [
            MLP(
                in_size=dim_objs * 2,
                out_size=dim_objs,
                h_size=hidden_size,
                n_layers=n_layer,
            )
        ]
        affector = [
            MLP(
                in_size=dim_objs,
                out_size=dim_objs,
                h_size=hidden_size,
                n_layers=n_layer,
            )
        ]
        aggregator = [
            MLP(
                in_size=dim_objs * 2,
                out_size=dim_out,
                h_size=hidden_size,
                n_layers=n_layer,
            )
        ]
        self.self_dynamics = nn.Sequential(*self_dynamics)
        self.rel_dynamics = nn.Sequential(*rel_dynamics)
        self.affector = nn.Sequential(*affector)
        self.aggregator = nn.Sequential(*aggregator)

    def forward(self, x, A=None):
        B, num_objs, dim_objs = x.shape  # 若考虑全局属性，就将全局特征与节点特征级�?
        if A == None:
            A = 1 - torch.eye(num_objs).unsqueeze(0).repeat(B, 1, 1).to(x.device)
        E, edges_exist = edges_from_adjacency_matrix(A, x)  # 若考虑每条边的属性可以级联然后再乘以存在矩阵

        self_effect = self.self_dynamics(x.reshape(-1, dim_objs)).reshape(
            B, num_objs, -1
        )
        rel_effect = self.rel_dynamics(E.reshape(-1, dim_objs * 2)).reshape(
            B, num_objs, num_objs, -1
        )
        rel_effect = rel_effect.sum(2)  # B, num_objs, dim_objs

        all_effect = self_effect + rel_effect
        # print(all_effect.view(-1, dim_objs).shape)
        effect = self.affector(all_effect.view(-1, dim_objs)).reshape(B, num_objs, -1)

        x_out = self.aggregator(
            torch.cat([x, effect], dim=2).view(-1, dim_objs * 2)
        ).reshape(B, num_objs, -1)
        return x_out


class Interaction_Network_3D(nn.Module):
    """
    缺点�? 1D的GN
        没有考虑每条边的属�? 三元组中只考虑了sender和receiver
        不能实时地反应边的性质(是否存在、类别、弹簧长度等)
        全局变量 直接 �? 每个节点 级联 输入
    Input :  nodes representaion (B K D H W) & adjacency_matrix (B K K)
    Output: updated nodes (B K D H W)
    """

    def __init__(self, dim_objs):
        super(Interaction_Network_3D, self).__init__()
        self.in_feat_dim = dim_objs
        self_dynamics = [
            nn.Conv2d(
                self.in_feat_dim, self.in_feat_dim, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
        ]
        rel_dynamics = [
            nn.Conv2d(
                self.in_feat_dim * 2,
                self.in_feat_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        ]
        aggregator = [
            nn.Conv2d(
                self.in_feat_dim * 2,
                self.in_feat_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        ]
        affector = [
            nn.Conv2d(
                self.in_feat_dim, self.in_feat_dim, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
        ]
        self.self_dynamics = nn.Sequential(*self_dynamics)
        self.rel_dynamics = nn.Sequential(*rel_dynamics)
        self.affector = nn.Sequential(*affector)
        self.aggregator = nn.Sequential(*aggregator)

    def forward(self, x, A=None):
        s = x
        batch_size, num_objs, dim, psz, psz = x.shape
        if A == None:
            A = 1 - torch.eye(num_objs).unsqueeze(0).repeat(batch_size, 1, 1).to(
                x.device
            )

        r, edges_exist = edges_from_adjacency_matrix(A, x, dim=3)
        r = self.rel_dynamics(r.reshape(-1, dim * 2, psz, psz))
        r = r.reshape(batch_size, num_objs, num_objs, dim, psz, psz)
        r = r.sum(dim=2)

        x = self.self_dynamics(x.reshape(-1, dim, psz, psz)).reshape(
            batch_size, num_objs, dim, psz, psz
        )

        pred = x + r
        a = self.affector(pred.reshape(-1, dim, psz, psz)).reshape(
            batch_size, num_objs, dim, psz, psz
        )

        a = torch.cat([a, s], 2)
        out = self.aggregator(a.reshape(-1, dim * 2, psz, psz)).reshape(
            batch_size, num_objs, dim, psz, psz
        )
        return out


class predictor_module(nn.Module):
    """
    提供K个目标的状态向量和特征�?(B 16 H W ) 生成 一个环境向�?(K个目标共�?)
    Input: nodes (states + global) (B K S+d^appr)
    outpit: next-step node states & displacement vector (B K S)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        appr_dim,
        embed_size=128,
        hidden_size=256,
        num_gn=4,
        num_layer=2,
        appr_width=28,
        is_rnn=False,
    ):
        super(predictor_module, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均 等价于全局平均池化
        self.encoder = GCN(
            in_size=input_dim + appr_dim,
            out_size=embed_size,
            hidden_size=hidden_size,
            n_layers=num_layer,
        )
        self.decoder = GCN(
            in_size=embed_size,
            out_size=output_dim,
            hidden_size=hidden_size,
            n_layers=num_layer,
        )
        gns = []
        for _ in range(num_gn):
            gns.append(
                GCN(in_size=embed_size, out_size=embed_size, hidden_size=hidden_size)
            )
        self.predictor = nn.ModuleList(gns)

        self.rnn_layer = nn.GRU(embed_size, embed_size, num_layers=2, batch_first=True)
        self.appr = appr_dim
        appr_process = [
            nn.Flatten(start_dim=2),
            nn.Linear(appr_width * appr_width, 1),
            nn.Tanh(),
        ]
        self.appr_process = nn.Sequential(*appr_process)

        self.is_rnn = is_rnn

    def forward(self, inpt, args, env_features=None, pos_dim=0, first_timestep=False):
        B, K, S = inpt.shape
        if pos_dim:
            inpts = positional_encoding(inpt, num_encoding_functions=pos_dim)
        else:
            inpts = inpt
        if self.appr:
            env_vector = (
                self.pool(env_features).view(B, 1, -1).repeat(1, K, 1).tanh()
            )  # 范围 [-1,1]
            node_vetor = torch.cat([inpts, env_vector], dim=-1)
        else:
            node_vetor = inpts

        embed_inpt = self.encoder(node_vetor)
        if self.is_rnn:
            embed_inpt = embed_inpt.view(B * K, 1, -1)
            if first_timestep:
                # self.hidden_state = None
                embed_outpt, self.hidden_state = self.rnn_layer(embed_inpt)
            else:
                embed_outpt, self.hidden_state = self.rnn_layer(
                    embed_inpt, self.hidden_state
                )
            embed_outpt = embed_outpt.view(B, K, -1)
        else:
            embed_outpt = embed_inpt

        for i in range(len(self.predictor)):  # 4步消息传递矩�?
            embed_outpt = self.predictor[i](embed_outpt) + embed_outpt

        delta = self.decoder(embed_outpt)  # B K S

        return delta


class RelativeContactAdapter(nn.Module):
    """Translation-invariant pairwise correction for near, approaching objects."""

    def __init__(self, state_dim, num_velocity, hidden_size=128, radius=0.5):
        super().__init__()
        self.num_velocity = num_velocity
        self.radius = radius
        # receiver/sender state without absolute xy, plus:
        # relative xy, relative velocity, distance, radial velocity, two scales.
        edge_dim = 2 * (state_dim - 2) + 8
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
        # The adapter is initially an exact no-op for checkpoint-compatible FT.
        nn.init.zeros_(self.node_mlp[-1].weight)
        nn.init.zeros_(self.node_mlp[-1].bias)

    def geometric_features(self, state):
        receiver = state[:, :, None, :]
        sender = state[:, None, :, :]
        rel_xy = sender[..., :2] - receiver[..., :2]
        if self.num_velocity > 0:
            rel_velocity = sender[..., 2:4] - receiver[..., 2:4]
        else:
            rel_velocity = torch.zeros_like(rel_xy)
        distance = torch.linalg.vector_norm(rel_xy, dim=-1, keepdim=True)
        radial_velocity = (
            rel_xy * rel_velocity
        ).sum(-1, keepdim=True) / distance.clamp_min(1e-6)
        receiver_scale = receiver[..., -1:].expand_as(distance)
        sender_scale = sender[..., -1:].expand_as(distance)
        geometry = torch.cat(
            [
                rel_xy,
                rel_velocity,
                distance,
                radial_velocity,
                receiver_scale,
                sender_scale,
            ],
            dim=-1,
        )
        return geometry, distance, radial_velocity

    def contact_gate(self, state):
        _, distance, radial_velocity = self.geometric_features(state)
        K = state.shape[1]
        proximity = torch.exp(-0.5 * (distance / self.radius) ** 2)
        approaching = torch.sigmoid(-radial_velocity / 0.05)
        nonself = 1.0 - torch.eye(
            K, device=state.device, dtype=state.dtype
        ).view(1, K, K, 1)
        return proximity * approaching * nonself

    def forward(self, state):
        B, K, _ = state.shape
        receiver = state[:, :, None, :].expand(-1, -1, K, -1)
        sender = state[:, None, :, :].expand(-1, K, -1, -1)
        geometry, _, _ = self.geometric_features(state)
        gate = self.contact_gate(state)
        edge_input = torch.cat(
            [receiver[..., 2:], sender[..., 2:], geometry], dim=-1
        )
        messages = self.edge_mlp(edge_input) * gate
        return self.node_mlp(messages.sum(dim=2))


class LongHorizonDeltaCalibrator(nn.Module):
    """Zero-initialized bounded correction on top of the original GSGN delta."""

    def __init__(
        self,
        embed_size,
        state_dim,
        num_velocity,
        hidden_size=128,
        scope="app",
        correction_scale=0.05,
    ):
        super().__init__()
        if scope not in ("app", "full"):
            raise ValueError("unknown calibrator scope: %s" % scope)
        self.scope = scope
        self.app_start = 2 + 2 * num_velocity
        self.correction_scale = correction_scale
        self.net = nn.Sequential(
            nn.Linear(embed_size + state_dim + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, embedding, state, rollout_phase=0.0):
        phase = torch.as_tensor(
            rollout_phase, device=state.device, dtype=state.dtype
        )
        if phase.numel() == 1:
            phase = phase.expand(*state.shape[:2], 1)
        else:
            phase = phase.reshape(*state.shape[:2], 1)
        raw = self.net(torch.cat([embedding, state, phase], dim=-1))
        correction = self.correction_scale * torch.tanh(raw)
        if self.scope == "app":
            correction = torch.cat(
                [
                    torch.zeros_like(correction[..., :self.app_start]),
                    correction[..., self.app_start:],
                ],
                dim=-1,
            )
        return correction


class TemporalHistoryAdapter(nn.Module):
    """Zero-initialized temporal correction over each object's recent states."""

    def __init__(self, state_dim, history=6, hidden_size=128, num_heads=4):
        super().__init__()
        self.history = history
        self.input_proj = nn.Linear(state_dim, hidden_size)
        self.position = nn.Parameter(torch.zeros(1, history, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, state_history):
        # state_history: [B, L, K, S], ending at the current input state.
        if state_history.shape[1] > self.history:
            state_history = state_history[:, -self.history:]
        if state_history.shape[1] < self.history:
            pad = state_history[:, :1].expand(
                -1, self.history - state_history.shape[1], -1, -1
            )
            state_history = torch.cat([pad, state_history], dim=1)
        batch, length, objects, state_dim = state_history.shape
        sequence = state_history.permute(0, 2, 1, 3).reshape(
            batch * objects, length, state_dim
        )
        encoded = self.input_proj(sequence) + self.position[:, :length]
        correction = self.output(self.encoder(encoded)[:, -1])
        return correction.view(batch, objects, 2)


class WholeRolloutTrajectoryCorrector(nn.Module):
    """Bidirectional, non-autoregressive XY correction for a complete rollout.

    Each object is processed as one temporal sequence containing the observed
    conditioning states followed by every state from the baseline rollout.
    The baseline is never fed back through this module, so residual errors do
    not compound autoregressively.
    """

    def __init__(
        self,
        state_dim,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        correction_scale=0.05,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("corrector hidden_size must be divisible by num_heads")
        self.correction_scale = correction_scale
        self.input_proj = nn.Linear(state_dim, hidden_size)
        self.segment_embedding = nn.Embedding(2, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2),
        )
        # Exact checkpoint-compatible no-op until the output head is trained.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def timestep_embedding(length, dim, device, dtype):
        """Deterministic sinusoidal embeddings for arbitrary rollout lengths."""
        half = dim // 2
        if half == 0:
            return torch.zeros(1, length, dim, device=device, dtype=dtype)
        exponent = -math.log(10000.0) * torch.arange(
            half, device=device, dtype=dtype
        ) / max(half - 1, 1)
        phase = torch.arange(length, device=device, dtype=dtype)[:, None]
        embedding = torch.cat(
            [torch.sin(phase * exponent.exp()), torch.cos(phase * exponent.exp())],
            dim=-1,
        )
        if embedding.shape[-1] < dim:
            embedding = F.pad(embedding, (0, dim - embedding.shape[-1]))
        return embedding.unsqueeze(0)

    def forward(self, conditioning_states, baseline_trajectory):
        # conditioning_states: [B, C, K, S]
        # baseline_trajectory: [B, T, K, S]
        if conditioning_states.ndim != 4 or baseline_trajectory.ndim != 4:
            raise ValueError("corrector inputs must have shape [B, time, K, state]")
        if (
            conditioning_states.shape[0] != baseline_trajectory.shape[0]
            or conditioning_states.shape[2:] != baseline_trajectory.shape[2:]
        ):
            raise ValueError("conditioning and rollout batch/object/state shapes differ")
        batch, pred_length, objects, state_dim = baseline_trajectory.shape
        cond_length = conditioning_states.shape[1]
        sequence = torch.cat([conditioning_states, baseline_trajectory], dim=1)
        sequence = sequence.permute(0, 2, 1, 3).reshape(
            batch * objects, cond_length + pred_length, state_dim
        )
        encoded = self.input_proj(sequence)
        encoded = encoded + self.timestep_embedding(
            encoded.shape[1], encoded.shape[2], encoded.device, encoded.dtype
        )
        segments = torch.cat(
            [
                torch.zeros(cond_length, device=encoded.device, dtype=torch.long),
                torch.ones(pred_length, device=encoded.device, dtype=torch.long),
            ]
        )
        encoded = encoded + self.segment_embedding(segments).unsqueeze(0)
        raw = self.output(self.encoder(encoded)[:, -pred_length:])
        residual = self.correction_scale * torch.tanh(raw)
        return residual.view(batch, objects, pred_length, 2).permute(0, 2, 1, 3)

    def correct(self, conditioning_states, baseline_trajectory):
        residual = self(conditioning_states, baseline_trajectory)
        return torch.cat(
            [
                baseline_trajectory[..., :2] + residual,
                baseline_trajectory[..., 2:],
            ],
            dim=-1,
        )


class predictor_module_with_appr(nn.Module):
    """
    提供K个目标的状态向量生成对应的 高斯图， 结合表征背景的特征图 生成 环境信息的表�?(向量)
    Input: nodes (states + global) (B K S+d^appr)
    outpit: (next-step node states) & displacement vector (B K S)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        appr_dim,
        embed_size=128,
        hidden_size=256,
        num_gn=4,
        appr_width=28,
        num_layer=2,
        is_rnn=False,
        num_key=4,
        use_contact_adapter=False,
        num_velocity=0,
        contact_radius=0.5,
        use_long_horizon_calibrator=False,
        calibrator_scope="app",
        calibrator_scale=0.05,
        use_temporal_adapter=False,
        temporal_history=6,
    ):
        super(predictor_module_with_appr, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均 等价于全局平均池化

        self.encoder = GCN(
            in_size=input_dim + appr_dim,
            out_size=embed_size,
            hidden_size=hidden_size,
            n_layers=num_layer,
        )
        self.decoder = GCN(
            in_size=embed_size,
            out_size=output_dim,
            hidden_size=hidden_size,
            n_layers=num_layer,
        )
        # MLP的层�? = n_layers + 2
        # self.encoder = MLP(in_size=input_dim + appr_dim, h_size=hidden_size,
        #                    out_size=embed_size, n_layers=num_layer)
        # self.decoder = MLP(in_size=embed_size, h_size=hidden_size,
        #                    out_size=output_dim, n_layers=num_layer)  # 好奇怪达�?
        gns = []
        for _ in range(num_gn):
            gns.append(
                GCN(in_size=embed_size, out_size=embed_size, hidden_size=hidden_size)
            )
            # gns.append(MLP(in_size=embed_size,h_size=hidden_size,out_size=embed_size,n_layers=num_layer))
        self.predictor = nn.ModuleList(gns)

        self.appr = appr_dim

        # self.max_pool = nn.MaxPool2d(kernel_size=5)  # 全局最大池�?
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.rnn_layer = nn.GRU(embed_size, embed_size, num_layers=2, batch_first=True)

        appr_process = [
            nn.Flatten(start_dim=2),
            nn.Linear(appr_width * appr_width, 1),
            nn.Tanh(),
        ]
        self.appr_process = nn.Sequential(*appr_process)
        # self.group_layer = nn.Conv2d(
        #     in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.is_rnn = is_rnn
        self.hidden_state = None
        self.contact_adapter = (
            RelativeContactAdapter(
                output_dim,
                num_velocity,
                hidden_size=min(hidden_size, 128),
                radius=contact_radius,
            )
            if use_contact_adapter
            else None
        )
        self.long_horizon_calibrator = (
            LongHorizonDeltaCalibrator(
                embed_size,
                output_dim,
                num_velocity,
                hidden_size=min(hidden_size, 128),
                scope=calibrator_scope,
                correction_scale=calibrator_scale,
            )
            if use_long_horizon_calibrator
            else None
        )
        self.temporal_adapter = (
            TemporalHistoryAdapter(
                output_dim,
                history=temporal_history,
                hidden_size=min(hidden_size, 128),
            )
            if use_temporal_adapter
            else None
        )

    def forward(
        self,
        inpt,
        args,
        env_features=None,
        pos_dim=0,
        first_timestep=False,
        rollout_phase=0.0,
        state_history=None,
    ):
        # B, K, _ = inpt.shape

        B, K, S = inpt.shape
        if pos_dim:
            inpts = positional_encoding(inpt, num_encoding_functions=pos_dim)
        else:
            inpts = inpt

        if self.appr:
            # gaussian_maps B K H W
            # env_features B 16 H W
            B, C, H, W = env_features.shape

            # 1. 第一种方�? 坐标&尺寸生成的高斯图点乘特征图后做全局平均池化
            # 5min 一个epoch 啊这啊这啊这 怎么这么慢啊
            # if S <= 2:
            #     z_scale = inpt[..., 2:]
            # else:
            #     z_scale = inpt[..., -1:]
            if args.n_coefficients > 0 or getattr(args, "use_scale", False):
                z_scale = inpt[..., -1:]
            else:
                z_scale = inpt[..., S:]

            gaussian_maps = get_keypoints_map(
                args, env_features, inpt[..., 0:2], z_scale
            )  # 只考虑2d坐标�?,深度参数默认�?1 对应 None 的情�?
            # 第三种方法了属实�?
            # gaussian_maps = get_keypoints_map_Filter(
            #     args, env_features, inpt[..., 0:2], inpt[..., -args.n_coefficients-1:])  # B K*C H W C=4�? 怪不得报�?
            # # print(gaussian_maps.shape)
            # appr_maps = env_features[:, None, :, :,
            #                          :] * gaussian_maps[:, :, None, :, :]  # B K*4 c=16 H W
            # appr_maps = appr_maps.reshape(B, K, -1, C, H, W)
            # appr_maps = self.group_layer(appr_maps.transpose(
            #     2, 3).reshape(B*K*C, -1, H, W))  # B*K*16 1 H W

            # appr_feats B K 16
            reduction = getattr(args, "agg_reduction", "sum")
            if reduction == "sum":
                # Same as (env * gauss).sum(h).sum(w), without the B K C H W tensor.
                env_vector = torch.einsum(
                    "bchw,bkhw->bkc", env_features, gaussian_maps
                )
            else:
                appr_maps = env_features[:, None, :, :, :] * gaussian_maps[
                    :, :, None, :, :
                ]
                if reduction == "avg":
                    env_vector = self.pool(
                        appr_maps.reshape(-1, C, H, W)
                    ).reshape(B, K, C)
                elif reduction == "max":
                    env_vector = self.max_pool(
                        appr_maps.reshape(-1, C, H, W)
                    ).reshape(B, K, C)
                else:
                    raise ValueError("unknown agg_reduction: %s" % reduction)
            # 2. 第二种方�? 根据坐标裁剪特征�?(获得局部的特征周围)后做全局最大池�?
            # 关键的问题在于：裁剪速度太慢,严重影响了模型效�? 8min
            #  B C H W + B K 2 ----> B K C 5 5
            # crop_features = crop_feature_map(
            #     env_features, inpt[..., 0:2], crop_size=5)
            # # B K C 5 5 ----> B*K C 5 5 -->B*K C 1 1 -->  B K C z
            # env_vector = self.max_pool(
            #     crop_features.reshape(-1, C, 5, 5)).reshape(B, -1, C)
            node_vetor = torch.cat([inpts, env_vector], dim=-1)
        else:
            node_vetor = inpts

        embed_inpt = self.encoder(node_vetor)

        if self.is_rnn:
            embed_inpt = embed_inpt.view(B * K, 1, -1)
            if first_timestep:
                embed_outpt, self.hidden_state = self.rnn_layer(embed_inpt)
            else:
                embed_outpt, self.hidden_state = self.rnn_layer(
                    embed_inpt, self.hidden_state
                )
        else:
            embed_outpt = embed_inpt

        embed_outpt = embed_outpt.view(B, K, -1)

        for i in range(len(self.predictor)):  # 4步消息传递矩�?
            embed_outpt = self.predictor[i](embed_outpt) + embed_outpt
            # embed_outpt = self.predictor[i](embed_outpt) # 有没有可以是因为 梯度爆炸�???? 模型太大，样本太少，导致难以训练，而跨层连接做到了

        delta = self.decoder(embed_outpt)  # B K S
        if self.contact_adapter is not None:
            motion_correction = self.contact_adapter(inpt)
            delta = torch.cat(
                [delta[..., :2] + motion_correction, delta[..., 2:]], dim=-1
            )
        if self.temporal_adapter is not None:
            if state_history is None:
                state_history = inpt.unsqueeze(1)
            temporal_correction = self.temporal_adapter(state_history)
            delta = torch.cat(
                [delta[..., :2] + temporal_correction, delta[..., 2:]], dim=-1
            )
        if self.long_horizon_calibrator is not None:
            delta = delta + self.long_horizon_calibrator(
                embed_outpt, inpt, rollout_phase
            )

        return delta


class predictor_module_mlp(nn.Module):
    """
    Same architecture as predictor_module_with_appr but with GCN blocks replaced
    by MLPs (no object-object message passing). Used for the P=MLP ablation rows.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        appr_dim,
        embed_size=128,
        hidden_size=256,
        num_gn=4,
        appr_width=28,
        num_layer=2,
        is_rnn=False,
        num_key=4,
    ):
        super(predictor_module_mlp, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.encoder = MLP(
            in_size=input_dim + appr_dim,
            out_size=embed_size,
            h_size=hidden_size,
            n_layers=num_layer,
        )
        self.decoder = MLP(
            in_size=embed_size,
            out_size=output_dim,
            h_size=hidden_size,
            n_layers=num_layer,
        )
        fcs = []
        for _ in range(num_gn):
            fcs.append(
                MLP(
                    in_size=embed_size,
                    out_size=embed_size,
                    h_size=hidden_size,
                    n_layers=num_layer,
                )
            )
        self.predictor = nn.ModuleList(fcs)

        self.appr = appr_dim
        self.is_rnn = is_rnn
        self.hidden_state = None

    def forward(self, inpt, args, env_features=None, pos_dim=0, first_timestep=False):
        B, K, S = inpt.shape
        if pos_dim:
            inpts = positional_encoding(inpt, num_encoding_functions=pos_dim)
        else:
            inpts = inpt

        if self.appr:
            B, C, H, W = env_features.shape
            if args.n_coefficients > 0 or getattr(args, "use_scale", False):
                z_scale = inpt[..., -1:]
            else:
                z_scale = inpt[..., S:]
            gaussian_maps = get_keypoints_map(
                args, env_features, inpt[..., 0:2], z_scale
            )
            reduction = getattr(args, "agg_reduction", "sum")
            if reduction == "sum":
                env_vector = torch.einsum(
                    "bchw,bkhw->bkc", env_features, gaussian_maps
                )
            else:
                appr_maps = env_features[:, None, :, :, :] * gaussian_maps[
                    :, :, None, :, :
                ]
                if reduction == "avg":
                    env_vector = self.pool(
                        appr_maps.reshape(-1, C, H, W)
                    ).reshape(B, K, C)
                elif reduction == "max":
                    env_vector = self.max_pool(
                        appr_maps.reshape(-1, C, H, W)
                    ).reshape(B, K, C)
                else:
                    raise ValueError("unknown agg_reduction: %s" % reduction)
            node_vetor = torch.cat([inpts, env_vector], dim=-1)
        else:
            node_vetor = inpts

        embed_inpt = self.encoder(node_vetor)
        embed_outpt = embed_inpt
        for i in range(len(self.predictor)):
            embed_outpt = self.predictor[i](embed_outpt) + embed_outpt

        delta = self.decoder(embed_outpt)  # B K S

        return delta


class predictor_module_apprmap(nn.Module):
    """
    不同在于 GN的输入是特征�? B K 16 28 28 面向特征图的GN
    无法引入速度历史速度！！?? ---- 一大问�?
    明天康康他的原代码是怎么设计 �? 特征�? 提取 参数(2D坐标�?)--- 自己的设计问�?
    Input: nodes (states + global) (B K S+d^appr)
    outpit: (next-step node states) & displacement vector (B K S)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        appr_dim,
        embed_size=128,
        hidden_size=256,
        num_gn=4,
        appr_width=28,
        is_rnn=False,
        num_layer=2,
    ):
        super(predictor_module_apprmap, self).__init__()

        gns = []
        for _ in range(num_gn):
            gns.append(Interaction_Network_3D(dim_objs=16))

        self.predictor = nn.ModuleList(gns)
        self.flat_layer = nn.Sequential(
            nn.Linear(112 * 112 // 16, 512),
            nn.ReLU(),  # 图像尺寸而变�?
            # nn.Linear(2048, 1024), nn.ReLU(),
            # nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, input_dim),
        )
        # self.group_layer = nn.Conv2d(
        #     in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1, groups=16)
        self.group_layer = nn.Conv2d(
            in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1
        )
        self.bottle_layer = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1
        )

    def forward(self, inpt, args, env_features=None, pos_dim=0, first_timestep=False):
        # env_features B 16 H W gaussian_maps B K H W
        K = inpt.shape[1]
        B, C, H, W = env_features.shape  # C=16
        # print(inpt.shape)
        # print(inpt[..., 2:].shape)
        # print(inpt[0, 0])
        # print(inpt[..., 2:] == None)
        if inpt.shape[-1] == 7:  # 只针�?4个旋转参数的Filter-Cophy感知模块，输�?7个参�?
            gaussian_maps = get_keypoints_map_Filter(
                args,
                env_features,
                inpt[..., 0:2],
                inpt[..., -args.n_coefficients - 1 :],
            )  # B K*C H W C=4�? 怪不得报�?
            # print(gaussian_maps.shape)
            appr_maps = (
                env_features[:, None, :, :, :] * gaussian_maps[:, :, None, :, :]
            )  # B K*4 c=16 H W
            appr_maps = appr_maps.reshape(B, K, -1, C, H, W)
            appr_maps = self.group_layer(
                appr_maps.transpose(2, 3).reshape(B * K * C, -1, H, W)
            )  # B*K*16 1 H W
            # gaussian_maps = self.group_layer(gaussian_maps_filter)  # 主要是降低通道数量
        else:  # 自定义的感知模块
            gaussian_maps = get_keypoints_map(
                args, env_features, inpt[..., 0:2], inpt[..., 2:]
            )  # 状态向量分�?2D坐标 以及 其他
            appr_maps = env_features[:, None, :, :, :] * gaussian_maps[:, :, None, :, :]
        # print(gaussian_maps.shape)
        # appr_maps B K 16 H W

        # print(appr_maps.shape)
        appr_maps = appr_maps.reshape(B, K, -1, H, W)  # B K 16 H W

        embed_inpt = appr_maps
        embed_outpt = embed_inpt

        for i in range(len(self.predictor)):  # 4步消息传递矩�?
            embed_outpt = self.predictor[i](embed_outpt)  # 直接预测下一步的特征�?,没有残差连接

        embed_outpt = self.bottle_layer(embed_outpt.reshape(B * K, -1, H, W)).view(
            B, K, H, W
        )

        delta = self.flat_layer(
            torch.flatten(embed_outpt, start_dim=2)
        )  # B K 16 28 28  线性变�? �? 1*1 卷积来降�?
        return delta


# class predictor_module(nn.Module):
#     """
#     Input: nodes (states + global) (B K S+d^appr)
#     outpit: next-step node states & displacement vector (B K S)
#     """

#     def __init__(self, input_dim, output_dim, appr_dim, embed_size=128, hidden_size=256, num_gn=4, appr_width=28):
#         super(predictor_module, self).__init__()

#         self.pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均 等价于全局平均池化
#         # self.encoder = MLP(in_size=input_dim + appr_dim,
#         #                    out_size=embed_size, h_size=hidden_size)
#         # self.encoder = Interaction_Network_1D(
#         #     dim_objs=input_dim + appr_dim, dim_out=embed_size)
#         #self.decoder = GCN(in_size=embed_size, out_size=output_dim)

#         self.encoder = GCN(in_size=input_dim + appr_dim,
#                            out_size=embed_size, hidden_size=hidden_size)
#         self.decoder = GCN(in_size=embed_size,
#                            out_size=output_dim, hidden_size=hidden_size)
#         # 激活函数由 Relu 改为 tanh MLP = （n_layer+1�?*(Linear+activation laryer) + 1 * Linear tanh好像不是适合做这�?
#         # 1/7的实验都是由tanh激活函数主导的~~
#         # 1/8的实验修正回Relu~
#         # self.decoder = MLP(in_size=embed_size,
#         #                    out_size=output_dim, h_size=hidden_size, layernorm=False)

#         gns = []
#         for _ in range(num_gn):
#             # gns.append(Interaction_Network_1D(dim_objs=embed_size))
#             gns.append(GCN(in_size=embed_size, out_size=embed_size))
#         self.predictor = nn.ModuleList(gns)
#         self.appr = appr_dim
#         appr_process = [nn.Flatten(start_dim=2), nn.Linear(
#             appr_width*appr_width, 1), nn.Tanh()]
#         self.appr_process = nn.Sequential(*appr_process)

#     def forward(self, inpt, env_features=None, pos_dim=0):

#         B, K, S = inpt.shape
#         if pos_dim:
#             inpts = positional_encoding(inpt, num_encoding_functions=pos_dim)
#         else:
#             inpts = inpt
#         if self.appr:
#             env_vector = self.pool(env_features).view(
#                 B, 1, -1).repeat(1, K, 1).tanh()  # 范围 [0-1]
#             # env_vector = self.appr_process(env_features).reshape(
#             #     B, 1, -1).repeat(1, K, 1)
#             node_vetor = torch.cat([inpts, env_vector], dim=-1)
#         else:
#             node_vetor = inpts

#         embed_inpt = self.encoder(node_vetor)
#         embed_outpt = embed_inpt

#         # for i in range(len(self.predictor)//2):  # 4步消息传递矩�? 每两个GN之间增加跨层连接

#         #     init_emb = embed_outpt
#         #     init_emb = self.predictor[2*i](init_emb)
#         #     embed_outpt = self.predictor[2*i+1](init_emb) + embed_outpt

#         for i in range(len(self.predictor)):  # 4步消息传递矩�?
#             embed_outpt = self.predictor[i](embed_outpt) + embed_outpt
#         #print('embed_outpt shape:', embed_outpt.shape)
#         delta = self.decoder(embed_outpt)  # B K S
#         #print('delta shape:', delta.shape)
#         #outpt = delta + inpt
#         # return delta.tanh()

#         return delta
# class whole_predictor_module(nn.Module):
#     """"
#     不考虑appr的版�?,完整的多步预�?
#     """

#     def __init__(self, args, state_dim, output_dim, embed_size=128, hidden_size=128, num_gn=4, appr_width=28, device='cpu'):
#         super(whole_predictor_module, self).__init__()

#         S = state_dim
#         if args.consider_appr:
#             appr_dim = 16
#         else:
#             appr_dim = 0
#         self.args = args
#         self.dynamics_model = predictor_module(
#             input_dim=S * (args.pos_emb*2+1), appr_dim=appr_dim, output_dim=output_dim, embed_size=embed_size, hidden_size=hidden_size, num_gn=num_gn, appr_width=appr_width).to(device)

#     def forward(self, input_state, target_state, p):
#         predict_list = []
#         # Encoder 部分
#         for i in range(input_state.shape[1]-1):
#             cur_obj = input_state[:, i]
#             delta_state = self.dynamics_model(
#                 cur_obj, pos_dim=self.args.pos_emb)
#             next_state = delta_state + cur_obj
#             predict_list.append(next_state)
#         # Decoder 部分
#         decoder_input = input_state[:, -1]
#         use_teacher_forcing = np.random.binomial(1, p, self.args.horizon)
#         for t in range(target_state.shape[1]):
#             cur_obj = decoder_input
#             delta_state = self.dynamics_model(
#                 cur_obj, pos_dim=self.args.pos_emb)
#             next_state = delta_state + cur_obj
#             predict_list.append(next_state)
#             if use_teacher_forcing[t]:
#                 decoder_input = target_state[:, t] + torch.randn_like(
#                     target_state[:, t], device=target_state.device) * self.args.noise_std * p  # 旨在训练集起�?
#             else:
#                 decoder_input = next_state

#         predicted_states = torch.stack(predict_list, dim=1)
#         return predicted_states


class predictor_module_with_cf(nn.Module):
    """
    还没有更改呢�?,再说,和基础版predictor_module 一模一�?
    Input: nodes (states + global) (B K S+d^appr)
    outpit: next-step node states & displacement vector (B K S)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        appr_dim,
        embed_size=128,
        hidden_size=256,
        num_gn=4,
        appr_width=28,
        is_rnn=False,
    ):
        super(predictor_module_with_cf, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均 等价于全局平均池化
        self.encoder = GCN(
            in_size=input_dim + appr_dim, out_size=embed_size, hidden_size=hidden_size
        )
        self.decoder = GCN(
            in_size=embed_size, out_size=output_dim, hidden_size=hidden_size
        )
        gns = []
        for _ in range(num_gn):
            gns.append(
                GCN(in_size=embed_size, out_size=embed_size, hidden_size=hidden_size)
            )
        self.predictor = nn.ModuleList(gns)

        self.rnn_layer = nn.GRU(embed_size, embed_size, num_layers=2, batch_first=True)
        self.appr = appr_dim
        appr_process = [
            nn.Flatten(start_dim=2),
            nn.Linear(appr_width * appr_width, 1),
            nn.Tanh(),
        ]
        self.appr_process = nn.Sequential(*appr_process)

        self.is_rnn = is_rnn

    def forward(self, inpt, env_features=None, pos_dim=0, first_timestep=False):
        B, K, S = inpt.shape
        if pos_dim:
            inpts = positional_encoding(inpt, num_encoding_functions=pos_dim)
        else:
            inpts = inpt
        if self.appr:
            env_vector = (
                self.pool(env_features).view(B, 1, -1).repeat(1, K, 1).tanh()
            )  # 范围 [0-1]
            node_vetor = torch.cat([inpts, env_vector], dim=-1)
        else:
            node_vetor = inpts

        embed_inpt = self.encoder(node_vetor)
        if self.is_rnn:
            embed_inpt = embed_inpt.view(B * K, 1, -1)
            if first_timestep:
                # self.hidden_state = None
                embed_outpt, self.hidden_state = self.rnn_layer(embed_inpt)
            else:
                embed_outpt, self.hidden_state = self.rnn_layer(
                    embed_inpt, self.hidden_state
                )
        else:
            embed_outpt = embed_inpt
        embed_outpt = embed_outpt.view(B, K, -1)

        for i in range(len(self.predictor)):  # 4步消息传递矩�?
            embed_outpt = self.predictor[i](embed_outpt) + embed_outpt

        delta = self.decoder(embed_outpt)  # B K S

        return delta


# dynamics_model = predictor_module_with_RNN(
#     input_dim=4, output_dim=4, appr_dim=0,is_rnn=True)
# inpts = torch.randn((8, 4, 4))
# delta = dynamics_model(inpt=inpts, first_timestep=True)
# print(delta.shape)
# print(delta[0, 0])
# inpts = torch.rand((8, 4, 4))
# delta = dynamics_model(inpt=inpts)
# print(delta.shape)
# print(delta[0, 0])
# print(dynamics_model.hidden_state.shape)

# 设计一个新的模�?
# 输入 多个目标的状态向�? B K 3，参考帧的特征图


def adjacency_matrix_from_edges(num_objs):
    A = 1 - torch.eye(num_objs).unsqueeze(0)
    # B, num_objs, _ = A.shape
    g_idx = torch.zeros([1, num_objs * num_objs, 3], dtype=torch.int64)
    for i in range(num_objs):
        for j in range(num_objs):
            g_idx[:, i * num_objs + j, 0] = i  # 发射�?
            g_idx[:, i * num_objs + j, 1] = j  # 接收�?
            g_idx[:, i * num_objs + j, 2] = A[:, i, j]  # 是否存在或边的属�?
    return g_idx

def edges_from_adjacency_matrix_new(g_idx, O):
    B, num_objs, dim_objs = O.shape
    g_idx = g_idx.repeat(B, 1, 1).to(O.device)
    i1 = g_idx[..., [0]].repeat(1, 1, dim_objs)  # B n*n 1
    i2 = g_idx[..., [1]].repeat(1, 1, dim_objs)
    y1 = torch.gather(O, 1, i1)
    y2 = torch.gather(O, 1, i2)
    r = torch.cat([y1, y2], dim=-1)  # 如果边有属性的话应该额外级�?,发现因果关系时，该边的属�? �? 可学习的
    r = r * g_idx[..., [2]]  # 该条边是否存�?
    edges_exist = g_idx[:, :, [2]]
    # B num_objs*num_objs 2*dim_objects
    E = r.view(B, num_objs * num_objs, -1)
    return E, edges_exist


class Interaction_Network(nn.Module):
    def __init__(self, dim_objs, dim_out=None, hidden_size=128, n_layer=2):
        super(Interaction_Network, self).__init__()
        if dim_out == None:
            dim_out = dim_objs
        self.dim_objs = dim_objs
        self_dynamics = [
            MLP(
                in_size=dim_objs,
                out_size=dim_objs,
                h_size=hidden_size,
                n_layers=n_layer,
            )
        ]
        rel_dynamics = [
            MLP(
                in_size=dim_objs * 2,
                out_size=dim_objs,
                h_size=hidden_size,
                n_layers=n_layer,
            )
        ]
        affector = [
            MLP(
                in_size=dim_objs,
                out_size=dim_objs,
                h_size=hidden_size,
                n_layers=n_layer,
            )
        ]
        aggregator = [
            MLP(
                in_size=dim_objs * 2,
                out_size=dim_out,
                h_size=hidden_size,
                n_layers=n_layer,
            )
        ]
        self.self_dynamics = nn.Sequential(*self_dynamics)
        self.rel_dynamics = nn.Sequential(*rel_dynamics)
        self.affector = nn.Sequential(*affector)
        self.aggregator = nn.Sequential(*aggregator)

    def forward(self, x, g_idx):
        B, num_objs, dim_objs = x.shape  # 若考虑全局属性，就将全局特征与节点特征级�?

        E, edges_exist = edges_from_adjacency_matrix_new(
            g_idx, x
        )  # 若考虑每条边的属性可以级联然后再乘以存在矩阵

        self_effect = self.self_dynamics(x.reshape(-1, dim_objs)).reshape(
            B, num_objs, -1
        )
        rel_effect = self.rel_dynamics(E.reshape(-1, dim_objs * 2)).reshape(
            B, num_objs, num_objs, -1
        )
        rel_effect = rel_effect.sum(2)  # B, num_objs, dim_objs

        all_effect = self_effect + rel_effect
        # print(all_effect.view(-1, dim_objs).shape)
        effect = self.affector(all_effect.view(-1, dim_objs)).reshape(B, num_objs, -1)

        x_out = self.aggregator(
            torch.cat([x, effect], dim=2).view(-1, dim_objs * 2)
        ).reshape(B, num_objs, -1)
        return x_out


class predictor_module_with_appr_new(nn.Module):
    """
    将GCN用Interaction_Network_1D代替
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        appr_dim,
        args=None,
    ):
        super(predictor_module_with_appr_new, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.encoder = Interaction_Network(
            dim_objs=input_dim + appr_dim,
            dim_out=args.embed_size,
            hidden_size=args.hidden_size,
            n_layer=args.num_layer,
        )
        self.decoder = Interaction_Network(
            dim_objs=args.embed_size,
            dim_out=output_dim,
            hidden_size=args.hidden_size,
            n_layer=args.num_layer,
        )
        INs = []
        for _ in range(args.num_gn):
            INs.append(
                Interaction_Network(
                    dim_objs=args.embed_size,
                    dim_out=args.embed_size,
                    hidden_size=args.hidden_size,
                    n_layer=args.num_layer,
                )
            )
        self.predictor = nn.ModuleList(INs)
        self.appr = appr_dim
        self.g_idx = adjacency_matrix_from_edges(args.n_keypoints)

    def forward(self, inpt, env_features=None, args=None):
        B, K, S = inpt.shape
        if self.appr:
            B, C, H, W = env_features.shape
            if args.n_coefficients > 0 or getattr(args, "use_scale", False):
                z_scale = inpt[..., -1:]
            else:
                z_scale = inpt[..., S:]
            gaussian_maps = get_keypoints_map(
                args, env_features, inpt[..., 0:2], z_scale
            )
            appr_maps = env_features[:, None, :, :, :] * gaussian_maps[:, :, None, :, :]
            env_vector = appr_maps.sum(-1).sum(-1)
            node_vetor = torch.cat([inpt, env_vector], dim=-1)
        else:
            node_vetor = inpt

        embed_outpt = self.encoder(node_vetor, self.g_idx)
        for i in range(len(self.predictor)):  # 4步消息传递矩�?
            embed_outpt = self.predictor[i](embed_outpt, self.g_idx) + embed_outpt

        delta = self.decoder(embed_outpt, self.g_idx)
        return delta


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import argparse

if __name__ == "__main__":
    B = 16
    num_objs = 4
    device = "cpu"
    g_idx = adjacency_matrix_from_edges(num_objs, B)

    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--epoch",
            default=300,
            type=int,
            help="Number of Epoch for training. Can be set to 0 for evaluation",
        )
        parser.add_argument(
            "--batchsize",
            default=16,
            type=int,
            help="Batchsize. Can be set to 1 for evaluation",
        )
        parser.add_argument("--lr", default=5e-4, type=float, help="Learning rate")
        parser.add_argument(
            "--dataset",
            default="blocktowerCF",
            type=str,
            choices=["blocktowerCF", "ballsCF"],
            help="Dataset used for training or evaluation",
        )
        parser.add_argument("--seed", type=int, default=1024)
        parser.add_argument(
            "--given_frames",
            type=int,
            default=1,
            help="Given previous n frames to predict states of future frames, must > 1",
        )
        parser.add_argument(
            "--horizon",
            type=int,
            default=25,
            help="predict states of horizon-step future frames",
        )
        parser.add_argument("--num_velocity", default=0, type=int)
        parser.add_argument(
            "--name",
            default="my_model_with_env",
            type=str,
            help="Name for weight saving, consider feature map around object positions",
        )
        # parser.add_argument('--pre_train', action='store_true')
        parser.add_argument("--weight_path", type=str, default="")
        parser.add_argument("--Perception_model_weight_path", type=str, default="")
        parser.add_argument(
            "--Perception_model",
            type=str,
            default="derendering",
            help="or derendering_with_rotate",
        )
        parser.add_argument(
            "--Inference_model",
            type=str,
            default="predictor_module",
            help="or else predictor_module_with_appr, predictor_module_apprmap",
        )
        parser.add_argument("--num_gn", default=4, type=int)
        parser.add_argument("--num_layer", default=2, type=int)
        parser.add_argument("--embed_size", default=128, type=int)
        parser.add_argument("--hidden_size", default=256, type=int)
        parser.add_argument(
            "--is_rnn",
            action="store_true",
            help="if use rnn why predict next state vector",
        )
        parser.add_argument(
            "--pos_emb", default=0, type=int, help="dim of pos embedding like nerf"
        )
        parser.add_argument(
            "--n_coefficients",
            default=4,
            type=int,
            help="Number of coefficients to use",
        )
        parser.add_argument(
            "--n_keypoints", default=4, type=int, help="Number of keypoints to use"
        )
        parser.add_argument(
            "--std", default=0.1, type=float, help="std of guassian-like heatmap"
        )
        parser.add_argument(
            "--consider_appr",
            action="store_true",
            help="if consider feature map around each object",
        )
        parser.add_argument(
            "--noise_std",
            type=float,
            default=0.0,
            help="add noise in training phase, 0.0003",
        )
        parser.add_argument(
            "--train_mode_verbose",
            default=0,
            type=int,
            help="0: free running mode, 1: teacher forcing, 2: Curriculum Learning",
        )
        parser.add_argument("--warm_epoch", default=5, type=int)
        parser.add_argument("--decay_epoch", default=20, type=int)
        parser.add_argument("--df_epoch", default=20, type=int)
        parser.add_argument("--autoenc_loss", action="store_true")
        parser.add_argument(
            "--loss_type",
            default="absoluate",
            type=str,
            help="predict states or states displacement, absoluate/relative/merge",
        )
        parser.add_argument("--pos_weight", default=1.0, type=float)
        parser.add_argument("--discount_factor", action="store_true")
        return parser.parse_args()

    args = get_args()
    model = predictor_module_with_appr_new(input_dim=6, output_dim=6, appr_dim=16)
