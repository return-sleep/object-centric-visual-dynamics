#!/usr/bin/env python
# coding: utf-8
"""
reader: Huilin XU
date: 2022/7/30
description :node state auto-encoder + GCN + MLP + SpatioTemporalLSTMCell + CIN(卷积IN)
"""

import torch
import torch.nn as nn

class InterNet(nn.Module):
    def __init__(self, in_feat_dim):
        super(InterNet, self).__init__()
        self.in_feat_dim = in_feat_dim
        # self dynamics, input object state, output new object state
        self_dynamics = [
            nn.Conv2d(self.in_feat_dim, self.in_feat_dim, kernel_size=3,
                      stride=1, padding=1), nn.ReLU(inplace=True)
        ]
        rel_dynamics = [
            nn.Conv2d(self.in_feat_dim * 2, self.in_feat_dim,
                      kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)
        ]
        aggregator = [
            nn.Conv2d(self.in_feat_dim * 2, self.in_feat_dim,
                      kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)
        ]

        N_EXTRA_SELFD_F = 0
        for _ in range(N_EXTRA_SELFD_F):
            self_dynamics.append(nn.Conv2d(self.in_feat_dim, self.in_feat_dim,
                                           kernel_size=3, stride=1, padding=1))
            self_dynamics.append(nn.ReLU(inplace=True))
        self.self_dynamics = nn.Sequential(*self_dynamics)
        # relation dynamics, input pairwise object states, output new object state
        
        N_EXTRA_RELD_F = 0 
        for _ in range(N_EXTRA_RELD_F):
            rel_dynamics.append(nn.Conv2d(self.in_feat_dim, self.in_feat_dim,
                                          kernel_size=3, stride=1, padding=1))
            rel_dynamics.append(nn.ReLU(inplace=True))
        self.rel_dynamics = nn.Sequential(*rel_dynamics)
        # affector
        affector = [
            nn.Conv2d(self.in_feat_dim, self.in_feat_dim, kernel_size=3,
                      stride=1, padding=1), nn.ReLU(inplace=True)
        ]
        N_EXTRA_AFFECTOR_F = 0 
        for _ in range(N_EXTRA_AFFECTOR_F):
            affector.append(nn.Conv2d(self.in_feat_dim, self.in_feat_dim,
                                      kernel_size=3, stride=1, padding=1))
            affector.append(nn.ReLU(inplace=True))
        self.affector = nn.Sequential(*affector)
        # aggregator 
        N_EXTRA_AGGREGATOR_F = 0
        for _ in range(N_EXTRA_AGGREGATOR_F):
            aggregator.append(nn.Conv2d(self.in_feat_dim, self.in_feat_dim,
                                        kernel_size=3, stride=1, padding=1))
            aggregator.append(nn.ReLU(inplace=True))
        self.aggregator = nn.Sequential(*aggregator)

    def forward(self, x, g_idx=None):
        s = x
        # of shape (b, o, dim, 7, 7)
        batch_size, num_objs, dim, psz, psz = x.shape
        x1 = x.repeat(1, num_objs - 1, 1, 1, 1)
        i1 = g_idx[..., [0], None, None].repeat(1, 1, dim, psz, psz)
        y1 = torch.gather(x1, 1, i1)
        i2 = g_idx[..., [1], None, None].repeat(1, 1, dim, psz, psz)
        y2 = torch.gather(x1, 1, i2)
        r = torch.cat([y1, y2], dim=2)
        r = r * g_idx[:, :, [2], None, None]
        r = r.reshape(-1, dim * 2, psz, psz)
        r = self.rel_dynamics(r)
        r = r.reshape(batch_size, num_objs, num_objs - 1, dim, psz, psz)
        r = r.sum(dim=2)

        x = self.self_dynamics(x.reshape(-1, dim, psz, psz)
                               ).reshape(batch_size, num_objs, dim, psz, psz)

        pred = x + r
        a = self.affector(pred.reshape(-1, dim, psz, psz)
                          ).reshape(batch_size, num_objs, dim, psz, psz)
        a = torch.cat([a, s], 2)
        out = self.aggregator(a.reshape(-1, dim * 2, psz, psz)
                              ).reshape(batch_size, num_objs, dim, psz, psz)
        return out

class MLP(nn.Module):
    """
    Functions: multiple Linear layers
    Input: [*,in_size]
    Output: [*,out_size]
    n_layer: (in-h_size) n_layer * (h_size - h_size) (h_size-out_size)
    """

    def __init__(self, in_size, h_size, out_size, n_layers, activation=nn.ReLU, batchnorm=False, n_channel=4):
        super(MLP, self).__init__()

        layer_list = []
        if batchnorm:
            # B K S -> num_feature = K
            layer_list.append(nn.BatchNorm1d(n_channel))

        layer_list.append(nn.Linear(in_size, h_size))
        layer_list.append(activation())

        for _ in range(n_layers):
            layer_list.append(nn.Linear(h_size, h_size))
            layer_list.append(activation())

        layer_list.append(nn.Linear(h_size, out_size))

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

    def __init__(self, in_size, out_size, n_layers, hidden_size, activation=nn.ReLU, batchnorm=False):
        super(GCN, self).__init__()
        # in_size, out_size, hidden_size,: Ds Dp DE
        # relation-centric function : columns represent the interaction terms
        self.f = MLP(2 * in_size, hidden_size, hidden_size, n_layers,
                     activation, batchnorm, n_channel=2 * in_size)  # batchnorm=False n_channel 没用
        # object-centric
        self.g = MLP(2 * hidden_size + in_size, hidden_size,
                     out_size, n_layers, activation, batchnorm, n_channel=4)

    def forward(self, x):
        # print(x.shape)
        B, K, S = x.shape

        x1 = x.unsqueeze(1).repeat(1, K, 1, 1)  # B K K S receiver
        # B K K S sender interaction = K(K-1)
        x2 = x.unsqueeze(2).repeat(1, 1, K, 1)

        # B K K S*2(sender+receiver) relation 没用携带任何信息，忽略
        x12 = torch.cat([x1, x2], dim=-1)
        interactions = self.f(x12.view(B * K * K, 2 * S)
                              ).view(B, K, K, -1)  # B K K hiden_size
        E = interactions * (1 - torch.eye(K).view(1, K, K, 1).repeat(B,
                            1, 1, interactions.shape[-1]).to(x.device))  # 去掉self-causation

        E = E.sum(2)  # B K hidden_size  aggragation function: sum dim=2,发送节点下标
        # 每个节点对应的X(外部力或者重力加速度相同) ->每个节点都相同 B hidden_size -> B k hidden_size
        E_sum = E.sum(1).unsqueeze(1).repeat(1, K, 1)
        # B K out_size linear layer: [anynumber H_in]
        out = self.g(torch.cat([x, E, E_sum], dim=-1))

        return out


class Encoder(nn.Module):
    """
    Input_size : B, T, K, state_size
    Output_size :  B, T, K, z_size
    """

    def __init__(self, state_size, z_size):
        super(Encoder, self).__init__()
        self.gn_encoder = GCN(in_size=state_size,
                              out_size=z_size, n_layers=0, hidden_size=32)

    def forward(self, x):
        B, T, K, S = x.shape
        z = self.gn_encoder(x.contiguous().view(B * T, K, S))
        z = z.view(B, T, K, -1)
        return z


class Decoder_independent(nn.Module):
    """
    Funtion: GCN independent process and project predictions to original keypoint space
    Input: [B T K input_size = z_size]
    Output: [B T K output_size = state_size]
    """

    def __init__(self, state_size, z_size):
        super(Decoder_independent, self).__init__()
        self.gn_decoder = GCN(
            in_size=z_size, out_size=state_size, n_layers=0, hidden_size=32)

    def forward(self, x):
        B, T, K, S = x.shape
        z = self.gn_decoder(x.contiguous().view(B * T, K, S))
        z = z.view(B, T, K, -1)
        return z


class Decoder(nn.Module):
    """
    Funtion: GRN+GCN temporally process and project predictions to original keypoint space
    Input: [B T K input_size = z_size]
    Output: [B T K output_size = state_size]
    """

    def __init__(self, state_size, z_size):
        super(Decoder, self).__init__()
        self.gn_decoder = GCN(
            in_size=z_size, out_size=state_size, n_layers=0, hidden_size=32)
        # 为什么需要rnn的形式解码呢 h_0 default to zeros
        self.rnn_decoder = nn.GRU(
            input_size=z_size, hidden_size=z_size, num_layers=1, batch_first=True)

    def forward(self, x):
        B, T, K, S = x.shape
        x = x.transpose(1, 2).reshape(B * K, T, -1)  # [B*K T z_size]
        # temporally process
        z, _ = self.rnn_decoder(x)  # [B*K T z_size]  h(t)初始化为0
        z = z.view(B, K, T, -1).transpose(1, 2)  # [B T K z_size]
        # contextualization with a graph network
        x = self.gn_decoder(z.reshape(B * T, K, -1)
                            ).reshape(B, T, K, -1)  # [B T K state_size]
        return x


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding,
                          bias=False),
                nn.LayerNorm([num_hidden * 7, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding,
                          bias=False),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding,
                          bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                          bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding,
                          bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding,
                          bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding,
                          bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                          bias=False),
            )
        self.conv_last = nn.Conv2d(
            num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(
            x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m


class ConfoundersModule(nn.Module):
    """
    Functions: 前n帧预测混杂因子
    Input: [B,T,K,S]
    Output: [B K cf_size]
    """

    def __init__(self, state_size, cf_size=32, n_layers=2, hidden_size=32, emb_size=128):
        super(ConfoundersModule, self).__init__()

        self.gn_ab = GCN(in_size=state_size, out_size=emb_size, n_layers=n_layers, hidden_size=hidden_size,
                         batchnorm=False)
        self.rnn_ab = nn.GRU(emb_size, cf_size, num_layers=2, batch_first=True)

    def forward(self, x):
        B, T, K, S = x.shape

        # [B, T, K, S] -> [B*T, K, S] -> [B*T, K, emb_size] -> [B,T,K, emb_size]
        embeddings = self.gn_ab(x.contiguous().view(
            B * T, K, S)).view(B, T, K, -1)
        # [B*K,T, emb_size]
        embeddings = embeddings.transpose(1, 2).reshape(B * K, T, -1)
        # B*K T cf_size
        output, _ = self.rnn_ab(embeddings)
        # B K T cf_size -> B K cf_size
        U = output.view(B, K, T, -1)[:, :, -1]
        return U


class DynamicModule(nn.Module):  # encoder_embedding(z_size) + cf(cf_size)
    def __init__(self, state_size, cf_size, n_layers, hidden_size, emb_size, n_gn):
        super(DynamicModule, self).__init__()

        gns = [GCN(in_size=state_size + cf_size, out_size=emb_size, n_layers=n_layers, hidden_size=hidden_size,
                   batchnorm=False)]
        for _ in range(1, n_gn):  # n_gn = 1  几层GN
            gns.append(GCN(in_size=emb_size, out_size=emb_size, n_layers=n_layers, hidden_size=hidden_size,
                           batchnorm=False))

        self.gn_cd = nn.ModuleList(gns)  # 列表形式表示多个子模块,没有定义一个网络,只是储存在一起
        self.rnn_cd = nn.GRU(emb_size, emb_size,
                             num_layers=2, batch_first=True)  # T=1时， GRU相当于GRUCell
        self.linear_cd = nn.Linear(emb_size, state_size, bias=False)

    def forward(self, inpt, hidden_cd):
        B, K, S = inpt.shape

        # B, K, S -> B, K, emb_size=128
        embedding = self.gn_cd[0](inpt)  # nn.Moudlelist 列表索引
        for i in range(1, len(self.gn_cd) - 1):  # len(self.gn_cd) = n_gn
            embedding = self.gn_cd[i](embedding)

        if len(self.gn_cd) > 1:
            embedding = self.gn_cd[-1](embedding)

        embedding = embedding.view(B * K, 1, -1)  # [B*K 1 emb_size]

        if hidden_cd is None:
            output, hidden_cd = self.rnn_cd(embedding)
        else:  # [B*K 1 emb_size] [B*K num_layer=2 embsize]
            output, hidden_cd = self.rnn_cd(
                embedding, hidden_cd)  # hidden 默认 zeros
        # [B*K 1 emb_size] -> [B*K emb_size] -> [B*K z_size] -> [B K z_size]
        delta = self.linear_cd(output[:, 0]).view(B, K, -1)
        return delta, hidden_cd
