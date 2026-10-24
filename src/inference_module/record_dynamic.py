#!/usr/bin/env python
# coding: utf-8
"""
reader: Huilin XU
date: 2022/7/31
description : Inference_model, previous t-1 frames and current frames t to predict future frames t+1
"""

import torch
import torch.nn as nn
from src.inference_module.modules import *
import numpy as np
import random


class Inference_module_v1(nn.Module):  # 8/13 07:25
    """
    Params:
        state_size: [B T K S]  S: dimension of node state
        z_size: project dynamic system into higher dimensional state space
        cf_size: size of confounder, used in GN+GRU
        emb_size: size of node embedding, used in GN
        n_layers: num of hidden layers, used in GRU
        hidden_size: size of hidden state H, used in GRU
    Return:
        cd_hat: reconstruction of cd sequence

    """

    def __init__(self, state_size=14, cf_size=32, n_layers=2, hidden_size=32, emb_size=None, n_gn=1, z_size=64):
        super().__init__()

        if emb_size is None:
            emb_size = hidden_size

        # 希望是前t-1帧预测confounder,然后级联当前第t帧的表示,dynamic_module进行变化delta, s(t+1)=s(t)+delta
        self.confounders_module = ConfoundersModule(z_size, cf_size, n_layers=n_layers, hidden_size=hidden_size,
                                                    emb_size=emb_size)
        self.dynamic_module = DynamicModule(z_size, cf_size, n_layers=n_layers, hidden_size=hidden_size,
                                            emb_size=emb_size, n_gn=n_gn)  # n_layer GCN中MLP的隐藏层数目 n_gn:多几层n_gn
        # Input: B T K S output: B T K out_size
        self.encoder = Encoder(state_size, z_size)
        self.decoder = Decoder(state_size, z_size)

    def forward(self, ab, given_frames=2, horizon=148, p=1.0):
        """
        teacher forcing
        Input:
            ab: input state sequence [B T K S]
            given_frames: given n frames to predict
            horizon: predict future horizon frames
            p: possibility of schedule sampling gt_value as input of next step
        Return: rec_ab [B horizon K S]
        """
        B, T, K, S = ab.shape
        hidden_cd = None

        assert T - given_frames >= horizon
        # B T K Z_size
        temp = self.encoder(ab)
        list_pose = [temp[:, i]
                     for i in range(given_frames)]
        list_predict = [temp[:, i]
                        for i in range(given_frames)]
        # list_pose = [self.encoder(ab[:, i]) for i in range(
        #     given_frames)]  # 初始化 state buffer + current frames

        # teacher_forcing = np.random.binomial(1, p, horizon) #一个batchsize里是否取真值不同
        use_teacher_forcing = True if random.random() < p else False  # 一个batchsize里是否取真值相同
        for t in range(horizon):  # teacher forcing

            # state_buffer 长度维持在 given_frames -1上
            state_buffer = list_pose[t:-1]
            cf = self.confounders_module(torch.stack(state_buffer, dim=1))
            input_embed = list_pose[-1]
            input_all = torch.cat([input_embed, cf], dim=-1)
            delta, hidden_cd = self.dynamic_module(input_all, hidden_cd)

            pose = input_embed + delta  # 直接相加
            # pose = delta  # 直接预测
            # skip connection 复杂化上一帧状态的映射& 动态变化 与
            # pose = self.project(torch.cat([input_embed, delta], dim=-1))
            # pose = self.linear(torch.cat([input_embed, delta], dim=-1))

            list_predict.append(pose)

            # if teacher_forcing[t] == 1:  # 真值
            if use_teacher_forcing:
                list_pose.append(temp[:, t+given_frames])
            else:  # 上一时刻的输出
                list_pose.append(pose)

        ab_hat = torch.stack(list_predict, dim=1)
        ab_hat = self.decoder(ab_hat)

        return ab_hat

    def prediction(self, given_state, horizon):
        """
        Functions: 前一时刻的预测输出,作为下一时刻的输入
        Input:
            given_state: [B given_frames K S] previous frame + current frames
            horizon: long-term prediction default=T-given_frames
        Output:
            a:[B horizon+given_frames K S] 逐时刻投影到原状态空间后拼接
        """

        hidden_cd = None
        # state_buffer = [ab[:, i] for i in range(given_frames-1)]
        temp = self.encoder(given_state)

        list_pose = [temp[:, i]
                     for i in range(temp.shape[1])]

        for t in range(horizon):  # 在低维空间的问题
            # B tau K z_size -> B K cf_size
            # state_buffer 长度维持在 given_frames -1上
            state_buffer = list_pose[t:-1]
            cf = self.confounders_module(torch.stack(state_buffer, dim=1))
            # B K z_size
            current_state = list_pose[-1]
            # B K cf_size+z_size
            input_all = torch.cat([current_state, cf], dim=-1)
            # B K z_size
            delta, hidden_cd = self.dynamic_module(input_all, hidden_cd)
            # B K z_size
            predict_pose = delta + current_state
            list_pose.append(predict_pose)
        # B horizon K z_size
        predict_state = torch.stack(list_pose, dim=1)
        predict_state = self.decoder(predict_state)

        return predict_state


class Inference_module_v2(nn.Module):  # 9/27 09:26
    """
    Params:
        state_size: [B T K S]  S: dimension of node state
        z_size: project dynamic system into higher dimensional state space
        cf_size: size of confounder, used in GN+GRU
        emb_size: size of node embedding, used in GN
        n_layers: num of hidden layers, used in GRU
        hidden_size: size of hidden state H, used in GRU
    Return:
        cd_hat: reconstruction of cd sequence

    """

    def __init__(self, state_size=7, cf_size=7, n_layers=2, hidden_size=32, emb_size=64, n_gn=4, z_size=128):
        super().__init__()

        if emb_size is None:
            emb_size = hidden_size

        # 希望是前t-1帧预测confounder,然后级联当前第t帧的表示,dynamic_module进行变化delta, s(t+1)=s(t)+delta
        self.confounders_module = ConfoundersModule(z_size, cf_size=z_size, n_layers=n_layers, hidden_size=hidden_size,
                                                    emb_size=emb_size)
        self.dynamic_module = DynamicModule(z_size, cf_size=0, n_layers=n_layers, hidden_size=hidden_size,
                                            emb_size=emb_size, n_gn=n_gn)  # n_layer GCN中MLP的隐藏层数目 n_gn:多几层n_gn
        # Input: B T K S output: B T K out_size
        self.encoder = Encoder(state_size, z_size)
        self.decoder = Decoder(state_size, z_size)
        self.gate = nn.Linear(z_size+z_size, z_size)

    def forward(self, ab, given_frames=2, horizon=148, p=1.0):
        """
        teacher forcing
        Input:
            ab: input state sequence [B T K S]
            given_frames: given n frames to predict
            horizon: predict future horizon frames
            p: possibility of schedule sampling gt_value as input of next step
        Return: rec_ab [B horizon K S]
        """
        B, T, K, S = ab.shape
        hidden_cd = None

        assert T - given_frames >= horizon
        # B T K Z_size
        temp = self.encoder(ab)
        list_pose = [temp[:, i]
                     for i in range(given_frames)]
        list_predict = [temp[:, i]
                        for i in range(given_frames)]

        use_teacher_forcing = True if random.random() < p else False  # 一个batchsize里是否取真值相同

        for t in range(horizon):  # teacher forcing

            # state_buffer 长度维持在 given_frames -1上
            state_buffer = list_pose[t:-1]  # 只考虑2帧啊啊啊啊，问题
            cf = self.confounders_module(torch.stack(
                state_buffer, dim=1))  # B Tau K cf_size

            input_embed = list_pose[-1]
            delta, hidden_cd = self.dynamic_module(
                input_embed, hidden_cd)  # 只输入当前状态

            pose_tilde = input_embed + delta  # 预测
            input_all = torch.cat([input_embed, cf], dim=-1)  # B K z_size*2
            K = torch.sigmoid(self.gate(input_all))
            pose = pose_tilde + K * (cf - pose_tilde)  # 校正

            # pose = input_embed + delta  # 直接相加

            list_predict.append(pose)

            if use_teacher_forcing:
                list_pose.append(temp[:, t+given_frames])
            else:  # 上一时刻的输出
                list_pose.append(pose)

        ab_hat = torch.stack(list_predict, dim=1)
        ab_hat = self.decoder(ab_hat)

        return ab_hat

    def prediction(self, given_state, horizon):
        """
        Functions: 前一时刻的预测输出,作为下一时刻的输入
        Input:
            given_state: [B given_frames K S] previous frame + current frames
            horizon: long-term prediction default=T-given_frames
        Output:
            a:[B horizon+given_frames K S] 逐时刻投影到原状态空间后拼接
        """

        hidden_cd = None
        # state_buffer = [ab[:, i] for i in range(given_frames-1)]
        temp = self.encoder(given_state)

        list_pose = [temp[:, i]
                     for i in range(temp.shape[1])]

        for t in range(horizon):  # 在低维空间的问题
            # B tau K z_size -> B K cf_size
            # state_buffer 长度维持在 given_frames -1上
            state_buffer = list_pose[t:-1]
            cf = self.confounders_module(torch.stack(state_buffer, dim=1))
            # B K z_size
            input_embed = list_pose[-1]
            # B K z_size
            delta, hidden_cd = self.dynamic_module(input_embed, hidden_cd)
            # B K z_size
            pose_tilde = input_embed + delta  # 预测
            # B K cf_size+z_size
            input_all = torch.cat([input_embed, cf], dim=-1)
            K = torch.sigmoid(self.gate(input_all))
            predict_pose = pose_tilde + K * (cf - pose_tilde)
            list_pose.append(predict_pose)
        # B horizon K z_size
        predict_state = torch.stack(list_pose, dim=1)
        predict_state = self.decoder(predict_state)

        return predict_state


class Inference_module_v3(nn.Module):  # 9/27 10:52
    """
    Params:
        state_size: [B T K S]  S: dimension of node state
        z_size: project dynamic system into higher dimensional state space
        cf_size: size of confounder, used in GN+GRU
        emb_size: size of node embedding, used in GN
        n_layers: num of hidden layers, used in GRU
        hidden_size: size of hidden state H, used in GRU
    Return:
        cd_hat: reconstruction of cd sequence

    """

    def __init__(self, state_size=7, cf_size=7, n_layers=2, hidden_size=32, emb_size=64, n_gn=4, z_size=128):
        super().__init__()

        if emb_size is None:
            emb_size = hidden_size

        # 希望是前t-1帧预测confounder,然后级联当前第t帧的表示,dynamic_module进行变化delta, s(t+1)=s(t)+delta
        self.confounders_module = ConfoundersModule(z_size, cf_size=z_size, n_layers=n_layers, hidden_size=hidden_size,
                                                    emb_size=emb_size)
        self.dynamic_module = DynamicModule(z_size, cf_size=0, n_layers=n_layers, hidden_size=hidden_size,
                                            emb_size=emb_size, n_gn=n_gn)  # n_layer GCN中MLP的隐藏层数目 n_gn:多几层n_gn
        # Input: B T K S output: B T K out_size
        self.encoder = Encoder(state_size, z_size)
        self.decoder = Decoder(state_size, z_size)
        self.gate = nn.Linear(z_size+z_size, z_size)

    def forward(self, ab, given_frames=2, horizon=148, p=1.0):
        """
        teacher forcing
        Input:
            ab: input state sequence [B T K S]
            given_frames: given n frames to predict
            horizon: predict future horizon frames
            p: possibility of schedule sampling gt_value as input of next step
        Return: rec_ab [B horizon K S]
        """
        B, T, K, S = ab.shape
        hidden_cd = None

        assert T - given_frames >= horizon
        # B T K Z_size
        temp = self.encoder(ab)
        list_pose = [temp[:, i]
                     for i in range(given_frames)]
        # list_predict = [temp[:, i]
        #                 for i in range(given_frames)]
        list_predict = [temp[:, 0]]

        use_teacher_forcing = True if random.random() < p else False  # 一个batchsize里是否取真值相同
        #rss = 1
        for t in range(given_frames-1):

            if t == 0:  # 初始化 还是 None?? zero??
                state_buffer = [torch.zeros_like(list_pose[-1])]
                input_embed = temp[:, t]  # 取真值
            else:
                state_buffer = list_pose[0:t]  # 看一看能不能叠起来
                input_embed = list_pose[t]  # 教师指导
                # if rss == 1:
                #     input_embed = temp[:, t]
                # else:
                #     input_embed = list_predict[-1]

            cf = self.confounders_module(torch.stack(
                state_buffer, dim=1))  # B Tau K cf_size

            delta, hidden_cd = self.dynamic_module(
                input_embed, hidden_cd)  # 只输入当前状态

            pose_tilde = input_embed + delta  # 预测
            input_all = torch.cat([input_embed, cf], dim=-1)  # B K z_size*2
            K = torch.sigmoid(self.gate(input_all))
            pose = pose_tilde + K * (cf - pose_tilde)  # 校正
            list_predict.append(pose)

        for t in range(horizon):  # teacher forcing

            # state_buffer 长度维持在 given_frames -1上
            state_buffer = list_pose[0:-1]  # 只考虑3帧啊啊啊啊，问题考虑全部帧会怎么样
            cf = self.confounders_module(torch.stack(
                state_buffer, dim=1))  # B Tau K cf_size

            input_embed = list_pose[-1]
            delta, hidden_cd = self.dynamic_module(
                input_embed, hidden_cd)  # 只输入当前状态

            pose_tilde = input_embed + delta  # 预测
            input_all = torch.cat([input_embed, cf], dim=-1)  # B K z_size*2
            K = torch.sigmoid(self.gate(input_all))
            pose = pose_tilde + K * (cf - pose_tilde)  # 校正

            # pose = input_embed + delta  # 直接相加

            list_predict.append(pose)

            if use_teacher_forcing:
                list_pose.append(temp[:, t+given_frames])
            else:  # 上一时刻的输出
                list_pose.append(pose)

        ab_hat = torch.stack(list_predict, dim=1)
        ab_hat = self.decoder(ab_hat)

        return ab_hat

    def prediction(self, given_state, horizon):
        """
        Functions: 前一时刻的预测输出,作为下一时刻的输入
        Input:
            given_state: [B given_frames K S] previous frame + current frames
            horizon: long-term prediction default=T-given_frames
        Output:
            a:[B horizon+given_frames K S] 逐时刻投影到原状态空间后拼接
        """

        hidden_cd = None
        # state_buffer = [ab[:, i] for i in range(given_frames-1)]
        temp = self.encoder(given_state)

        list_pose = [temp[:, i]
                     for i in range(temp.shape[1])]
        given_frames = temp.shape[1]

        for t in range(given_frames-1):
            if t == 0:  # 初始化 还是 None?? zero??
                state_buffer = [torch.zeros_like(list_pose[-1])]
                input_embed = temp[:, t]  # 取真值
            else:
                state_buffer = list_pose[0:t]  # 看一看能不能叠起来
                input_embed = list_pose[t]  # 教师指导
                # if rss == 1:
                #     input_embed = temp[:, t]
                # else:
                #     input_embed = list_predict[-1]

            cf = self.confounders_module(torch.stack(
                state_buffer, dim=1))  # B Tau K cf_size

            delta, hidden_cd = self.dynamic_module(
                input_embed, hidden_cd)  # 只输入当前状态

            pose_tilde = input_embed + delta  # 预测
            input_all = torch.cat([input_embed, cf], dim=-1)  # B K z_size*2
            K = torch.sigmoid(self.gate(input_all))
            pose = pose_tilde + K * (cf - pose_tilde)  # 校正
            # list_pose.append(pose)

        for t in range(horizon):  # 在低维空间的问题
            # B tau K z_size -> B K cf_size
            # state_buffer 长度维持在 given_frames -1上
            state_buffer = list_pose[0:-1]
            cf = self.confounders_module(torch.stack(state_buffer, dim=1))
            # B K z_size
            input_embed = list_pose[-1]
            # B K z_size
            delta, hidden_cd = self.dynamic_module(input_embed, hidden_cd)
            pose_tilde = input_embed + delta  # 预测

            # B K cf_size+z_size
            input_all = torch.cat([input_embed, cf], dim=-1)
            K = torch.sigmoid(self.gate(input_all))
            pose = pose_tilde + K * (cf - pose_tilde)

            # B K z_size
            list_pose.append(pose)
        # B horizon K z_size
        predict_state = torch.stack(list_pose, dim=1)
        predict_state = self.decoder(predict_state)

        return predict_state


class Inference_module_v4(nn.Module):  # 9/28 1:30
    """
    Params:
        state_size: [B T K S]  S: dimension of node state
        z_size: project dynamic system into higher dimensional state space
        cf_size: size of confounder, used in GN+GRU
        emb_size: size of node embedding, used in GN
        n_layers: num of hidden layers, used in GRU
        hidden_size: size of hidden state H, used in GRU
    Return:
        cd_hat: reconstruction of cd sequence

    """

    def __init__(self, state_size=7, cf_size=7, n_layers=2, hidden_size=32, emb_size=64, n_gn=4, z_size=128):
        super().__init__()

        if emb_size is None:
            emb_size = hidden_size

        # 希望是前t-1帧预测confounder,然后级联当前第t帧的表示,dynamic_module进行变化delta, s(t+1)=s(t)+delta
        self.confounders_module = ConfoundersModule(z_size, cf_size=z_size, n_layers=n_layers, hidden_size=hidden_size,
                                                    emb_size=emb_size)
        self.dynamic_module = DynamicModule(z_size, cf_size=0, n_layers=n_layers, hidden_size=hidden_size,
                                            emb_size=emb_size, n_gn=n_gn)  # n_layer GCN中MLP的隐藏层数目 n_gn:多几层n_gn
        # Input: B T K S output: B T K out_size
        self.encoder = Encoder(state_size, z_size)
        self.decoder = Decoder_independent(state_size, z_size)
        self.gate = GCN(z_size+z_size, z_size,
                        n_layers=0, hidden_size=32)  # B K

    def forward(self, ab, given_frames=2, horizon=148, p=1.0):
        """
        teacher forcing
        Input:
            ab: input state sequence [B T K S]
            given_frames: given n frames to predict
            horizon: predict future horizon frames
            p: possibility of schedule sampling gt_value as input of next step
        Return: rec_ab [B horizon K S]
        """
        B, T, K, S = ab.shape
        hidden_cd = None

        assert T - given_frames >= horizon
        # B T K Z_size
        temp = self.encoder(ab)
        list_pose = [temp[:, i]
                     for i in range(given_frames)]
        # list_predict = [temp[:, i]
        #                 for i in range(given_frames)]
        list_predict = [temp[:, 0]]

        use_teacher_forcing = True if random.random() < p else False  # 一个batchsize里是否取真值相同
        #rss = 1
        for t in range(given_frames-1):

            if t == 0:  # 初始化 还是 None?? zero??
                state_buffer = [torch.zeros_like(list_pose[-1])]
                input_embed = temp[:, t]  # 取真值
            else:
                state_buffer = list_pose[0:t]  # 看一看能不能叠起来
                input_embed = list_pose[t]  # 教师指导
                # if rss == 1:
                #     input_embed = temp[:, t]
                # else:
                #     input_embed = list_predict[-1]

            cf = self.confounders_module(torch.stack(
                state_buffer, dim=1))  # B Tau K cf_size

            delta, hidden_cd = self.dynamic_module(
                input_embed, hidden_cd)  # 只输入当前状态

            pose_tilde = input_embed + delta  # 预测
            input_all = torch.cat([input_embed, cf], dim=-1)  # B K z_size*2
            K = torch.sigmoid(self.gate(input_all))
            pose = pose_tilde + K * (cf - pose_tilde)  # 校正
            list_predict.append(pose)

        for t in range(horizon):  # teacher forcing

            # state_buffer 长度维持在 given_frames -1上
            state_buffer = list_pose[0:-1]  # 只考虑3帧啊啊啊啊，问题考虑全部帧会怎么样
            cf = self.confounders_module(torch.stack(
                state_buffer, dim=1))  # B Tau K cf_size

            input_embed = list_pose[-1]
            delta, hidden_cd = self.dynamic_module(
                input_embed, hidden_cd)  # 只输入当前状态

            pose_tilde = input_embed + delta  # 预测
            input_all = torch.cat([input_embed, cf], dim=-1)  # B K z_size*2
            K = torch.sigmoid(self.gate(input_all))
            pose = pose_tilde + K * (cf - pose_tilde)  # 校正

            # pose = input_embed + delta  # 直接相加

            list_predict.append(pose)

            if use_teacher_forcing:
                list_pose.append(temp[:, t+given_frames])
            else:  # 上一时刻的输出
                list_pose.append(pose)

        ab_hat = torch.stack(list_predict, dim=1)
        ab_hat = self.decoder(ab_hat)

        return ab_hat

    def prediction(self, given_state, horizon):
        """
        Functions: 前一时刻的预测输出,作为下一时刻的输入
        Input:
            given_state: [B given_frames K S] previous frame + current frames
            horizon: long-term prediction default=T-given_frames
        Output:
            a:[B horizon+given_frames K S] 逐时刻投影到原状态空间后拼接
        """

        hidden_cd = None
        # state_buffer = [ab[:, i] for i in range(given_frames-1)]
        temp = self.encoder(given_state)

        list_pose = [temp[:, i]
                     for i in range(temp.shape[1])]
        given_frames = temp.shape[1]

        for t in range(given_frames-1):
            if t == 0:  # 初始化 还是 None?? zero??
                state_buffer = [torch.zeros_like(list_pose[-1])]
                input_embed = temp[:, t]  # 取真值
            else:
                state_buffer = list_pose[0:t]  # 看一看能不能叠起来
                input_embed = list_pose[t]  # 教师指导

            cf = self.confounders_module(torch.stack(
                state_buffer, dim=1))  # B Tau K cf_size

            delta, hidden_cd = self.dynamic_module(
                input_embed, hidden_cd)  # 只输入当前状态

            pose_tilde = input_embed + delta  # 预测
            input_all = torch.cat([input_embed, cf], dim=-1)  # B K z_size*2
            K = torch.sigmoid(self.gate(input_all))
            pose = pose_tilde + K * (cf - pose_tilde)  # 校正
            # list_pose.append(pose)

        for t in range(horizon):  # 在低维空间的问题
            # B tau K z_size -> B K cf_size
            # state_buffer 长度维持在 given_frames -1上
            state_buffer = list_pose[0:-1]
            cf = self.confounders_module(torch.stack(state_buffer, dim=1))
            # B K z_size
            input_embed = list_pose[-1]
            # B K z_size
            delta, hidden_cd = self.dynamic_module(input_embed, hidden_cd)
            pose_tilde = input_embed + delta  # 预测

            # B K cf_size+z_size
            input_all = torch.cat([input_embed, cf], dim=-1)
            K = torch.sigmoid(self.gate(input_all))
            pose = pose_tilde + K * (cf - pose_tilde)

            # B K z_size
            list_pose.append(pose)
        # B horizon K z_size
        predict_state = torch.stack(list_pose, dim=1)
        predict_state = self.decoder(predict_state)

        return predict_state


class Inference_module_v5(nn.Module):  # 9/28 03:35
    """
    Params:
        state_size: [B T K S]  S: dimension of node state
        z_size: project dynamic system into higher dimensional state space
        cf_size: size of confounder, used in GN+GRU
        emb_size: size of node embedding, used in GN
        n_layers: num of hidden layers, used in GRU
        hidden_size: size of hidden state H, used in GRU
    Return:
        cd_hat: reconstruction of cd sequence

    """

    def __init__(self, state_size=14, cf_size=32, n_layers=2, hidden_size=32, emb_size=None, n_gn=1, z_size=64):
        super().__init__()

        if emb_size is None:
            emb_size = hidden_size

        # 希望是前t-1帧预测confounder,然后级联当前第t帧的表示,dynamic_module进行变化delta, s(t+1)=s(t)+delta
        self.confounders_module = ConfoundersModule(z_size, cf_size, n_layers=n_layers, hidden_size=hidden_size,
                                                    emb_size=emb_size)
        self.dynamic_module = DynamicModule(z_size, cf_size, n_layers=n_layers, hidden_size=hidden_size,
                                            emb_size=emb_size, n_gn=n_gn)  # n_layer GCN中MLP的隐藏层数目 n_gn:多几层n_gn
        # Input: B T K S output: B T K out_size
        self.encoder = Encoder(state_size, z_size)
        self.decoder = Decoder(state_size, z_size)

    def forward(self, ab, given_frames=2, horizon=148, p=1.0, rp=1.0):
        """
        teacher forcing
        Input:
            ab: input state sequence [B T K S]
            given_frames: given n frames to predict
            horizon: predict future horizon frames
            p: possibility of schedule sampling gt_value as input of next step
        Return: rec_ab [B horizon K S]
        """
        B, T, K, S = ab.shape
        hidden_cd = None

        assert T - given_frames >= horizon
        # B T K Z_size
        temp = self.encoder(ab)
        list_pose = [temp[:, i]
                     for i in range(given_frames)]
        list_predict = [temp[:, 0]]

        # teacher_forcing = np.random.binomial(1, p, horizon) #一个batchsize里是否取真值不同
        use_teacher_forcing = True if random.random() < p else False  # 一个batchsize里是否取真值相同
        encoder_use_teacher_forcing = True if random.random() < rp else False

        for t in range(given_frames - 1):

            if t == 0:  # 第一帧预测时
                state_buffer = [torch.zeros_like(temp[:, t])]  # B K D
                input_embed = temp[:, t]

            cf = self.confounders_module(torch.stack(state_buffer, dim=1))

            input_all = torch.cat([input_embed, cf], dim=-1)
            # B K d
            delta, hidden_cd = self.dynamic_module(input_all, hidden_cd)

            pose = input_embed + delta  # 直接相加 # 预测的结果
            list_predict.append(pose)

            if encoder_use_teacher_forcing:
                state_buffer.append(temp[:, t])
                input_embed = temp[:, t+1]
            else:
                state_buffer.append(input_embed)
                input_embed = pose

        for t in range(horizon):  # teacher forcing

            # state_buffer 长度维持在 given_frames -1上
            # 课程学习,state_buffer 里面存储的是真值还是上一步的输出
            state_buffer = list_pose[0:-1]
            cf = self.confounders_module(torch.stack(state_buffer, dim=1))
            input_embed = list_pose[-1]
            input_all = torch.cat([input_embed, cf], dim=-1)
            delta, hidden_cd = self.dynamic_module(input_all, hidden_cd)

            pose = input_embed + delta  # 直接相加
            list_predict.append(pose)

            # if teacher_forcing[t] == 1:  # 真值
            if use_teacher_forcing:
                list_pose.append(temp[:, t+given_frames])
            else:  # 上一时刻的输出
                list_pose.append(pose)

        ab_hat = torch.stack(list_predict, dim=1)
        ab_hat = self.decoder(ab_hat)

        return ab_hat

    def prediction(self, given_state, horizon):
        """
        Functions: 前一时刻的预测输出,作为下一时刻的输入
        Input:
            given_state: [B given_frames K S] previous frame + current frames
            horizon: long-term prediction default=T-given_frames
        Output:
            a:[B horizon+given_frames K S] 逐时刻投影到原状态空间后拼接
        """

        hidden_cd = None
        # state_buffer = [ab[:, i] for i in range(given_frames-1)]
        temp = self.encoder(given_state)

        list_pose = [temp[:, i]
                     for i in range(temp.shape[1])]
        given_frames = temp.shape[1]

        for t in range(given_frames - 1):

            if t == 0:  # 第一帧预测时
                state_buffer = [torch.zeros_like(temp[:, t])]
                input_embed = temp[:, t]

            cf = self.confounders_module(torch.stack(state_buffer, dim=1))
            input_all = torch.cat([input_embed, cf], dim=-1)
            delta, hidden_cd = self.dynamic_module(input_all, hidden_cd)
            pose = input_embed + delta  # 直接相加 # 预测的结果

            state_buffer.append(temp[:, t])
            input_embed = temp[:, t+1]

        for t in range(horizon):  # teacher forcing

            # state_buffer 长度维持在 given_frames -1上
            # 课程学习,state_buffer 里面存储的是真值还是上一步的输出
            state_buffer = list_pose[0:-1]
            cf = self.confounders_module(torch.stack(state_buffer, dim=1))
            input_embed = list_pose[-1]
            input_all = torch.cat([input_embed, cf], dim=-1)
            delta, hidden_cd = self.dynamic_module(input_all, hidden_cd)

            pose = input_embed + delta  # 直接相加

            list_pose.append(pose)
        # B horizon K z_size
        predict_state = torch.stack(list_pose, dim=1)
        predict_state = self.decoder(predict_state)

        return predict_state


class Inference_module_v6(nn.Module):  # 9/28 03:35
    """
    Params:
        state_size: [B T K S]  S: dimension of node state
        z_size: project dynamic system into higher dimensional state space
        cf_size: size of confounder, used in GN+GRU
        emb_size: size of node embedding, used in GN
        n_layers: num of hidden layers, used in GRU
        hidden_size: size of hidden state H, used in GRU
    Return:
        cd_hat: reconstruction of cd sequence

    """

    def __init__(self, state_size=14, cf_size=32, n_layers=2, hidden_size=32, emb_size=None, n_gn=1, z_size=64):
        super().__init__()

        if emb_size is None:
            emb_size = hidden_size

        # 希望是前t-1帧预测confounder,然后级联当前第t帧的表示,dynamic_module进行变化delta, s(t+1)=s(t)+delta
        self.confounders_module = ConfoundersModule(z_size, cf_size, n_layers=n_layers, hidden_size=hidden_size,
                                                    emb_size=emb_size)
        self.dynamic_module = DynamicModule(z_size, cf_size, n_layers=n_layers, hidden_size=hidden_size,
                                            emb_size=emb_size, n_gn=n_gn)  # n_layer GCN中MLP的隐藏层数目 n_gn:多几层n_gn
        # Input: B T K S output: B T K out_size
        self.encoder = Encoder(state_size, z_size)
        self.decoder = Decoder(state_size, z_size)

    def forward(self, ab, given_frames=2, horizon=148, p=1.0, rp=1.0):
        """
        teacher forcing
        Input:
            ab: input state sequence [B T K S]
            given_frames: given n frames to predict
            horizon: predict future horizon frames
            p: possibility of schedule sampling gt_value as input of next step
        Return: rec_ab [B horizon K S]
        """
        B, T, K, S = ab.shape
        hidden_cd = None

        assert T - given_frames >= horizon
        # B T K Z_size
        temp = self.encoder(ab)
        list_pose = [temp[:, i]
                     for i in range(given_frames)]
        list_predict = [temp[:, 0]]

        use_teacher_forcing = True if random.random() < p else False  # 一个batchsize里是否取真值相同
        encoder_use_teacher_forcing = True if random.random() < rp else False

        for t in range(given_frames - 1):

            if t == 0:  # 第一帧预测时
                input_embed = temp[:, t]
            # B K d
            delta, hidden_cd = self.dynamic_module(input_embed, hidden_cd)

            pose = input_embed + delta  # 直接相加 # 预测的结果
            list_predict.append(pose)

            if encoder_use_teacher_forcing:
                input_embed = temp[:, t+1]
            else:
                input_embed = pose

        for t in range(horizon):  # teacher forcing

            input_embed = list_pose[-1]
            delta, hidden_cd = self.dynamic_module(input_embed, hidden_cd)
            pose = input_embed + delta  # 直接相加
            list_predict.append(pose)

            # if teacher_forcing[t] == 1:  # 真值
            if use_teacher_forcing:
                list_pose.append(temp[:, t+given_frames])
            else:  # 上一时刻的输出
                list_pose.append(pose)

        ab_hat = torch.stack(list_predict, dim=1)
        ab_hat = self.decoder(ab_hat)

        return ab_hat

    def prediction(self, given_state, horizon):
        """
        Functions: 前一时刻的预测输出,作为下一时刻的输入
        Input:
            given_state: [B given_frames K S] previous frame + current frames
            horizon: long-term prediction default=T-given_frames
        Output:
            a:[B horizon+given_frames K S] 逐时刻投影到原状态空间后拼接
        """

        hidden_cd = None
        # state_buffer = [ab[:, i] for i in range(given_frames-1)]
        temp = self.encoder(given_state)

        list_pose = [temp[:, i]
                     for i in range(temp.shape[1])]
        given_frames = temp.shape[1]

        for t in range(given_frames - 1):

            input_embed = temp[:, t]
            delta, hidden_cd = self.dynamic_module(input_embed, hidden_cd)
            pose = input_embed + delta  # 直接相加 # 预测的结果

        for t in range(horizon):  # teacher forcing

            input_embed = list_pose[-1]
            delta, hidden_cd = self.dynamic_module(input_embed, hidden_cd)

            pose = input_embed + delta  # 直接相加

            list_pose.append(pose)
        # B horizon K z_size
        predict_state = torch.stack(list_pose, dim=1)
        predict_state = self.decoder(predict_state)

        return predict_state


class Inference_module_v7(nn.Module):  # 10/6 06:42
    """
    Params:
        state_size: [B T K S]  S: dimension of node state
        z_size: project dynamic system into higher dimensional state space
        cf_size: size of confounder, used in GN+GRU
        emb_size: size of node embedding, used in GN
        n_layers: num of hidden layers, used in GRU
        hidden_size: size of hidden state H, used in GRU
    Return:
        cd_hat: reconstruction of cd sequence

    """

    def __init__(self, state_size=14, cf_size=32, n_layers=2, hidden_size=32, emb_size=None, n_gn=1, z_size=64):
        super().__init__()

        if emb_size is None:
            emb_size = hidden_size

        # 希望是前t-1帧预测confounder,然后级联当前第t帧的表示,dynamic_module进行变化delta, s(t+1)=s(t)+delta
        self.confounders_module = ConfoundersModule(z_size, cf_size, n_layers=n_layers, hidden_size=hidden_size,
                                                    emb_size=emb_size)
        self.dynamic_module = DynamicModule(z_size, cf_size, n_layers=n_layers, hidden_size=hidden_size,
                                            emb_size=emb_size, n_gn=n_gn)  # n_layer GCN中MLP的隐藏层数目 n_gn:多几层n_gn
        # Input: B T K S output: B T K out_size
        self.encoder = Encoder(state_size, z_size)
        self.decoder = Decoder(state_size, z_size)

    def forward(self, ab, given_frames=2, horizon=148, p=1.0, rp=1.0):
        """
        teacher forcing
        Input:
            ab: input state sequence [B T K S]
            given_frames: given n frames to predict
            horizon: predict future horizon frames
            p: possibility of schedule sampling gt_value as input of next step
        Return: rec_ab [B horizon K S]
        """
        B, T, K, S = ab.shape
        hidden_cd = None

        assert T - given_frames >= horizon
        # B T K Z_size
        temp = self.encoder(ab)
        list_pose = [temp[:, i]
                     for i in range(given_frames)]
        list_predict = [temp[:, i]
                        for i in range(given_frames)]
        # list_pose = [self.encoder(ab[:, i]) for i in range(
        #     given_frames)]  # 初始化 state buffer + current frames

        teacher_forcing = np.random.binomial(
            1, p, horizon)  # 一个batchsize里是否取真值不同
        # use_teacher_forcing = True if random.random() < p else False  # 一个batchsize里是否取真值相同
        for t in range(horizon):  # teacher forcing

            # state_buffer 长度维持在 given_frames -1上
            state_buffer = list_pose[0:-1]  # 所有历史帧
            cf = self.confounders_module(torch.stack(state_buffer, dim=1))
            input_embed = list_pose[-1]
            input_all = torch.cat([input_embed, cf], dim=-1)
            delta, hidden_cd = self.dynamic_module(input_all, hidden_cd)

            pose = input_embed + delta  # 直接相加
            # pose = delta  # 直接预测
            # skip connection 复杂化上一帧状态的映射& 动态变化 与
            # pose = self.project(torch.cat([input_embed, delta], dim=-1))
            # pose = self.linear(torch.cat([input_embed, delta], dim=-1))

            list_predict.append(pose)

            if teacher_forcing[t] == 1:  # 真值
                # if use_teacher_forcing:
                list_pose.append(temp[:, t+given_frames])
            else:  # 上一时刻的输出
                list_pose.append(pose)

        ab_hat = torch.stack(list_predict, dim=1)
        ab_hat = self.decoder(ab_hat)

        return ab_hat

    def prediction(self, given_state, horizon):
        """
        Functions: 前一时刻的预测输出,作为下一时刻的输入
        Input:
            given_state: [B given_frames K S] previous frame + current frames
            horizon: long-term prediction default=T-given_frames
        Output:
            a:[B horizon+given_frames K S] 逐时刻投影到原状态空间后拼接
        """

        hidden_cd = None
        # state_buffer = [ab[:, i] for i in range(given_frames-1)]
        temp = self.encoder(given_state)

        list_pose = [temp[:, i]
                     for i in range(temp.shape[1])]

        for t in range(horizon):  # 在低维空间的问题
            # B tau K z_size -> B K cf_size
            # state_buffer 长度维持在 given_frames -1上
            state_buffer = list_pose[0:-1]
            cf = self.confounders_module(torch.stack(state_buffer, dim=1))
            # B K z_size
            current_state = list_pose[-1]
            # B K cf_size+z_size
            input_all = torch.cat([current_state, cf], dim=-1)
            # B K z_size
            delta, hidden_cd = self.dynamic_module(input_all, hidden_cd)
            # B K z_size
            predict_pose = delta + current_state
            list_pose.append(predict_pose)
        # B horizon K z_size
        predict_state = torch.stack(list_pose, dim=1)
        predict_state = self.decoder(predict_state)

        return predict_state


class Delta_Module(nn.Module):
    """
    delta_module: 基于上一时刻的状态预测改变的状态(速度,加速度,coef的改变量)
    state size : keypoint state dim
    target size : predict state dim
    hidden size: high-dimensional state space
    emb_size : MLP-hidden layer hidden_size
    n_layers: num of hidden layers in MLP block
    n_gn: num of message-passing steps
    """

    def __init__(self, state_size, target_size=None, emb_size=128, hidden_size=128, n_layers=2, n_gn=4):
        super(Delta_Module, self).__init__()

        if target_size == None:
            target_size = state_size
        self.encoder = GCN(in_size=state_size, out_size=hidden_size,
                           n_layers=n_layers, hidden_size=emb_size)
        self.decoder = GCN(in_size=hidden_size, out_size=target_size,
                           n_layers=n_layers, hidden_size=emb_size)
        gns = []
        for _ in range(n_gn):
            gns.append(GCN(in_size=hidden_size, out_size=hidden_size,
                       hidden_size=emb_size, n_layers=2))
        self.predictor = nn.ModuleList(gns)  # 没有循环单元

    def forward(self, inpt):

        B, K, S = inpt.shape
        embed_inpt = self.encoder(inpt)
        embed_outpt = embed_inpt
        for i in range(len(self.predictor)):  # 4步消息传递矩阵
            embed_outpt = self.predictor[i](embed_outpt) + embed_outpt
        outpt = self.decoder(embed_outpt)
        return outpt


class ResIN(nn.Module):
    def __init__(self, in_size, out_size, n_layers, hidden_size, activation=nn.ReLU, batchnorm=False):
        super(ResIN, self).__init__()
        self.IN = GCN(in_size, out_size, n_layers, hidden_size, activation)
        self.norm = nn.LayerNorm(out_size)

    def forward(self, x):
        return self.norm(self.IN(x)+x)





class Delta_Module_with_LN(nn.Module):
    """
    delta_module: 基于上一时刻的状态预测改变的状态(速度,加速度,coef的改变量)
    state size : keypoint state dim
    target size : predict state dim
    hidden size: high-dimensional state space
    emb_size : MLP-hidden layer hidden_size
    n_layers: num of hidden layers in MLP block
    n_gn: num of message-passing steps
    """

    def __init__(self, state_size, target_size=None, emb_size=128, hidden_size=128, n_layers=2, n_gn=4):
        super(Delta_Module_with_LN, self).__init__()

        if target_size == None:
            target_size = state_size
        # self.encoder = MLP(in_size=state_size, h_size=emb_size,
        #                    out_size=hidden_size, n_layers=n_layers)
        # self.decoder = MLP(in_size=hidden_size, h_size=emb_size,
        #                    out_size=target_size, n_layers=n_layers)
        self.encoder = GCN(in_size=state_size, out_size=hidden_size,
                           n_layers=n_layers, hidden_size=emb_size)
        self.decoder = GCN(in_size=hidden_size, out_size=target_size,
                           n_layers=n_layers, hidden_size=emb_size)
        gns = []
        for _ in range(n_gn):
            gns.append(ResIN(in_size=hidden_size, out_size=hidden_size,
                       hidden_size=emb_size, n_layers=2))
        self.predictor = nn.ModuleList(gns)  # 没有循环单元

    def forward(self, inpt):

        B, K, S = inpt.shape
        embed_inpt = self.encoder(inpt)
        _, _, hidden_size = embed_inpt.shape
        norm_layer = nn.LayerNorm(hidden_size).to(inpt.device)
        embed_outpt = norm_layer(embed_inpt)
        for i in range(len(self.predictor)):  # 4步消息传递矩阵
            embed_outpt = self.predictor[i](embed_outpt)
        outpt = self.decoder(embed_outpt)
        return outpt


class Delta_Module_no_res(nn.Module):
    """
    没有跨层连接
    delta_module: 基于上一时刻的状态预测改变的状态(速度,加速度,coef的改变量)
    state size : keypoint state dim
    target size : predict state dim
    hidden size: high-dimensional state space
    emb_size : MLP-hidden layer hidden_size
    n_layers: num of hidden layers in MLP block
    n_gn: num of message-passing steps
    """

    def __init__(self, state_size, target_size=None, emb_size=128, hidden_size=128, n_layers=2, n_gn=4):
        super(Delta_Module_no_res, self).__init__()

        if target_size == None:
            target_size = state_size
        self.encoder = GCN(in_size=state_size, out_size=hidden_size,
                           n_layers=n_layers, hidden_size=emb_size)
        self.decoder = GCN(in_size=hidden_size, out_size=target_size,
                           n_layers=n_layers, hidden_size=emb_size)
        gns = []
        for _ in range(n_gn):
            gns.append(GCN(in_size=hidden_size, out_size=hidden_size,
                       hidden_size=emb_size, n_layers=2))
        self.predictor = nn.ModuleList(gns)  # 没有循环单元

    def forward(self, inpt):

        B, K, S = inpt.shape
        embed_inpt = self.encoder(inpt)
        embed_outpt = embed_inpt
        for i in range(len(self.predictor)):  # 4步消息传递矩阵
            embed_outpt = self.predictor[i](embed_outpt)
        outpt = self.decoder(embed_outpt)
        return outpt


class Delta_Module_no_autocoder(nn.Module):
    """
    没有跨层连接
    delta_module: 基于上一时刻的状态预测改变的状态(速度,加速度,coef的改变量)
    state size : keypoint state dim
    target size : predict state dim
    hidden size: high-dimensional state space
    emb_size : MLP-hidden layer hidden_size
    n_layers: num of hidden layers in MLP block
    n_gn: num of message-passing steps
    """

    def __init__(self, state_size, target_size=None, emb_size=128, hidden_size=128, n_layers=2, n_gn=4):
        super(Delta_Module_no_autocoder, self).__init__()

        if target_size == None:
            target_size = state_size

        gns = []
        gns.append(GCN(in_size=state_size, out_size=hidden_size,
                       n_layers=n_layers, hidden_size=emb_size))
        for _ in range(n_gn-2):
            gns.append(GCN(in_size=hidden_size, out_size=hidden_size,
                       hidden_size=emb_size, n_layers=2))
        gns.append(GCN(in_size=hidden_size, out_size=target_size,
                       n_layers=n_layers, hidden_size=emb_size))
        self.predictor = nn.ModuleList(gns)  # 没有循环单元

    def forward(self, inpt):

        B, K, S = inpt.shape
        embed_outpt = inpt

        for i in range(len(self.predictor)):  # 4步消息传递矩阵
            embed_outpt = self.predictor[i](embed_outpt)

        return embed_outpt


class Delta_Module_fc(nn.Module):
    """
    将图神经网络用全连接层代替
    delta_module: 基于上一时刻的状态预测改变的状态(速度,加速度,coef的改变量)
    state size : keypoint state dim
    target size : predict state dim
    hidden size: high-dimensional state space
    emb_size : MLP-hidden layer hidden_size
    n_layers: num of hidden layers in MLP block
    n_gn: num of message-passing steps
    """

    def __init__(self, state_size, target_size=None, emb_size=128, hidden_size=128, n_layers=2, n_gn=4):
        super(Delta_Module_fc, self).__init__()

        if target_size == None:
            target_size = state_size
        self.encoder = MLP(in_size=state_size, h_size=emb_size,
                           out_size=hidden_size, n_layers=n_layers)
        self.decoder = MLP(in_size=hidden_size, h_size=emb_size,
                           out_size=target_size, n_layers=n_layers)
        fcs = []
        for _ in range(n_gn):
            fcs.append(MLP(in_size=hidden_size, out_size=hidden_size,
                       h_size=emb_size, n_layers=n_layers))
        self.predictor = nn.ModuleList(fcs)

    def forward(self, inpt):

        B, K, S = inpt.shape
        embed_inpt = self.encoder(inpt)
        embed_outpt = embed_inpt
        for i in range(len(self.predictor)):  # 4步消息传递矩阵
            #embed_outpt = self.predictor[i](embed_outpt) + embed_outpt
            embed_outpt = self.predictor[i](embed_outpt)
        outpt = self.decoder(embed_outpt)
        return outpt


class Delta_Module_multiscale(nn.Module):
    """
    delta_module: 基于上一时刻的状态预测改变的状态(速度,加速度,coef的改变量)
    state size : keypoint state dim
    target size : predict state dim
    hidden size: high-dimensional state space
    emb_size : MLP-hidden layer hidden_size
    n_layers: num of hidden layers in MLP block
    n_gn: num of message-passing steps
    """

    def __init__(self, state_size, target_size=None, emb_size=128, hidden_size=128, n_layers=2, n_gn=4):
        super(Delta_Module_multiscale, self).__init__()

        if target_size == None:
            target_size = state_size
        self.encoder = GCN(in_size=state_size, out_size=hidden_size,
                           n_layers=n_layers, hidden_size=emb_size)
        self.decoder = GCN(in_size=hidden_size*3, out_size=target_size,
                           n_layers=n_layers, hidden_size=emb_size)
        gns_path1 = []
        gns_path2 = []
        gns_path3 = []
        for _ in range(n_gn):
            gns_path1.append(GCN(in_size=hidden_size, out_size=hidden_size,
                                 hidden_size=emb_size, n_layers=2))
        for _ in range(n_gn//2):
            gns_path2.append(GCN(in_size=hidden_size, out_size=hidden_size,
                                 hidden_size=emb_size, n_layers=2))
        for _ in range(1):
            gns_path3.append(GCN(in_size=hidden_size, out_size=hidden_size,
                                 hidden_size=emb_size, n_layers=2))
        self.predictor1 = nn.ModuleList(gns_path1)  # 没有循环单元
        self.predictor2 = nn.ModuleList(gns_path2)
        self.predictor3 = nn.ModuleList(gns_path3)

    def forward(self, inpt):

        #B, K, S = inpt.shape
        embed_inpt = self.encoder(inpt)
        embed_outpt1 = embed_inpt
        embed_outpt2 = embed_inpt
        embed_outpt3 = embed_inpt
        for i in range(len(self.predictor1)):  # 4步消息传递矩阵
            embed_outpt1 = self.predictor1[i](embed_outpt1) + embed_outpt1
        for i in range(len(self.predictor2)):  # 4步消息传递矩阵
            embed_outpt2 = self.predictor2[i](embed_outpt2) + embed_outpt2
        for i in range(len(self.predictor3)):  # 4步消息传递矩阵
            embed_outpt3 = self.predictor3[i](embed_outpt3) + embed_outpt3
        embed_outpt = torch.cat(
            [embed_outpt1, embed_outpt2, embed_outpt3], dim=-1)
        outpt = self.decoder(embed_outpt)
        return outpt


class Delta_Module_v1(nn.Module):
    """
    预测模块添加了一个GRU,使得encoder的东西有意义
    delta_module: 基于上一时刻的状态预测改变的状态(速度,加速度,coef的改变量)
    state size : keypoint state dim
    target size : predict state dim
    hidden size: high-dimensional state space
    emb_size : MLP-hidden layer hidden_size
    n_layers: num of hidden layers in MLP block
    n_gn: num of message-passing steps
    """

    def __init__(self, state_size, target_size=None, emb_size=128, hidden_size=128, n_layers=2, n_gn=4):
        super(Delta_Module_v1, self).__init__()

        if target_size == None:
            target_size = state_size
        self.encoder = GCN(in_size=state_size, out_size=hidden_size,
                           n_layers=n_layers, hidden_size=emb_size)
        self.decoder = GCN(in_size=hidden_size, out_size=target_size,
                           n_layers=n_layers, hidden_size=emb_size)
        gns = []
        for _ in range(n_gn):
            gns.append(GCN(in_size=hidden_size, out_size=hidden_size,
                       hidden_size=emb_size, n_layers=2))
        self.predictor = nn.ModuleList(gns)  # 没有循环单元
        self.rnn_layer = nn.GRU(hidden_size, hidden_size,
                                num_layers=2, batch_first=True)

    def forward(self, inpt, hidden_state):

        B, K, S = inpt.shape
        embed_inpt = self.encoder(inpt)
        embed_outpt = embed_inpt
        for i in range(len(self.predictor)):  # 4步消息传递矩阵
            embed_outpt = self.predictor[i](embed_outpt) + embed_outpt
        embed_outpt = embed_outpt.view(B * K, 1, -1)  # B*K 1 S
        if hidden_state is None:
            outpt, hidden_state = self.rnn_layer(embed_outpt)
        else:
            outpt, hidden_state = self.rnn_layer(embed_outpt, hidden_state)

        outpt = outpt.view(B, K, -1)
        delta = self.decoder(outpt)

        return delta, hidden_state


class Delta_Module_transformer(nn.Module):
    """
    全部用transformer代替
    delta_module: 基于上一时刻的状态预测改变的状态(速度,加速度,coef的改变量)
    state size : keypoint state dim
    target size : predict state dim
    hidden size: high-dimensional state space
    emb_size : MLP-hidden layer hidden_size
    n_layers: num of hidden layers in MLP block
    n_gn: num of message-passing steps
    """

    def __init__(self, state_size, target_size=None, emb_size=128, hidden_size=128, n_layers=2, n_gn=4):
        super(Delta_Module_transformer, self).__init__()
        if target_size == None:
            target_size = state_size
        self.encoder = nn.Linear(
            in_features=state_size, out_features=hidden_size, bias=False)
        self.decoder = nn.Linear(
            in_features=hidden_size, out_features=target_size, bias=False)
        # self.decoder = GCN(in_size=hidden_size, out_size=target_size,
        #                    n_layers=n_layers, hidden_size=emb_size)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size, nhead=4, dim_feedforward=hidden_size*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_gn)

    def forward(self, inpt):
        embed_inpt = self.encoder(inpt)
        embed_outpt = self.transformer_encoder(embed_inpt)
        outpt = self.decoder(embed_outpt)

        return outpt


class my_inference_transformer(nn.Module):
    def __init__(self, state_size=11, n_layers=2, emb_size=128, n_gn=1, z_size=128):
        super().__init__()
        self.delta_module = Delta_Module_transformer(
            state_size=state_size, emb_size=emb_size, hidden_size=z_size, n_layers=n_layers, n_gn=n_gn)

    def forward(self, ab, given_frames=3, horizon=100, p=0.0, rp=1.0, n_vel=2):
        B, T, K, S = ab.shape  # B T-2 K S+4 输入的状态序列包括了两个时刻的速度 horizon+given_frams -2
        input_state = ab[:, 0:given_frames-n_vel]
        target_state = ab[:, given_frames-n_vel:]

        predict_states = []
        delta_states = []
        for i in range(input_state.shape[1]-1):
            delta_state = self.delta_module(ab[:, i])
            delta_states.append(delta_state)
            next_state = delta_state + ab[:, i]  # B K S
            #next_state = delta_state
            predict_states.append(next_state)

        decoder_input = input_state[:, -1]
        use_teacher_forcing = np.random.binomial(1, p, horizon)
        for t in range(target_state.shape[1]):  # Horizon 时刻
            delta_state = self.delta_module(decoder_input)
            delta_states.append(delta_state)
            next_state = delta_state + decoder_input
            #next_state = delta_state
            predict_states.append(next_state)
            if use_teacher_forcing[t]:
                decoder_input = target_state[:, t]
            else:
                decoder_input = next_state

        predict_state = torch.stack(predict_states, dim=1)
        delta_hat = torch.stack(delta_states, dim=1)

        return predict_state, delta_hat


class my_inference(nn.Module):  # 10/6 06:42
    """
    delta_module: 基于上一时刻的状态预测改变的状态(加速度,coef的改变量)
    Params:
        state_size: [B T-2 K S]  S: dimension of node state
        z_size: project dynamic system into higher dimensional state space
        emb_size: size of node embedding, used in GN
        n_layers: num of hidden layers
        hidden_size: size of hidden state H
    Return:
        cd_hat: reconstruction of cd sequence  [B T-3 K S]

    """

    def __init__(self, state_size=11, n_layers=2, emb_size=128, n_gn=1, z_size=128):
        super().__init__()

        # if emb_size is None:
        #     emb_size = hidden_size

        # 希望是前t-1帧预测confounder,然后级联当前第t帧的表示,dynamic_module进行变化delta, s(t+1)=s(t)+delta
        self.delta_module = Delta_Module(
            state_size=state_size, emb_size=emb_size, hidden_size=z_size, n_layers=n_layers, n_gn=n_gn)
        # self.delta_module = Delta_Module_with_LN(
        #     state_size=state_size, emb_size=emb_size, hidden_size=z_size, n_layers=n_layers, n_gn=n_gn)
        # self.delta_module = Delta_Module_no_res(
        #     state_size=state_size, emb_size=emb_size, hidden_size=z_size, n_layers=n_layers, n_gn=n_gn)
        # self.delta_module = Delta_Module_no_autocoder(
        #     state_size=state_size, emb_size=emb_size, hidden_size=z_size, n_layers=n_layers, n_gn=n_gn)
        # self.delta_module = Delta_Module_fc(
        #     state_size=state_size, emb_size=emb_size, hidden_size=z_size, n_layers=n_layers, n_gn=n_gn)

    def forward(self, ab, given_frames=3, horizon=100, p=0.0, rp=1.0, n_vel=2):
        """

        teacher forcing
        Input:
            ab: input state sequence [B T K S] S=11, T = horizon+given_frams -2
            given_frames: given n frames to predict
            horizon: predict future horizon frames
            p: possibility of schedule sampling gt_value as input of next step
        Return: rec_ab [B horizon K S]
        """
        B, T, K, S = ab.shape  # B T-2 K S+4 输入的状态序列包括了两个时刻的速度 horizon+given_frams -2
        input_state = ab[:, 0:given_frames-n_vel]
        target_state = ab[:, given_frames-n_vel:]

        predict_states = []
        delta_states = []
        #use_teacher_forcing = True if random.random() < p else False

        for i in range(input_state.shape[1]-1):
            delta_state = self.delta_module(ab[:, i])
            delta_states.append(delta_state)
            next_state = delta_state + ab[:, i]  # B K S
            predict_states.append(next_state)

        decoder_input = input_state[:, -1]
        use_teacher_forcing = np.random.binomial(1, p, horizon)
        for t in range(target_state.shape[1]):  # Horizon 时刻
            delta_state = self.delta_module(decoder_input)
            delta_states.append(delta_state)
            next_state = delta_state + decoder_input
            predict_states.append(next_state)
            if use_teacher_forcing[t]:
                decoder_input = target_state[:, t]
            else:
                decoder_input = next_state

        predict_state = torch.stack(predict_states, dim=1)
        delta_hat = torch.stack(delta_states, dim=1)

        return predict_state, delta_hat

    def prediction(self, given_state, horizon):
        """
        Functions: 前一时刻的预测输出,作为下一时刻的输入
        Input:
            given_state: [B given_frames-2 K S] previous frame + current frames
            horizon: long-term prediction default=T-given_frames
        Output:
            a:[B horizon+given_frames-3 K S] 逐时刻投影到原状态空间后拼接
        """
        given_states_len = given_state.shape[1]  # B Tau K S
        predict_states = []

        # for i in range(given_states_len-1):
        #     next_state = self.delta_module(
        #         given_state[:, i]) + given_state[:, i]   # B K S
        #     # predict_states.append(next_state)

        decoder_input = given_state[:, -1]
        for t in range(horizon):  # Horizon 时刻 只考虑预测的horizon的重建误差
            next_state = self.delta_module(
                decoder_input) + decoder_input  # valid的时候没用到这个函数?
            predict_states.append(next_state)
            decoder_input = next_state

        predict_state = torch.stack(predict_states, dim=1)

        return predict_state


class my_inference_multi_scale(nn.Module):  # 10/6 06:42
    """
    delta_module: 基于上一时刻的状态预测改变的状态(加速度,coef的改变量)
    Params:
        state_size: [B T-2 K S]  S: dimension of node state
        z_size: project dynamic system into higher dimensional state space
        emb_size: size of node embedding, used in GN
        n_layers: num of hidden layers
        hidden_size: size of hidden state H
    Return:
        cd_hat: reconstruction of cd sequence  [B T-3 K S]

    """

    def __init__(self, state_size=11, n_layers=2, emb_size=128, n_gn=1, z_size=128):
        super().__init__()

        # if emb_size is None:
        #     emb_size = hidden_size

        # 希望是前t-1帧预测confounder,然后级联当前第t帧的表示,dynamic_module进行变化delta, s(t+1)=s(t)+delta
        self.delta_module = Delta_Module_multiscale(
            state_size=state_size, emb_size=emb_size, hidden_size=z_size, n_layers=n_layers, n_gn=n_gn)

    def forward(self, ab, given_frames=3, horizon=100, p=1.0, rp=1.0, n_vel=2):
        """

        teacher forcing
        Input:
            ab: input state sequence [B T K S] S=11, T = horizon+given_frams -2
            given_frames: given n frames to predict
            horizon: predict future horizon frames
            p: possibility of schedule sampling gt_value as input of next step
        Return: rec_ab [B horizon K S]
        """
        B, T, K, S = ab.shape  # B T-2 K S+4 输入的状态序列包括了两个时刻的速度 horizon+given_frams -2
        input_state = ab[:, 0:given_frames-n_vel]
        target_state = ab[:, given_frames-n_vel:]

        predict_states = []
        delta_states = []
        #use_teacher_forcing = True if random.random() < p else False

        for i in range(input_state.shape[1]-1):
            delta_state = self.delta_module(ab[:, i])
            delta_states.append(delta_state)
            next_state = delta_state + ab[:, i]  # B K S
            predict_states.append(next_state)

        decoder_input = input_state[:, -1]
        use_teacher_forcing = np.random.binomial(1, p, horizon)
        for t in range(target_state.shape[1]):  # Horizon 时刻
            delta_state = self.delta_module(decoder_input)
            delta_states.append(delta_state)
            next_state = delta_state + decoder_input
            predict_states.append(next_state)
            if use_teacher_forcing[t]:
                decoder_input = target_state[:, t]
            else:
                decoder_input = next_state

        predict_state = torch.stack(predict_states, dim=1)
        delta_hat = torch.stack(delta_states, dim=1)

        return predict_state, delta_hat

    def prediction(self, given_state, horizon):
        """
        Functions: 前一时刻的预测输出,作为下一时刻的输入
        Input:
            given_state: [B given_frames-2 K S] previous frame + current frames
            horizon: long-term prediction default=T-given_frames
        Output:
            a:[B horizon+given_frames-3 K S] 逐时刻投影到原状态空间后拼接
        """
        given_states_len = given_state.shape[1]  # B Tau K S
        predict_states = []

        for i in range(given_states_len-1):
            next_state = self.delta_module(
                given_state[:, i]) + given_state[:, i]   # B K S
            # predict_states.append(next_state)

        decoder_input = given_state[:, -1]
        for t in range(horizon):  # Horizon 时刻 只考虑预测的horizon的重建误差
            next_state = self.delta_module(
                decoder_input) + decoder_input  # valid的时候没用到这个函数?
            predict_states.append(next_state)
            decoder_input = next_state

        predict_state = torch.stack(predict_states, dim=1)

        return predict_state


class my_inference_with_GRU(nn.Module):  # 10/6 06:42
    """
    delta_module: 基于上一时刻的状态预测改变的状态(加速度,coef的改变量)
    Params:
        state_size: [B T-2 K S]  S: dimension of node state
        z_size: project dynamic system into higher dimensional state space
        emb_size: size of node embedding, used in GN
        n_layers: num of hidden layers
        hidden_size: size of hidden state H
    Return:
        cd_hat: reconstruction of cd sequence  [B T-3 K S]

    """

    def __init__(self, state_size=11, n_layers=2, emb_size=128, n_gn=1, z_size=128):
        super().__init__()

        # if emb_size is None:
        #     emb_size = hidden_size

        # 希望是前t-1帧预测confounder,然后级联当前第t帧的表示,dynamic_module进行变化delta, s(t+1)=s(t)+delta
        self.delta_module = Delta_Module_v1(
            state_size=state_size, emb_size=emb_size, hidden_size=z_size, n_layers=n_layers, n_gn=n_gn)

    def forward(self, ab, given_frames=3, horizon=100, p=1.0, rp=1.0, n_vel=2):
        """

        teacher forcing
        Input:
            ab: input state sequence [B T K S] S=11, T = horizon+given_frams -2
            given_frames: given n frames to predict
            horizon: predict future horizon frames
            p: possibility of schedule sampling gt_value as input of next step
        Return: rec_ab [B horizon K S]
        """
        B, T, K, S = ab.shape  # B T-2 K S+4 输入的状态序列包括了两个时刻的速度 horizon+given_frams -2
        input_state = ab[:, 0:given_frames-n_vel]
        target_state = ab[:, given_frames-n_vel:]

        predict_states = []
        delta_states = []
        #use_teacher_forcing = True if random.random() < p else False

        hidden_state = None

        for i in range(input_state.shape[1]-1):
            delta_state, hidden_state = self.delta_module(
                ab[:, i], hidden_state)
            delta_states.append(delta_state)
            next_state = delta_state + ab[:, i]  # B K S
            predict_states.append(next_state)

        decoder_input = input_state[:, -1]
        use_teacher_forcing = np.random.binomial(1, p, horizon)
        for t in range(target_state.shape[1]):  # Horizon 时刻
            delta_state, hidden_state = self.delta_module(
                decoder_input, hidden_state)
            delta_states.append(delta_state)
            next_state = delta_state + decoder_input
            predict_states.append(next_state)
            if use_teacher_forcing[t]:
                decoder_input = target_state[:, t]
            else:
                decoder_input = next_state

        predict_state = torch.stack(predict_states, dim=1)
        delta_hat = torch.stack(delta_states, dim=1)

        return predict_state, delta_hat

    def prediction(self, given_state, horizon):
        """
        Functions: 前一时刻的预测输出,作为下一时刻的输入
        Input:
            given_state: [B given_frames-2 K S] previous frame + current frames
            horizon: long-term prediction default=T-given_frames
        Output:
            a:[B horizon+given_frames-3 K S] 逐时刻投影到原状态空间后拼接
        """
        given_states_len = given_state.shape[1]  # B Tau K S
        predict_states = []

        for i in range(given_states_len-1):
            next_state = self.delta_module(
                given_state[:, i]) + given_state[:, i]   # B K S
            # predict_states.append(next_state)

        decoder_input = given_state[:, -1]
        for t in range(horizon):  # Horizon 时刻 只考虑预测的horizon的重建误差
            next_state = self.delta_module(
                decoder_input) + decoder_input  # valid的时候没用到这个函数?
            predict_states.append(next_state)
            decoder_input = next_state

        predict_state = torch.stack(predict_states, dim=1)

        return predict_state


class my_inference_v1(nn.Module):  # 10/8 06:42
    """
    delta_module: 基于上一时刻的状态预测改变的状态(加速度,coef的改变量)
    Params:
        state_size: [B T-2 K S]  S: dimension of node state
        z_size: project dynamic system into higher dimensional state space
        emb_size: size of node embedding, used in GN
        n_layers: num of hidden layers
        hidden_size: size of hidden state H
    Return:
        cd_hat: reconstruction of cd sequence  [B T-3 K S]

    """

    def __init__(self, state_size=11, n_layers=2, emb_size=128, n_gn=1, z_size=128):
        super().__init__()

        # 希望是前t-1帧预测confounder,然后级联当前第t帧的表示,dynamic_module进行变化delta, s(t+1)=s(t)+delta
        self.delta_module = Delta_Module(
            state_size=state_size, target_size=7, emb_size=emb_size, hidden_size=z_size, n_layers=n_layers, n_gn=n_gn)

    def forward(self, ab, given_frames=3, horizon=100, p=1.0, rp=1.0, n_vel=2):
        """
        teacher forcing
        Input:
            ab: input state sequence [B T K S] S=11, T = horizon+given_frams -2
            given_frames: given n frames to predict
            horizon: predict future horizon frames
            p: possibility of schedule sampling gt_value as input of next step
        Return: rec_ab [B horizon K S]
        """
        B, T, K, S = ab.shape  # B T-2 K S+4 输入的状态序列包括了两个时刻的速度 horizon+given_frams -2
        input_state = ab[:, 0:given_frames-n_vel]
        target_state = ab[:, given_frames-n_vel:]

        predict_states = []
        delta_states = []
        #use_teacher_forcing = True if random.random() < p else False
        next_state = torch.zeros_like(ab[:, 0]).to(ab.device)  # B K S

        for i in range(input_state.shape[1]-1):
            delta_state = self.delta_module(ab[:, i])  # B K S-4 加速度
            delta_states.append(delta_state)
            next_state[..., -5:] = ab[:, i][..., -5:] + \
                delta_state[..., -5:]  # 下一时刻的系数
            # coef 控制在0-1
            # next_state[..., -5:] = torch.sigmoid(next_state[..., -5:])
            next_state[..., 4:6] = ab[:, i][..., 2:4]  # 下一时刻的上一时刻的速度
            next_state[..., 2:4] = delta_state[..., 0:2]  # 下一时刻的速度 直接预测速度
            next_state[..., 0:2] = ab[:, i][..., 0:2] + \
                next_state[..., 2:4]  # 下一时刻的位置
            predict_states.append(next_state)

        decoder_input = input_state[:, -1]
        use_teacher_forcing = np.random.binomial(1, p, horizon)
        for t in range(target_state.shape[1]):  # Horizon 时刻

            delta_state = self.delta_module(decoder_input)  # 加速度和系数的变化量
            delta_states.append(delta_state)

            next_state[..., -5:] = decoder_input[..., -5:] + \
                delta_state[..., -5:]  # 下一时刻的系数
            next_state[..., 4:6] = decoder_input[..., 2:4]  # 下一时刻的上一时刻的速度

            # next_state[..., 2:4] = decoder_input[..., 2:4] + \
            #     delta_state[..., 0:2]  # 下一时刻的速度
            next_state[..., 2:4] = delta_state[..., 0:2]  # 下一时刻的速度 直接预测速度

            next_state[..., 0:2] = decoder_input[..., 0:2] + \
                next_state[..., 2:4]  # 下一时刻的位置
            predict_states.append(next_state)

            if use_teacher_forcing[t]:
                decoder_input = target_state[:, t]
            else:
                decoder_input = next_state

        delta_hat = torch.stack(delta_states, dim=1)  # B H S=7 (a, delta_c)
        predict_state = torch.stack(predict_states, dim=1)

        return predict_state, delta_hat

    def prediction(self, given_state, horizon):
        """
        Functions: 前一时刻的预测输出,作为下一时刻的输入
        Input:
            given_state: [B given_frames-2 K S] previous frame + current frames
            horizon: long-term prediction default=T-given_frames
        Output:
            a:[B horizon+given_frames-3 K S] 逐时刻投影到原状态空间后拼接
        """
        given_states_len = given_state.shape[1]  # B Tau K S
        predict_states = []
        next_state = torch.zeros_like(given_state[:, 0]).to(
            given_state.device)  # B K S

        for i in range(given_states_len-1):

            next_state = self.delta_module(given_state[:, i])  # B K S
            delta_state = self.delta_module(given_state[:, i])  # 加速度和系数的变化量
            next_state[..., -5:] = given_state[:, i][..., -5:] + \
                delta_state[..., -5:]  # 下一时刻的系数
            next_state[..., 4:6] = given_state[:, i][..., 2:4]  # 下一时刻的上一时刻的速度

            # next_state[..., 2:4] = given_state[:, i][..., 2:4] + \
            #     delta_state[..., 0:2]  # 下一时刻的速度  如何更新 ---> 预测加速度
            next_state[..., 2:4] = delta_state[..., 0:2]  # 直接预测速度
            next_state[..., 0:2] = given_state[:, i][..., 0:2] + \
                next_state[..., 2:4]  # 下一时刻的位置
            predict_states.append(next_state)

        decoder_input = given_state[:, -1]
        for t in range(horizon):  # Horizon 时刻

            delta_state = self.delta_module(decoder_input)  # 加速度和系数的变化量
            next_state[..., -5:] = decoder_input[..., -5:] + \
                delta_state[..., -5:]  # 下一时刻的系数
            next_state[..., 4:6] = decoder_input[..., 2:4]  # 下一时刻的上一时刻的速度

            # next_state[..., 2:4] = decoder_input[..., 2:4] + \
            #     delta_state[..., 0:2]  # 下一时刻的速度 直接预测加速度

            next_state[..., 2:4] = delta_state[..., 0:2]  # 下一时刻的速度 直接预测速度

            next_state[..., 0:2] = decoder_input[..., 0:2] + \
                next_state[..., 2:4]  # 下一时刻的位置

            predict_states.append(next_state)
            decoder_input = next_state

        predict_state = torch.stack(predict_states, dim=1)

        return predict_state


class my_inference_with_cf_module(nn.Module):
    def __init__(self, state_size=11, n_layers=2, emb_size=128, n_gn=1, z_size=128, cf_size=8):
        super().__init__()

        # 希望是前t-1帧预测confounder,然后级联当前第t帧的表示,dynamic_module进行变化delta, s(t+1)=s(t)+delta
        self.confounders_module = ConfoundersModule(state_size, cf_size, n_layers=n_layers, hidden_size=emb_size,
                                                    emb_size=emb_size)
        self.delta_module = Delta_Module(
            state_size=state_size + cf_size, target_size=state_size, emb_size=emb_size, hidden_size=z_size, n_layers=n_layers, n_gn=n_gn)

    def forward(self, input_states,  horizon=100):
        """
        Input:
            input_states: B T=given_frames-num_vel K=4 S=7+2*num_vel
            horizon: predicted lengtg
        Output:
            pred_states: B hirozn K S=7+2*num_vel
        """

        # tau-1
        state_buffer = input_states
        cur_state = input_states[:, -1]  # B K S
        list_pred = []
        for t in range(horizon):
            cf = self.confounders_module(state_buffer)  # B K cf
            delta_state = self.delta_module(
                torch.cat([cur_state, cf], dim=-1))  # B K S+cf_size -> B K S
            pred_state = delta_state + cur_state
            list_pred.append(pred_state)
            state_buffer = torch.cat(
                [state_buffer, pred_state.unsqueeze(1)], dim=1)
            cur_state = pred_state

        pred_states = torch.stack(list_pred, dim=1)  # B horizon K S

        return pred_states


class my_inference_with_cf_module_v1(nn.Module):
    def __init__(self, state_size=11, n_layers=2, emb_size=128, n_gn=1, z_size=128, cf_size=8):
        super().__init__()

        # 希望是前t-1帧预测confounder,然后级联当前第t帧的表示,dynamic_module进行变化delta, s(t+1)=s(t)+delta
        self.confounders_module = ConfoundersModule(state_size, cf_size, n_layers=n_layers, hidden_size=emb_size,
                                                    emb_size=emb_size)
        self.delta_module = Delta_Module(
            state_size=state_size + cf_size, target_size=state_size, emb_size=emb_size, hidden_size=z_size, n_layers=n_layers, n_gn=n_gn)

    def forward(self, ab, given_frames=3,  horizon=100, p=1.0, n_vel=2):
        input_states = ab[:, 0:given_frames-n_vel]
        target_states = ab[:, given_frames-n_vel:]
        state_buffer = input_states
        predict_states = []

        prior_clue = self.confounders_module(state_buffer)
        for i in range(input_states.shape[1]-1):
            delta_state = self.delta_module(
                torch.cat([input_states[:, i], prior_clue], dim=-1))
            predict_state = delta_state + input_states[:, i]


class my_inference_with_const_cf(nn.Module):  # 10/6 06:42
    """
    delta_module: 基于上一时刻的状态预测改变的状态(加速度,coef的改变量)
    Params:
        state_size: [B T-2 K S]  S: dimension of node state
        z_size: project dynamic system into higher dimensional state space
        emb_size: size of node embedding, used in GN
        n_layers: num of hidden layers
        hidden_size: size of hidden state H
    Return:
        cd_hat: reconstruction of cd sequence  [B T-3 K S]

    """

    def __init__(self, state_size=11, n_layers=2, emb_size=128, n_gn=1, z_size=128, cf_size=8):
        super().__init__()

        # if emb_size is None:
        #     emb_size = hidden_size

        # 希望是前t-1帧预测confounder,然后级联当前第t帧的表示,dynamic_module进行变化delta, s(t+1)=s(t)+delta
        self.confounders_module = ConfoundersModule(state_size, cf_size, n_layers=n_layers, hidden_size=emb_size,
                                                    emb_size=emb_size)
        self.delta_module = Delta_Module(
            state_size=state_size+cf_size, target_size=state_size, emb_size=emb_size, hidden_size=z_size, n_layers=n_layers, n_gn=n_gn)

    def forward(self, ab, given_frames=3, horizon=100, p=1.0, rp=1.0, n_vel=2):
        """

        teacher forcing
        Input:
            ab: input state sequence [B T-n_vel K S] S=11, T = horizon+given_frams -2
            given_frames: given n frames to predict
            horizon: predict future horizon frames
            p: possibility of schedule sampling gt_value as input of next step
        Return: rec_ab [B horizon K S]
        """
        B, T, K, S = ab.shape  # B T-2 K S+4 输入的状态序列包括了两个时刻的速度 horizon+given_frams -2
        input_state = ab[:, 0:given_frames-n_vel]
        target_state = ab[:, given_frames-n_vel:]

        predict_states = []
        delta_states = []
        #use_teacher_forcing = True if random.random() < p else False

        prior_clue = self.confounders_module(input_state)

        for i in range(input_state.shape[1]-1):
            delta_state = self.delta_module(
                torch.cat([input_state[:, i], prior_clue], dim=-1))
            delta_states.append(delta_state)
            next_state = delta_state + input_state[:, i]  # B K S
            predict_states.append(next_state)

        decoder_input = input_state[:, -1]
        use_teacher_forcing = np.random.binomial(1, p, horizon)
        for t in range(target_state.shape[1]):  # Horizon 时刻
            delta_state = self.delta_module(
                torch.cat([decoder_input, prior_clue], dim=-1))
            delta_states.append(delta_state)
            next_state = delta_state + decoder_input
            predict_states.append(next_state)
            if use_teacher_forcing[t]:
                decoder_input = target_state[:, t]
            else:
                decoder_input = next_state

        predict_state = torch.stack(predict_states, dim=1)
        delta_hat = torch.stack(delta_states, dim=1)

        return predict_state, delta_hat

    def prediction(self, given_state, horizon):
        """
        Functions: 前一时刻的预测输出,作为下一时刻的输入
        Input:
            given_state: [B given_frames-2 K S] previous frame + current frames
            horizon: long-term prediction default=T-given_frames
        Output:
            a:[B horizon K S] 逐时刻投影到原状态空间后拼接
        """
        # given_states_len = given_state.shape[1]  # B Tau K S
        predict_states = []
        prior_clue = self.confounders_module(given_state)

        # for i in range(given_states_len-1):
        #     next_state = self.delta_module(
        #         torch.cat([given_state[:, i], prior_clue], dim=-1)) + given_state[:, i]   # B K S
        #     # predict_states.append(next_state)

        decoder_input = given_state[:, -1]
        for t in range(horizon):  # Horizon 时刻 只考虑预测的horizon的重建误差
            next_state = self.delta_module(
                torch.cat([decoder_input, prior_clue], dim=-1)) + decoder_input  # valid的时候没用到这个函数?
            predict_states.append(next_state)
            decoder_input = next_state

        predict_state = torch.stack(predict_states, dim=1)

        return predict_state
