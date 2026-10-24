import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def CalDelta_xy(keypoints_np, img_w):
    # print('use delta_xy')
    keypoints_np1 = np.zeros_like(keypoints_np)
    # print(cfg.img_w) # 112 or 64 根据设置而变化的
    keypoints_np1[:, :, 0] = np.round(
        (1 + keypoints_np[:, :, 0]) * (img_w - 1) / 2
    )  # 最靠近的像素位�?
    keypoints_np1[:, :, 1] = np.round(
        (1 - keypoints_np[:, :, 1]) * (img_w - 1) / 2
    )  # 最靠近的像素位�?
    keypoints_np1[:, :, 0] = keypoints_np1[:, :, 0] / (img_w - 1) * 2 - 1
    keypoints_np1[:, :, 1] = 1 - keypoints_np1[:, :, 1] / (img_w - 1) * 2

    delta_xy = keypoints_np1 - keypoints_np  # 距离
    return delta_xy


class Derendering(nn.Module):
    def __init__(
        self,
        n_keypoints=4,
        n_coefficients=4,
        mode="fixed",
        std=0.1,
        use_scale=False,
        device=torch.device("cpu"),
        img_width=112,
        img_height=None,
        img_channel=3,
        filter_size=5,
        num_filter=32,
        use_pixelshuffle=False,
        kpt_margin=0.0,
        shared_object_decoder=False,
    ):
        super(Derendering, self).__init__()
        assert mode in ["fixed", "learned"]

        self.encoder = Encoder(
            k=n_keypoints, img_channel=img_channel, num_filter=num_filter
        )
        self.decoder = Decoder(
            k=n_keypoints,
            n_param=n_coefficients,
            img_channel=img_channel,
            num_filter=num_filter,
            use_pixelshuffle=use_pixelshuffle,
            shared_objects=shared_object_decoder,
        )
        # 提取coeficient C+1 dim
        # self.extractor = nn.Sequential(
        #     nn.Linear(img_width * img_width // 16, 2048), nn.ReLU(),  # 图像尺寸而变�?
        #     nn.Linear(2048, 1024), nn.ReLU(),
        #     nn.Linear(1024, 512), nn.ReLU(),
        #     nn.Linear(512, n_coefficients + 1)
        # )
        if img_height == None:
            img_height = img_width
        self.extractor = nn.Sequential(
            nn.Linear(img_width * img_height // 16, num_filter * 64),
            nn.ReLU(),  # 图像尺寸而变�?
            nn.Linear(num_filter * 64, num_filter * 32),
            nn.ReLU(),
            nn.Linear(num_filter * 32, num_filter * 16),
            nn.ReLU(),
            nn.Linear(num_filter * 16, n_coefficients + 1),
        )

        self.mode = mode  # filter bank mode=fixed  or learned
        self.std = std
        # Keypoint coordinate range is [-1-kpt_margin, 1+kpt_margin]; >0 lets
        # keypoints sit slightly outside the frame so half-entered objects
        # keep a correct centroid (the gaussian canvas stays [-1,1] and is
        # simply truncated). 0.0 = historical behavior, old checkpoints OK.
        self.kpt_margin = kpt_margin
        self.k = n_keypoints
        self.n_coefficients = n_coefficients
        self.filter_size = filter_size
        self.shared_object_decoder = shared_object_decoder
        # When n_coefficients==0, extractor still outputs 1 dim (scale/exist).
        # use_scale=True multiplies the gaussian by that dim (pos+scale).
        # use_scale=False is vanilla KeypointNet (position only).
        self.use_scale = use_scale or n_coefficients > 0

        if n_coefficients > 0:
            if mode == "fixed":
                self.bank = get_dilatation_filters(
                    filter_size, filter_size, n_coefficients
                )
                self.bank = self.bank.to(device).view(
                    1, 1, self.n_coefficients, filter_size, filter_size
                )
                # (1,K,C,filter_W,filter_H)
                self.bank = self.bank.repeat(1, self.k, 1, 1, 1)
            elif mode == "learned":
                self.bank = nn.Parameter(
                    torch.randn(
                        1, 1, n_coefficients, filter_size, filter_size
                    ).repeat(1, self.k, 1, 1, 1)
                    / 5,
                    requires_grad=True,
                )

    def forward(self, source, target, is_grid=False):
        out = {}
        B, C, H, W = source.shape
        # source target
        features, heatmap, coef = self.encoder(torch.cat([source, target], dim=0))

        source_features = features[:B]
        target_heatmap = heatmap[B:]
        target_coef = coef[B:]  # [B,K,28,28]

        target_parameter = self.extractor(
            torch.flatten(target_coef, start_dim=2)
        )  # [B,K,28*28] -># [B,K,C+1]
        target_parameter = torch.sigmoid(target_parameter)  # c_i 范围[0,1]

        # [B,K,28,28] -> [B,K,2]
        target_keypoints_location = self.get_keypoint_location(target_heatmap)

        if is_grid:
            keypoints_np = target_keypoints_location.data.cpu().numpy()
            delta_xy = CalDelta_xy(keypoints_np)
            delta_xypt = torch.FloatTensor(delta_xy).to(source.device)
            target_keypoints_location = delta_xypt

        target = self.create_img(
            source_features, target_keypoints_location, target_parameter
        )

        out["source_features"] = source_features
        out["target_keypoints"] = target_keypoints_location
        out["target_parameter"] = target_parameter
        # 重构的目标图�?
        out["target_heatmap"] = target_heatmap
        out["target"] = target  # [B 3 112 112 ]

        return out

    def get_keypoint_location(self, keypoints_features):
        """
        Functions : get keypoint location from K heatmaps
        paper: Unsupervised Learning of Object Landmarks through Conditional Image Generation
        Input: K heatmaps shape = [B K 28 28]
        Return: 2D coordinate for K keypoints  shape = [B K 2]
        """
        # renormalised to a probability distribution via (spatial) Softmax
        # B K 每个heatmap上所有值之�?=1 B K H W
        normalized_keypoints = spatial_softmax(keypoints_features)
        S_row = normalized_keypoints.sum(-1)  # N, K, H
        S_col = normalized_keypoints.sum(-2)  # N, K, W

        m = self.kpt_margin
        u_row = S_row.mul(
            torch.linspace(-1 - m, 1 + m, S_row.size(-1), device=normalized_keypoints.device)
        ).sum(
            -1
        )  # ？有点奇�?
        u_col = S_col.mul(
            torch.linspace(-1 - m, 1 + m, S_col.size(-1), device=normalized_keypoints.device)
        ).sum(-1)
        return torch.stack((u_row, u_col), -1)

    def create_img(self, features, keypoints_locations, parameters):
        B, C, W, H = features.shape
        """
        Functions : fusing the source dense feature map and the target keypoints and coefficients to recon target img
        Return: target_img shape=[B 3 112 112]
        """
        # each heatmap is replaced with a Gaussian-like function centred keypoints_locations
        # B K 28 28
        target_keypoints = gaussian_map(keypoints_locations, W, H, std=self.std)

        if self.n_coefficients != 0:  # n_coefficients = 0 即原版的keypointNet
            # (B, K, C, filter_W, filter_H)
            bank = self.bank.repeat(B, 1, 1, 1, 1)

            # 四个系数加权的~~~ 四个旋转卷积�?
            weighted_bank = bank * parameters[:, :, :-1].view(
                B, self.k, self.n_coefficients, 1, 1
            )  # [B K C=4 H W]
            weighted_bank = (
                parameters[:, :, -1].view(B, self.k, 1, 1, 1) * weighted_bank
            )
            weighted_bank = weighted_bank.view(
                B * self.k * self.n_coefficients, 1, self.filter_size, self.filter_size
            )  # [B*K*C 1 fs fs] [output_channel, input_channel, H,W]

            target_keypoints = target_keypoints.unsqueeze(2).repeat(
                1, 1, self.n_coefficients, 1, 1
            )
            target_keypoints = target_keypoints.view(
                1, B * self.k * self.n_coefficients, W, H
            )  # [1 B*K*C 28 28]

            # minibatch = 1 out_channel=in_channel=groups=B*K*C
            keypoints_map = F.conv2d(
                target_keypoints,
                weighted_bank,
                padding=self.filter_size // 2,
                groups=B * self.k * self.n_coefficients,
            )  # [1 B*K*C 28 28]
            # B K*C 28 28
            keypoints_map = keypoints_map.view(B, -1, W, H)
            if self.shared_object_decoder:
                keypoints_map = keypoints_map.view(
                    B, self.k, self.n_coefficients, W, H).sum(dim=1)

        else:
            keypoints_map = target_keypoints  # B K 28 28
            if self.use_scale and parameters is not None and parameters.shape[-1] > 0:
                keypoints_map = keypoints_map * parameters[:, :, -1].view(
                    B, self.k, 1, 1
                )

        # B K*C+16 28 28
        inpt = torch.cat([features, keypoints_map], dim=1)
        target_images = self.decoder(inpt)

        return target_images

    def get_keypoints_map(self, features, keypoints_locations, parameters):
        """
        Functions : fusing the source dense feature map and the target keypoints and coefficients to recon target img
        Return: target_img shape=[B 3 112 112]
        """
        # each heatmap is replaced with a Gaussian-like function centred keypoints_locations
        B, C, W, H = features.shape

        # B K 28 28 没有除以最大值啊啊啊
        target_keypoints = gaussian_map(keypoints_locations, W, H, std=self.std)

        if self.n_coefficients != 0:  # n_coefficients = 0 即原版的keypointNet
            # (B, K, C, filter_W, filter_H)
            bank = self.bank.repeat(B, 1, 1, 1, 1)

            weighted_bank = bank * parameters[:, :, :-1].view(
                B, self.k, self.n_coefficients, 1, 1
            )  # [B K C H W]
            weighted_bank = (
                parameters[:, :, -1].view(B, self.k, 1, 1, 1) * weighted_bank
            )
            weighted_bank = weighted_bank.view(
                B * self.k * self.n_coefficients, 1, self.filter_size, self.filter_size
            )  # [B*K*C 1 fs fs] [output_channel, input_channel, H,W]

            target_keypoints = target_keypoints.unsqueeze(2).repeat(
                1, 1, self.n_coefficients, 1, 1
            )
            target_keypoints = target_keypoints.view(
                1, B * self.k * self.n_coefficients, W, H
            )  # [1 B*K*C 28 28]

            # minibatch = 1 out_channel=in_channel=groups=B*K*C
            keypoints_map = F.conv2d(
                target_keypoints,
                weighted_bank,
                padding=self.filter_size // 2,
                groups=B * self.k * self.n_coefficients,
            )  # [1 B*K*C 28 28]
            # B K*C 28 28
            keypoints_map = keypoints_map.view(B, -1, W, H)
            if self.shared_object_decoder:
                keypoints_map = keypoints_map.view(
                    B, self.k, self.n_coefficients, W, H).sum(dim=1)
        else:
            keypoints_map = target_keypoints
            if self.use_scale and parameters is not None and parameters.shape[-1] > 0:
                keypoints_map = keypoints_map * parameters[:, :, -1].view(
                    B, self.k, 1, 1
                )
        return keypoints_map  # B K*C  H W  B 4*4 28 28

    # 确定不是class encoder 的函数吗
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
        keypoints = F.softplus(self.encoder.regressor(x_pose))  # [B, K, 28, 28]
        target_parameter = self.extractor(torch.flatten(x_coef, start_dim=2))
        target_parameter = torch.sigmoid(target_parameter)  # [B, K, C]
        target_keypoints_location = self.get_keypoint_location(keypoints)  # [B, K, 2]
        return target_keypoints_location, target_parameter

    def get_keypoints_all(self, x):
        target_keypoints_location, target_parameter = self.get_keypoints_only(x)
        return torch.cat([target_keypoints_location, target_parameter], dim=-1)


class Block(nn.Module):
    """
    Functions: conv+BN+Relu
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1
    ):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        # 改成false有用吗啊�?
        self.bn = nn.BatchNorm2d(out_channels)
        # self.bn = nn.BatchNorm2d(out_channels, track_running_stats=True,momentum=0.2)
        # self.bn = nn.BatchNorm2d(out_channels, track_running_stats=False)

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

    def __init__(self, k, img_channel, num_filter=32):
        super(Encoder, self).__init__()
        # self.net = nn.Sequential(
        #     Block(img_channel, 32, kernel_size=(7, 7), stride=1,
        #           padding=3),  # H,W 保持不变  通道数根�? 数据而变�? 3->1
        #     Block(32, 32, kernel_size=(3, 3), stride=1),  # H=112 padding=1
        #     Block(32, 64, kernel_size=(3, 3), stride=2),  # H=56
        #     Block(64, 64, kernel_size=(3, 3), stride=1),  # H=56
        #     Block(64, 128, kernel_size=(3, 3), stride=2)  # H=28
        # )
        # self.net_feat = Block(128, 16, kernel_size=(3, 3), stride=1)  # 28 �?

        # self.net_pose = Block(128, 128, kernel_size=(3, 3), stride=1)  # 28

        # self.net_coef = Block(128, k, kernel_size=(3, 3), stride=1)  # 28

        # self.regressor = nn.Conv2d(
        #     128, k, kernel_size=(1, 1))  # 28 Gaussian mapping??
        self.net = nn.Sequential(
            Block(
                img_channel, num_filter, kernel_size=(7, 7), stride=1, padding=3
            ),  # H,W 保持不变  通道数根�? 数据而变�? 3->1
            Block(
                num_filter, num_filter, kernel_size=(3, 3), stride=1
            ),  # H=112 padding=1
            Block(num_filter, num_filter * 2, kernel_size=(3, 3), stride=2),  # H=56
            Block(num_filter * 2, num_filter * 2, kernel_size=(3, 3), stride=1),  # H=56
            Block(num_filter * 2, num_filter * 4, kernel_size=(3, 3), stride=2),  # H=28
        )
        self.net_feat = Block(num_filter * 4, 16, kernel_size=(3, 3), stride=1)  # 28 �?

        self.net_pose = Block(
            num_filter * 4, num_filter * 4, kernel_size=(3, 3), stride=1
        )  # 28

        self.net_coef = Block(num_filter * 4, k, kernel_size=(3, 3), stride=1)  # 28

        self.regressor = nn.Conv2d(
            num_filter * 4, k, kernel_size=(1, 1)
        )  # 28 Gaussian mapping??

    def forward(self, x):
        x = self.net(x)  # [B, 128, 28, 28]

        x_feat = self.net_feat(x)  # [B, 16, 28, 28]
        x_pose = self.net_pose(x)  # [B, 128, 28, 28]
        x_coef = self.net_coef(x)  # [B, k, 28, 28]

        pose = F.softplus(self.regressor(x_pose))  # [B, K, 28, 28]
        return x_feat, pose, x_coef


class Decoder(nn.Module):
    def __init__(self, k, n_param, img_channel, num_filter=32,
                 use_pixelshuffle=False, shared_objects=False):
        super(Decoder, self).__init__()
        if n_param == 0:  # 是否考虑了额外的coef
            i = k
        else:
            i = n_param if shared_objects else k * n_param
        # self.net = nn.Sequential(
        #     Block(16 + i, 128, kernel_size=(3, 3), stride=1),  # 6
        #     Block(128, 128, kernel_size=(3, 3), stride=1),
        #     Block(128, 64, kernel_size=(3, 3), stride=1),  # 5
        #     nn.UpsamplingBilinear2d(scale_factor=2),  # 双线性上采样 56*56
        #     Block(64, 64, kernel_size=(3, 3), stride=1),  # 4
        #     Block(64, 32, kernel_size=(3, 3), stride=1),  # 3
        #     nn.UpsamplingBilinear2d(scale_factor=2, ),  # 双线性上采样 112*112
        #     Block(32, 32, kernel_size=(3, 3), stride=1),  # 2
        #     Block(32, 32, kernel_size=(7, 7), stride=1, padding=3),  # 1
        #     nn.Conv2d(32, img_channel, 1)  # 通道�? 3->1
        # )
        if use_pixelshuffle:
            # Learned sub-pixel upsampling instead of fixed bilinear. Keeps the
            # same spatial ladder (H/4 -> H/2 -> H) and per-level channel widths,
            # but the network learns its own upsampling kernels. Changes
            # decoder.net indices 3 and 6 -> old checkpoints are incompatible.
            self.net = nn.Sequential(
                Block(16 + i, num_filter * 4, kernel_size=(3, 3), stride=1),  # 6
                Block(num_filter * 4, num_filter * 4, kernel_size=(3, 3), stride=1),
                Block(num_filter * 4, num_filter * 2, kernel_size=(3, 3), stride=1),  # 5
                Block(num_filter * 2, num_filter * 8, kernel_size=(3, 3), stride=1),  # 4
                nn.PixelShuffle(2),  # nf*8 -> nf*2, H/4 -> H/2
                Block(num_filter * 2, num_filter, kernel_size=(3, 3), stride=1),  # 3
                Block(num_filter, num_filter * 4, kernel_size=(3, 3), stride=1),  # 2
                nn.PixelShuffle(2),  # nf*4 -> nf, H/2 -> H
                Block(num_filter, num_filter, kernel_size=(7, 7), stride=1, padding=3),  # 1
                nn.Conv2d(num_filter, img_channel, 1),  # 通道数 3->1
            )
        else:
            self.net = nn.Sequential(
                Block(16 + i, num_filter * 4, kernel_size=(3, 3), stride=1),  # 6
                Block(num_filter * 4, num_filter * 4, kernel_size=(3, 3), stride=1),
                Block(num_filter * 4, num_filter * 2, kernel_size=(3, 3), stride=1),  # 5
                nn.UpsamplingBilinear2d(scale_factor=2),  # 双线性上采样 56*56
                Block(num_filter * 2, num_filter * 2, kernel_size=(3, 3), stride=1),  # 4
                Block(num_filter * 2, num_filter, kernel_size=(3, 3), stride=1),  # 3
                nn.UpsamplingBilinear2d(
                    scale_factor=2,
                ),  # 双线性上采样 112*112
                Block(num_filter, num_filter, kernel_size=(3, 3), stride=1),  # 2
                Block(num_filter, num_filter, kernel_size=(7, 7), stride=1, padding=3),  # 1
                nn.Conv2d(num_filter, img_channel, 1),  # 通道�? 3->1
            )

    def forward(self, x):
        return torch.tanh(self.net(x))


def spatial_softmax(features):
    # [B,K]+(-1.)=[B, K, -1] --> [B,K,28*28]
    features_reshape = features.reshape(features.shape[:-2] + (-1,))
    output = F.softmax(features_reshape, dim=-1)
    output = output.reshape(features.shape)
    return output


def gaussian_map(mu, width, height, std):
    # features: (N, K, 2)
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]
    y = torch.linspace(-1.0, 1.0, height, device=mu.device)
    x = torch.linspace(-1.0, 1.0, width, device=mu.device)
    # [B K 1 1]
    mu_y, mu_x = mu_y.unsqueeze(-1), mu_x.unsqueeze(-1)
    # [1 1 H 1] [1 1 1 W]
    y = y.reshape(1, 1, height, 1)
    x = x.reshape(1, 1, 1, width)
    # 广播机制！broadcast
    dist = ((y - mu_y) ** 2 + (x - mu_x) ** 2) / std**2
    g_yx = torch.exp(-dist)

    return g_yx


def get_dilatation_filters(W, H, n_params):
    """
    Functions: get fixed filter bank

    theta = [0,..,(n-1)/n]*pi

    Z[k, i, j] =(y == x*tanh(theta))

    Return:
        z.shape (n_params, W, H)
    """
    X = torch.linspace(-5, 5, W)
    Y = torch.linspace(-5, 5, H)
    Z = torch.zeros(n_params, W, H)
    # 系数的维度决定了filter的倾斜角度 0~pi的等�?
    # çłťć°ççť´ĺşŚĺłĺŽäşfilterçĺžćč§ĺşŚ 0~piçç­ĺ?
    for k, theta in enumerate(torch.arange(0, n_params) * (np.pi) / n_params):
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                x0 = (x + y * torch.tan(theta)) / (1 + torch.tan(theta) ** 2)
                y0 = x0 * torch.tan(theta)

                Z[k, i, j] = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2))
    return Z


# z = get_dilatation_filters(W=5, H=5, n_params=4)
# print(z)


# z = torch.zeros((2, 4, 28, 28))
# output = get_keypoint_location(z)
