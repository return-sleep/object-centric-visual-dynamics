import json
import os
import imageio
import numpy as np
import torch
import torch.nn as nn
import cv2
import skimage

import matplotlib.pyplot as plt
import scipy
import re
import urllib
import requests
import html
import hashlib

from PIL import Image, ImageDraw

try:
    plt.style.use("seaborn-v0_8")
except OSError:
    pass

MSE = torch.nn.MSELoss()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def linear_anneal(epoch, start_value, final_value, start_epoch, final_epoch):
    """ """
    if epoch < start_epoch:
        value = start_value
    elif epoch < final_epoch:
        value = start_value + (epoch - start_epoch) / (final_epoch - start_epoch) * (
            final_value - start_value
        )
    else:
        value = final_value
    return value


def get_boundary_kernel_new(kernel_size=32, boundary_width=6):
    """
    Will return something like this:
    ============
    =          =
    =          =
    ============
    size will be (kernel_size, kernel_size)
    """
    filter = torch.zeros(kernel_size, kernel_size)
    filter[:, :] = 1.0 / (kernel_size**2)
    # Set center to zero
    filter[
        boundary_width : kernel_size - boundary_width,
        boundary_width : kernel_size - boundary_width,
    ] = 0.0

    return filter


def get_loss(cd, cd_hat, autoencode):
    """
    Functions: compute loss
    Input: cd/cd_hat/autoencode [T,K, dim(s)] dim(s)=7=2(kpts)+5(coef)
    Return:
        [loss, loss_kpts, loss_coef, loss_state, loss_encoding]
    """
    MSE = nn.MSELoss()
    loss_kpts = MSE(cd[..., :2], cd_hat[..., :2])  # cd [T K S=kpts+coef]
    # loss_coef = MSE(cd[..., 2:7], cd_hat[..., 2:7])
    # loss_state = MSE(cd[..., :7], cd_hat[..., :7])

    # 投影到高维状态隐空间 dynamics更加简�?? s(t) �?? dec(enc(s(t)))
    loss_encoding = MSE(cd, autoencode)
    loss_rec = MSE(cd, cd_hat)
    loss = loss_rec * 1e3 + loss_encoding * 1e3  # 公式 6
    return loss, loss_rec * 1e3, loss_encoding * 1e3, loss_kpts * 1e3


def get_loss_with_speed_exist(cd, cd_hat, pos_weight=1.0, df=1.0):
    """
    [loss_rec * 1e3, loss_kpts * 1e3, loss_velocity * 1e3, loss_coef * 1e3]
    可以用于计算无autoencoder和无速度的重建损失coef_loss 计算有问�??
    是否考虑位置的权重高一�??-----降低
    """
    MSE = nn.MSELoss()
    n_coef = cd.shape[-1] - 2  # 笑死，之前写错了�??1/28 改正
    loss_kpts = MSE(cd[..., :2], cd_hat[..., :2])
    loss_exist = MSE(cd[..., -1], cd_hat[..., -1])

    # loss_coef = MSE(cd[..., -5:], cd_hat[..., -5:]) # 超出索引是全部的�??
    # loss_velocity = MSE(cd[..., 2:-5], cd_hat[..., 2:-5])
    # 2/1 之后 coef只考虑旋转参数�?? 之前考虑旋转参数+尺寸参数
    loss_coef = MSE(cd[..., -n_coef:-1], cd_hat[..., -n_coef:-1])  # 超出索引是全部的�??
    loss_velocity = MSE(cd[..., 2:-n_coef], cd_hat[..., 2:-n_coef])

    # loss_rec = MSE(cd, cd_hat)
    if n_coef == 1:
        loss_rec = MSE(cd, cd_hat)
    else:
        loss_rotate = MSE(cd[..., 2:-1], cd_hat[..., 2:-1])
        loss_rec = (
            loss_kpts * 2 + loss_exist + loss_rotate * (n_coef - 1) * pos_weight
        ) / cd.shape[-1]

    return (
        loss_rec * 1e3,
        loss_kpts * 1e3,
        loss_velocity * 1e3,
        loss_coef * 1e3,
        loss_exist * 1e3,
    )


def get_loss_without_coef(cd, cd_hat, pos_weight=1.0, df=1.0):
    MSE = nn.MSELoss()
    S = cd.shape[-1]
    loss_kpts = MSE(cd[..., :2], cd_hat[..., :2])
    loss_velocity = MSE(cd[..., 2:], cd_hat[..., 2:])
    loss_exist = MSE(cd[..., S:], cd_hat[..., S:])
    loss_coef = MSE(cd[..., S:], cd_hat[..., S:])
    loss_rec = MSE(cd, cd_hat)
    return (
        loss_rec * 1e3,
        loss_kpts * 1e3,
        loss_velocity * 1e3,
        loss_coef * 1e3,
        loss_exist * 1e3,
    )


def get_loss_mse(cd, cd_hat, n_coef=5):
    MSE = nn.MSELoss()
    loss_rec = MSE(cd, cd_hat)
    loss_kpts = MSE(cd[..., :2], cd_hat[..., :2])
    loss_exist = MSE(cd[..., -1], cd_hat[..., -1])
    loss_coef = MSE(cd[..., -n_coef:-1], cd_hat[..., -n_coef:-1])  # 超出索引是全部的�??
    loss_velocity = MSE(cd[..., 2:-n_coef], cd_hat[..., 2:-n_coef])
    return (
        loss_rec * 1e3,
        loss_kpts * 1e3,
        loss_velocity * 1e3,
        loss_coef * 1e3,
        loss_exist * 1e3,
    )


def get_loss_with_df(
    cd,
    cd_hat,
    pos_weight=1.0,
    df=1.0,
    n_coef=5,
    xy_weight=None,
    velocity_weight=None,
    appearance_weight=1.0,
    scale_weight=1.0,
    temporal_late_weight=1.0,
):
    """
    [loss_rec * 1e3, loss_kpts * 1e3, loss_velocity * 1e3, loss_coef * 1e3]
    可以用于计算无autoencoder和无速度的重建损失coef_loss 计算有问�??
    是否考虑位置的权重高一�??-----降低
    是否考虑折扣因子 discount factor
    只考虑7d的状态表征的时候是没问题的!
    """
    MSE = nn.MSELoss()

    # loss = (cd - cd_hat)**2
    # n_coef = cd.shape[-1] - 2  # 笑死，之前写错了�??1/28 改正
    # n_coef = 5
    loss_kpts = MSE(cd[..., :2], cd_hat[..., :2])
    loss_exist = MSE(cd[..., -1], cd_hat[..., -1])
    loss_coef = MSE(cd[..., -n_coef:-1], cd_hat[..., -n_coef:-1])  # 超出索引是全部的�??
    loss_velocity = MSE(cd[..., 2:-n_coef], cd_hat[..., 2:-n_coef])

    pred_size = cd.shape[1]
    # B T K S -> b T S -> T S
    loss_per = ((cd - cd_hat) ** 2).mean(2).mean(0)  # T S
    # loss_t = loss_per[:, 0:2].sum(-1)+loss_per[:, -1]+loss_per[:, -n_coef:-
    #                                                            1].sum(-1)*pos_weight
    # 包括了速度呢， n_coef作为参数输入 写错了啊这，把权重加到系数上去了�??
    # loss_t = loss_per[:, 0:-n_coef].sum(-1)+loss_per[:, -1]+loss_per[:, -n_coef:-
    #                                                                  1].sum(-1)*pos_weight
    # 这样才合理吧，位置和速度的系数增加了www
    # Keep old behavior exactly when the new component weights are omitted.
    # Historically pos_weight multiplied both xy and all finite-difference
    # velocity dimensions; explicit weights let long-rollout experiments
    # emphasize position without accidentally amplifying every velocity lag.
    xy_w = pos_weight if xy_weight is None else xy_weight
    vel_w = pos_weight if velocity_weight is None else velocity_weight
    loss_t = (
        loss_per[:, :2].sum(-1) * xy_w
        + loss_per[:, 2:-n_coef].sum(-1) * vel_w
        + loss_per[:, -n_coef:-1].sum(-1) * appearance_weight
        + loss_per[:, -1] * scale_weight
    )
    loss_t = loss_t / (cd.shape[-1])  # T 自带权重
    # T 时间衰减? 最后没有用到的样子
    df_t = torch.pow(
        torch.as_tensor(df, device=cd.device, dtype=loss_t.dtype),
        torch.arange(pred_size, device=cd.device, dtype=loss_t.dtype),
    )
    if temporal_late_weight != 1.0:
        df_t = df_t * torch.linspace(
            1.0,
            temporal_late_weight,
            pred_size,
            device=cd.device,
            dtype=loss_t.dtype,
        )
    # 1
    loss_rec = ((loss_t * df_t) / df_t.sum()).sum()  # 麻了多写了一�??
    # loss_rec = MSE(cd, cd_hat)  #--- 等同于mse

    return (
        loss_rec * 1e3,
        loss_kpts * 1e3,
        loss_velocity * 1e3,
        loss_coef * 1e3,
        loss_exist * 1e3,
    )


def get_loss_with_speed(cd, cd_hat):
    """
    [loss_rec * 1e3, loss_kpts * 1e3, loss_velocity * 1e3, loss_coef * 1e3]
    可以用于计算无autoencoder和无速度的重建损�?? 针对Filter-cophy �?? 感知模块
    """
    MSE = nn.MSELoss()
    loss_kpts = MSE(cd[..., :2], cd_hat[..., :2])
    loss_coef = MSE(cd[..., -5:], cd_hat[..., -5:])
    loss_velocity = MSE(cd[..., 2:-5], cd_hat[..., 2:-5])
    loss_rec = MSE(cd, cd_hat)

    return loss_rec * 1e3, loss_kpts * 1e3, loss_velocity * 1e3, loss_coef * 1e3


def vis(state, predict_state):
    """
    可视化预测结果的location与真值的�??
    Input: [B T K S]
    """
    print("vis")
    return


def crop_feature_map(feature_map, pos, crop_size=5):
    """
    Input:
        feature_map: B, K, C, H, W , B C H W  �?? B K H W  的广�?? --- 点乘
        pos : B K 2
    Output:
        crop_feaure: B K C crop_size, crop_size

    """
    B, K, C, H, W = feature_map.shape
    mu_x = pos[..., 0]
    mu_y = pos[..., 1]
    pos_x, pos_y = np.array(torch.round((mu_y.cpu().detach() + 1) / 2 * H)), np.array(
        torch.round((mu_x.cpu().detach() + 1) / 2 * W)
    )
    crop_img = torch.zeros([B, K, C, crop_size, crop_size])
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

            crop_img[i, j] = feature_map[i, j, :, x0:x1, y0:y1]
    return crop_img


# def visualize(state, predicted_state):
#     """
#     Functions: show state and predicted_state
#     Input: state 4-d [? T K ?]
#     """
#     import matplotlib.pyplot as plt
#     plt.style.use("seaborn")

#     fig = plt.figure(figsize=(10, 5))
#     ax1 = fig.add_subplot(121)
#     axs = [fig.add_subplot(2, 4, 3), fig.add_subplot(
#         2, 4, 4), fig.add_subplot(2, 4, 7), fig.add_subplot(2, 4, 8)]
#     colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
#     for k in range(4):
#         ax1.plot(state[0, :, k, 1], -state[0, :, k, 0], c="black", marker=".")
#         ax1.plot(predicted_state[0, :, k, 1], -
#                  predicted_state[0, :, k, 0], marker=".")  # 倒着�??

#         for i in range(4):  # n_coef
#             axs[k].plot(state[0, :100, k, 2 + i],
#                         label="Alpha " + str(i), c=colors[i])
#             axs[k].plot(predicted_state[0, :100, k, 2 + i], '--', c=colors[i])
#             axs[k].set_ylim(0, 1)

#         axs[k].legend()

#     ax1.set_xlim([-1, 1])
#     ax1.set_ylim([-1, 1])
#     plt.tight_layout()
#     plt.show()
#     plt.savefig('just_try.png')


def vis_img(cd, cd_hat, log_dir, batch_idx):
    # [b horzion C H W] 每个batch 的第一个视频预测结�?? T C H W
    cd = (cd + 1) / 2  # [-1,1] -> [0,1]
    cd_hat = (cd_hat + 1) / 2
    fig, ax = plt.subplots(2, 8, figsize=(12, 4))

    # 预测空间分成8�??
    idx = np.linspace(0, cd.shape[0] - 1, 8).astype(int)
    # idx = [9, 29, 49]
    for k in range(len(idx)):
        ax[0][k].imshow(cd[idx[k]])
        ax[1][k].imshow(cd_hat[idx[k]])
        ax[0][k].axis("off")
        ax[1][k].axis("off")
        ax[0][k].set_title("t=" + str(idx[k] + 1))
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, batch_idx + "_compare.jpg"))


def vis_kpts(vis_path, keypoints_ab, rgb_seq, video_name="img_kpts", video_idx=1):
    """
    keypoints_ab : torch cuda T K 2
    rgb_seq: numpy T C H W
    """
    print(keypoints_ab.shape, rgb_seq.shape)
    coordinator = (keypoints_ab.cpu().detach().numpy() + 1) * 56  # 150 24 2
    # print(coordinator.min(), coordinator.max())
    rgb_seq = (rgb_seq + 1) / 2 * 255.0
    T = rgb_seq.shape[0]
    K = coordinator.shape[1]
    color_list = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
    ]

    img_kpts_list = []
    for t in range(T):
        img_kpts = np.uint8(rgb_seq[t].transpose(1, 2, 0)).copy()
        for s in range(K):
            y = coordinator[t, s, 0]
            x = coordinator[t, s, 1]
            img_kpts = cv2.circle(img_kpts, (int(x), int(y)), 2, color_list[s], -1)

        img_kpts_list.append(img_kpts)
    output_path = os.path.join(vis_path, "video_{:06d}".format(video_idx))

    imageio.mimsave(
        os.path.join(output_path, video_name + ".mp4"), img_kpts_list, fps=10
    )


def vis_save_video(save_path, ex, gt_state, rec_state, rgb_data, rec_rgb, given_rgb):
    """
    Functions: 保存视频和图�??,只考虑预测部分
    Input:
        save_path: 'output/ballsCF/logs_for_visualize'
        ex: video_index
        gt_state: [horizon K S]  gt_state = gt_state[given_frames:given_frames+horizon]  cuda
        rec_state: [horizon K S]  cuda
        rgb_data:  [horizon 3 112 112] rgb_data = rgb_data[given_frames:given_frames+horizon] cuda
        rec_rgb: [horizon 3 112 112]  cuda
        given_rgb: [frames 3 112 112 ] cuda
    """

    list_loss_position = []
    list_loss_img = []
    horizon = rec_rgb.shape[0]

    for t in range(horizon):
        loss_position = MSE(gt_state[t, :, :2], rec_state[t, :, :2]).item() * 1e3
        loss_img = MSE(rgb_data[t], rec_rgb[t]).item() * 1e3
        list_loss_position.append(loss_position)
        list_loss_img.append(loss_img)

    loss_position = np.mean(list_loss_position)
    loss_img = np.mean(list_loss_img)
    out = {
        "ex": ex,
        "horizon": horizon,
        "loss_position": loss_position,
        "loss_img": loss_img,
        "list_loss_position": list_loss_position,
        "list_loss_img": list_loss_img,
    }
    print("loss_position:", loss_position, "loss_img:", loss_img)

    output_path = os.path.join(save_path, "video_{:06d}".format(ex))

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print("save vis result in ... ", output_path)
    with open(os.path.join(output_path, "result.json"), "w") as f:
        json.dump(out, f, indent=2)

    # %% 可视化保持视频和等间隔的图像�??
    rgb_data = (rgb_data.cpu().detach().numpy() + 1) * 255 / 2.0
    rec_rgb = (rec_rgb.cpu().detach().numpy() + 1) * 255 / 2.0
    given_rgb = (given_rgb.cpu().detach().numpy() + 1) * 255 / 2.0

    list_compare = []
    list_raw_img = []
    list_rec_img = []
    list_init_img = []
    pad = np.zeros((112, 4, 3), dtype=np.uint8)
    pad_left = np.zeros((112, 6, 3), dtype=np.uint8)
    pad_right = np.zeros((112, 6, 3), dtype=np.uint8)

    for t in range(given_rgb.shape[0]):
        list_init_img.append(np.uint8(given_rgb[t].transpose(1, 2, 0)))

    for t in range(horizon):
        list_rec_img.append(np.uint8(rec_rgb[t].transpose(1, 2, 0)))
        list_raw_img.append(np.uint8(rgb_data[t].transpose(1, 2, 0)))

        # compare_img = np.concatenate((pad_left, np.uint8(rgb_data[t].transpose(
        #     1, 2, 0)), pad, np.uint8(rec_rgb[t].transpose(1, 2, 0)), pad_right), axis=-2)
        compare_img = np.concatenate(
            (
                np.uint8(rgb_data[t].transpose(1, 2, 0)),
                np.uint8(rec_rgb[t].transpose(1, 2, 0)),
            ),
            axis=-3,
        )
        list_compare.append(compare_img)

    print("开始生成video")
    print("video_idx = %d, video_lens = %d frames" % (ex, horizon))

    imageio.mimsave(os.path.join(output_path, "init.mp4"), list_init_img, fps=10)
    imageio.mimsave(os.path.join(output_path, "raw.mp4"), list_raw_img, fps=10)
    imageio.mimsave(os.path.join(output_path, "rec.mp4"), list_rec_img, fps=10)
    imageio.mimsave(os.path.join(output_path, "compare.mp4"), list_compare, fps=10)

    # save_step = np.arange(0, horizon, horizon // 5).tolist()
    save_step = np.linspace(0, horizon - 1, 6).astype(int).tolist()
    save_step.append(horizon - 1)
    print("save future frame index :", save_step)
    imgs = []
    recs = []

    for t in save_step:
        if t >= horizon:
            break
        imgs.append(np.concatenate((list_raw_img[t], pad), axis=-2))
        recs.append(np.concatenate((list_rec_img[t], pad), axis=-2))

    vis_img = np.concatenate(imgs, axis=-2)
    # print(vis_img.shape)
    vis_rec = np.concatenate(recs, axis=-2)
    vis_com = np.concatenate((vis_img, vis_rec), axis=-3)

    imageio.imsave(os.path.join(output_path, "vis_img_step.png"), vis_img)
    imageio.imsave(os.path.join(output_path, "vis_rec_step.png"), vis_rec)
    imageio.imsave(os.path.join(output_path, "vis_com_step.png"), vis_com)

    print("loss_postion", out["loss_position"], "loss_img", out["loss_img"])
    plt.plot(np.arange(horizon), out["list_loss_position"], label="loss_position")
    plt.plot(np.arange(horizon), out["list_loss_img"], label="loss_img")
    plt.legend()
    plt.xlabel("horizon / frame")
    plt.ylabel("loss / e-3")
    plt.grid(ls="--")
    plt.savefig(os.path.join(output_path, "loss_result.png"))
    return out


def remove_background(rgb, dataset):
    """
    Try to compute the mask between background and foreground
    Args:
        rgb: T,  H, W, C
    Returns:
        mask : T, H, W, 1
    """

    rgb = (rgb + 1) / 2  # (0,1)

    # STEP 1 : Found most probable background plt.read �??0,1�?? H W 3

    backgrounds = [
        plt.imread(
            f"Datasets/CoDy/backgrounds/plane_{i}{'_balls' if dataset in ('balls', 'ballsCF') else ''}.png"
        )[..., :3]
        for i in range(4)
    ]  # 只有balls数据集是平面背景

    error = [((rgb - b.reshape((1, 112, 112, 3))) ** 2).mean() for b in backgrounds]
    background = backgrounds[np.argmin(error)]  # mse最小的为最适合的背�??

    # STEP 2 : Remove and threshold (T H W)--> [0,1] 相当于mask
    difference = ((rgb - background.reshape((1, 112, 112, 3))) ** 2).sum(-1)
    difference = difference > 0.1

    for t in range(difference.shape[0]):  # 可用来填充孔�??
        difference[t] = skimage.morphology.binary_closing(difference[t])

    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(rgb[0])
    # ax[1].imshow(background)
    # ax[2].matshow(difference[0])
    #
    # for a in ax:
    #    a.axis("off")
    # fig.tight_layout()
    #
    # for t in range(rgb.shape[0]):
    #    ax[0].clear()
    #    ax[2].clear()
    #    ax[1].clear()
    #    ax[0].axis("off")
    #    ax[2].axis("off")
    #    ax[1].axis("off")
    #    ax[0].imshow(rgb[t])
    #    ax[2].matshow(difference[t])
    #    ax[1].imshow(rgb[t] * difference[t].reshape((112, 112, 1)))
    #    plt.pause(0.001)
    # T H W 1 binary mask
    return difference.reshape((difference.shape[0], 112, 112, 1))


def open_url(
    url: str,
    num_attempts: int = 10,
    verbose: bool = True,
    return_filename: bool = False,
):
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match("^[a-z]+://", url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith("file://"):
        filename = urllib.parse.urlparse(url).path
        if re.match(r"^/[a-zA-Z]:", filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [
                                html.unescape(link)
                                for link in content_str.split('"')
                                if "export=download" in link
                            ]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError(
                                "Google Drive download quota exceeded -- please try again later"
                            )

                    match = re.search(
                        r'filename="([^"]*)"',
                        res.headers.get("Content-Disposition", ""),
                    )
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)


def compute_stats(feats: np.ndarray):
    mu = feats.mean(axis=0)  # [d]
    sigma = np.cov(feats, rowvar=False)  # [d, d]

    return mu, sigma


def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray):
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(
        np.dot(sigma_gen, sigma_real), disp=False
    )  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)


def compute_our_fvd(
    videos_fake: np.ndarray, videos_real: np.ndarray, device: str = "cuda"
) -> float:
    detector_url = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
    # Return raw features before the softmax layer.
    detector_kwargs = dict(rescale=False, resize=False, return_features=True)

    with open_url(detector_url, verbose=False) as f:
        detector = torch.jit.load(f).eval().to(device)

    videos_fake = torch.from_numpy(videos_fake).permute(0, 4, 1, 2, 3).to(device)
    videos_real = torch.from_numpy(videos_real).permute(0, 4, 1, 2, 3).to(device)

    feats_fake = detector(videos_fake, **detector_kwargs).cpu().numpy()
    feats_real = detector(videos_real, **detector_kwargs).cpu().numpy()

    return compute_fvd(feats_fake, feats_real)


def update_json(dir, new_data):
    old_data = {}
    if os.path.isfile(dir):
        with open(dir, "r") as f:
            old_data = json.load(f)
    old_data.update(new_data)
    with open(dir, "w") as f:
        json.dump(old_data, f, indent=2)


# seed_fake = 1
# seed_real = 2
# num_videos = 128
# video_len = 16

# videos_fake = np.random.RandomState(seed_fake).rand(
#     num_videos, video_len, 224, 224, 3).astype(np.float32)
# videos_real = np.random.RandomState(seed_real).rand(
#     num_videos, video_len, 224, 224, 3).astype(np.float32)

# print('<==============================================================>')
# print('Computing our FVD...')
# print('<==============================================================>')
# our_fvd_result = compute_our_fvd(videos_fake, videos_real)


def vizu(
    source,
    target,
    target_hat,
    target_keypoints,
    log_dir,
    idx,
    img_width,
    img_height=None,
):
    """
    Functions:可视化原图像、目标图像、提取的关键点和重建目标图像的结�?? 2*2
    Input: [B C H W]
    """
    # source_img target_img rec_target keypoints
    if img_height == None:
        img_height = img_width

    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    source = source.detach().cpu().numpy()[-1].transpose((1, 2, 0))  # H W C
    target = target.detach().cpu().numpy()[-1].transpose((1, 2, 0))
    pred = target_hat.detach().cpu().numpy()[-1].transpose((1, 2, 0))

    source = (source + 1) / 2
    target = (target + 1) / 2
    pred = (pred + 1) / 2
    # print(pred.max(),pred.min())

    ax1.imshow(source.squeeze())  # 内部会自动归一�??
    ax2.imshow(target.squeeze())
    ax3.imshow(target.squeeze())
    ax4.imshow(pred.squeeze())

    keypoints_target = target_keypoints.detach().cpu().numpy()[-1]
    K = keypoints_target.shape[0]
    # X, Y = (keypoints_target[:, 1] + 1) / 2 * \
    #     112, (keypoints_target[:, 0] + 1) / 2 * 112  # X,Y�??2D坐标 [0,112]
    X, Y = (keypoints_target[:, 1] + 1) / 2 * img_width, (
        keypoints_target[:, 0] + 1
    ) / 2 * img_height  # X,Y�??2D坐标 [0,112] 真好一个xy轴相�??
    c_bank = ["red", "green", "blue", "yellow"]
    # c_bank = ['green',  'blue', 'red',  'yellow']
    for k in range(K):
        ax3.scatter(X[k], Y[k], c=c_bank[k], s=15)
    ax4.scatter(X, Y, c="white", s=15)

    ax1.axis("off")  # 取消坐标�??
    ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")

    ax1.set_title("Source image")
    ax2.set_title("Target image")
    ax3.set_title("Keypoints")
    ax4.set_title("Reconstruction")

    plt.tight_layout()
    plt.show()

    log_dir = os.path.join(log_dir, "vis")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # plt.savefig(os.path.join(
    #     log_dir, 'video_'+str(idx[-1].item())+'_keypointNet_result'+'.png'))
    plt.savefig(
        os.path.join(log_dir, "video_" + str(idx[-1]) + "_keypointNet_result" + ".png")
    )


def vizu_kth(source, target, target_hat, target_keypoints, log_dir, idx, img_width):
    """
    Functions:可视化原图像、目标图像、提取的关键点和重建目标图像的结�?? 2*2
    Input: [B C H W]
    """
    # source_img target_img rec_target keypoints

    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    source = source.detach().cpu().numpy()[-1].transpose((1, 2, 0))  # H W C
    target = target.detach().cpu().numpy()[-1].transpose((1, 2, 0))
    pred = target_hat.detach().cpu().numpy()[-1].transpose((1, 2, 0))

    source = (source + 1) / 2
    target = (target + 1) / 2
    pred = (pred + 1) / 2
    print(pred.max(), pred.min())

    ax1.imshow(source.squeeze(), cmap="gray", vmin=0, vmax=1)  # 内部会自动归一�??
    ax2.imshow(target.squeeze(), cmap="gray", vmin=0, vmax=1)
    ax3.imshow(target.squeeze(), cmap="gray", vmin=0, vmax=1)
    ax4.imshow(pred.squeeze(), cmap="gray", vmin=0, vmax=1)

    keypoints_target = target_keypoints.detach().cpu().numpy()[-1]
    K = keypoints_target.shape[0]
    # X, Y = (keypoints_target[:, 1] + 1) / 2 * \
    #     112, (keypoints_target[:, 0] + 1) / 2 * 112  # X,Y�??2D坐标 [0,112]
    X, Y = (keypoints_target[:, 1] + 1) / 2 * img_width, (
        keypoints_target[:, 0] + 1
    ) / 2 * img_width  # X,Y�??2D坐标 [0,112]
    c_bank = ["red", "green", "blue", "yellow"]
    # c_bank = ['green',  'blue', 'red',  'yellow']
    for k in range(K):
        # ax3.scatter(X[k], Y[k], c=c_bank[k], s=15)
        ax3.scatter(X[k], Y[k], c="white", s=15, marker="*")
    ax4.scatter(X, Y, c="white", s=15, marker="*")

    ax1.axis("off")  # 取消坐标�??
    ax2.axis("off")
    ax3.axis("off")
    ax4.axis("off")

    ax1.set_title("Source image")
    ax2.set_title("Target image")
    ax3.set_title("Keypoints")
    ax4.set_title("Reconstruction")

    plt.tight_layout()
    plt.show()

    log_dir = os.path.join(log_dir, "vis")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # plt.savefig(os.path.join(
    #     log_dir, 'video_'+str(idx[-1].item())+'_keypointNet_result'+'.png'))
    plt.savefig(
        os.path.join(log_dir, "video_" + str(idx[-1]) + "_keypointNet_result" + ".png")
    )


def make_image(tensor, colored_kp=False, key_points=None):
    tensor = tensor.cpu().clamp(-0.5, 0.5) + 0.5
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    tensor = tensor.transpose(0, 1).transpose(1, 2).numpy() * 255

    if colored_kp:
        radius = 1
        color_bar = np.array(
            [
                [248, 248, 24],
                [246, 221, 41],
                [254, 191, 60],
                [213, 190, 39],
                [148, 202, 73],
                [83, 204, 125],
                [41, 195, 170],
                [1, 183, 202],
                [33, 164, 227],
                [45, 142, 242],
                [58, 115, 255],
                [71, 86, 247],
            ]
        )
        for i in range(len(key_points)):
            x = [
                np.clip(key_points[i][0] - radius, 0, 63),
                np.clip(key_points[i][0] + radius, 0, 63),
            ]
            y = [
                np.clip(key_points[i][1] - radius, 0, 63),
                np.clip(key_points[i][1] + radius, 0, 63),
            ]
            tensor[y[0] : y[1] + 1, key_points[i][0]] = color_bar[i]
            tensor[key_points[i][1], x[0] : x[1] + 1] = color_bar[i]
    # pdb.set_trace()
    return Image.fromarray(np.uint8(tensor))


def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)


def save_image(filename, tensor, colored_kp=False, key_points=None):
    img = make_image(tensor, colored_kp, key_points)
    img.save(filename)


def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)


def save_gif(filename, inputs, duration=0.15, colored_kp=False, key_points=None):
    images = []
    m = 0
    for tensor in inputs:
        img = image_tensor(tensor, padding=0)
        img = img.cpu()
        img = img.transpose(0, 1).transpose(1, 2).clamp(-0.5, 0.5)
        img = (img.numpy() + 0.5) * 255
        if colored_kp:
            img += 255
            radius = 1
            kp = key_points[m]
            color_bar = np.array(
                [
                    [248, 248, 24],
                    [246, 221, 41],
                    [254, 191, 60],
                    [213, 190, 39],
                    [148, 202, 73],
                    [83, 204, 125],
                    [41, 195, 170],
                    [1, 183, 202],
                    [33, 164, 227],
                    [45, 142, 242],
                    [58, 115, 255],
                    [71, 86, 247],
                ]
            )
            for i in range(len(kp)):
                x = [
                    np.clip(kp[i][0] - radius, 0, 63),
                    np.clip(kp[i][0] + radius, 0, 63),
                ]
                y = [
                    np.clip(kp[i][1] - radius, 0, 63),
                    np.clip(kp[i][1] + radius, 0, 63),
                ]
                img[y[0] : y[1] + 1, kp[i][0]] = color_bar[i]
                img[kp[i][1], x[0] : x[1] + 1] = color_bar[i]
            m += 1
        images.append(img.astype("uint8"))
    imageio.mimsave(filename, images, duration=duration)
