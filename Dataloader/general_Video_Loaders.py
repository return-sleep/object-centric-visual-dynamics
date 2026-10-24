import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os
from random import randint
import torchvision
import pybullet as pb
import cv2
import torch
import json


def read_datasetdir(dataset, env):
    """
    Function: 返回指定视频数据集的存放路径

    Input:
        dataset(str) : 'CoDy' or 'Cophy'
        env(str) : 'blocktowerCF' or 'ballsCF'
    Return:
        dataset_dir(str) : 数据集绝对路径
    """
    with open("Datasets/dataset_dir.json", "r") as f:
        datasets_dirlist = json.load(f)
        assert dataset in ["CoDy", "Cophy"]
        assert env in ["blocktowerCF", "ballsCF"]
        dataset_dir = datasets_dirlist[dataset][env]
    return dataset_dir


class Video_loader(Dataset):
    def __init__(
        self,
        mode="train",
        resolution=112,
        load_cd=False,
        sampling_mode="full",
        load_ab=True,
        load_state=False,
        dataset="CoDy",
        env="blocktowerCF",
        max_frames=None,
    ):
        """
        Dataloader for BallsCF
        :param mode: 'train', 'test' or 'val' split
        :param resolution: Image resolution, default is 112x112
        :param load_cd: if False, the dataloader does not read CD video
        :param sampling_mode: 'rand' for random selection in the video, 'fix' for sampling at fixed timestamps and
        'full' for loading the entire video. This is usefull for training De-rendering, where only two images are
        needed instead of the entire video
        :param load_ab: if False, the dataloader does not read AB video
        :param load_state: True to load the 2D projection of the ground truth state
        """
        # 执行父类的构造函数，使得我们能够调用父类的属性。
        super(Video_loader, self).__init__()
        assert sampling_mode in ["rand", "fix", "full"]

        self.index = None
        path = read_datasetdir(dataset, env)
        self.dataloc = path
        # print(self.dataloc)
        assert os.path.isdir(self.dataloc)

        self.mode = mode
        self.resolution = resolution
        self.load_cd = load_cd
        self.sampling_mode = sampling_mode
        self.load_ab = load_ab
        self.load_state = load_state
        self.video_length = 0
        self.max_frames = max_frames

    def load_index(self, splits_filename):
        # 读取训练/验证/测试txt文件 -> list [video index]
        print("video:", splits_filename, self.mode)
        with open(
            f"{splits_filename}_{self.mode}.txt", "r"
        ) as file:  # Datasets/collision_val.txt
            self.index = [int(k) for k in file.readlines()]

    def __len__(self):
        return len(self.index)

    def get_projection_matrix(self):
        raise NotImplementedError("Please Implement this method")

    def __getitem__(self, item):
        ex = self.index[item]
        out = {"ex": ex}
        # print('ex', out)
        if self.load_ab:
            # 目标视频的路径
            ab = os.path.join(self.dataloc, str(ex), "ab", "rgb.mp4")
            rgb_ab, r_ab = get_rgb(
                ab, self.sampling_mode, self.video_length, self.max_frames
            )
            out["rgb_ab"] = rgb_ab

        if self.load_cd:
            cd = os.path.join(self.dataloc, str(ex), "cd", "rgb.mp4")
            rgb_cd, r_cd = get_rgb(
                cd, self.sampling_mode, self.video_length, self.max_frames
            )
            out["rgb_cd"] = rgb_cd

        if self.load_state and self.load_cd:
            states = np.load(os.path.join(self.dataloc, str(ex), "cd", "states.npy"))

            viewMatrix, projectionMatrix = self.get_projection_matrix()
            positions = states[..., :3]  # 三维坐标
            # print('load_cd', positions.shape)
            pose_2d = []
            for t in range(positions.shape[0]):
                pose_2d.append([])
                for k in range(positions.shape[1]):
                    if not np.all(positions[t, k] == 0):
                        pose_2d[-1].append(
                            convert_to_2d(
                                positions[t, k], viewMatrix, projectionMatrix, 112
                            )
                        )
                    else:
                        pose_2d[-1].append(np.zeros(2))
            pose_2d = np.array(pose_2d)
            out["pose_2D_cd"] = pose_2d[r_cd, :, :]  # 索引

        if self.load_state and self.load_ab:
            states = np.load(os.path.join(self.dataloc, str(ex), "ab", "states.npy"))

            viewMatrix, projectionMatrix = self.get_projection_matrix()
            positions = states[..., :3]  # T 4 3 (N = 4 D = 3)
            # print('load_ab', positions.shape)
            # print(viewMatrix) # 仿射变换的矩阵
            # print(projectionMatrix) # 投影的矩阵
            pose_2d = []
            for t in range(positions.shape[0]):
                pose_2d.append([])  # 4个列表
                for k in range(positions.shape[1]):
                    if not np.all(positions[t, k] == 0):
                        pose_2d[-1].append(
                            convert_to_2d(
                                positions[t, k], viewMatrix, projectionMatrix, 112
                            )
                        )  # 单个物体的坐标list
                    else:
                        pose_2d[-1].append(np.zeros(2))  # 3维的d也是反着的
            pose_2d = np.array(pose_2d)
            # print(pose_2d[20:35:5, -1])
            out["pose_2D_ab"] = pose_2d[r_ab, :, :]

        # out dict {'idx','rgb_ab','rgb_cd','pose_2D_cd'}
        return out


class ballsCF_Video(Video_loader):  # subclass的使用方法
    def __init__(self, **kwargs):  # __init__用于录入参数，放在self里面
        # 继承另一个类的参数 **kwargs 录入另一个类的参数
        # 任意数目的关键字参数符号
        super(ballsCF_Video, self).__init__(**kwargs)

        # 关键在这~~~ 数据集的txt文件路径
        self.load_index(f"Datasets/CoDy/ballsCF_4_samemass")
        # self.load_index(f"Datasets/CoDy/original_datasetsplit/ballsCF_4")
        self.video_length = 150  # 和继承相比的顺序怎么看？

    def __len__(self):
        return len(self.index)

    def get_projection_matrix(self):
        # 合成相机由两个4x4矩阵指定：视图矩阵和投影矩阵
        # camera位置/ 目标(焦点)位置 / up vector of the camera
        viewMatrix = (
            np.array(pb.computeViewMatrix([0, 0.01, 8], [0, 0, 0], [0, 0, 1]))
            .reshape((4, 4))
            .transpose()
        )  # 相机的空间位置 相机指向的点 相机“头顶”的指向量 up at y 少见啊
        # 视野/长宽比/near plane distance / far plance distance
        projectionMatrix = (
            np.array(pb.computeProjectionMatrixFOV(60, 1, 4, 20))
            .reshape((4, 4))
            .transpose()
        )  # 摄像头视线夹角（视野）/ 长宽比/ 摄像头焦距下限（近面距离）/ 摄像头焦距上限（远面距离）
        return viewMatrix, projectionMatrix


class blocktowerCF_Video(Video_loader):
    def __init__(self, **kwargs):
        super(blocktowerCF_Video, self).__init__(**kwargs)

        # Datasets/CoDy/original_datasetsplit/blocktowerCF_4_test.txt
        self.load_index(f"Datasets/CoDy/blocktowerCF_4_samemass")
        # self.load_index(f"Datasets/CoDy/original_datasetsplit/blocktowerCF_4")
        self.video_length = 150  # 25*6

    def __len__(self):
        return len(self.index)

    def get_projection_matrix(self):
        viewMatrix = (
            np.array(pb.computeViewMatrix([0, -7, 4.5], [0, 0, 1.5], [0, 0, 1]))
            .reshape((4, 4))
            .transpose()
        )
        projectionMatrix = (
            np.array(pb.computeProjectionMatrixFOV(60, 112 / 112, 4, 20))
            .reshape((4, 4))
            .transpose()
        )  # 三角形 根号3 scale 用来控制成像平面的r,l,b,t
        return viewMatrix, projectionMatrix  # 焦距和近平面 相等


def convert_to_2d(pose, view, projection, resolution, dim=2):
    center_pose = np.concatenate([pose[:3], np.ones((1))], axis=-1).reshape(
        (4, 1)
    )  # 一个点 在 3D空间
    center_pose = view @ center_pose
    center_pose = projection @ center_pose
    # 点的归一化 (nx,ny,nz,n) = (x,y,z,1) -> 视图+投影的3D坐标
    center_pose = center_pose[:3] / center_pose[-1]
    center_pose = (center_pose + 1) / 2 * resolution  # （-1~1）-> (0 112)
    center_pose[1] = resolution - center_pose[1]  # 调整一下y的 翻转?? x，112-y,z
    # if dim ==3:
    #     center_pose[2] = resolution - center_pose[2] # 是否需要呢
    return center_pose[:dim].astype(int).flatten()  # 只需要 x,y坐标


def _decode_mp4_prefix(filedir, max_frames=None):
    """Decode the first max_frames of an mp4 as uint8 [T, H, W, C]."""
    try:
        import av

        container = av.open(filedir)
        stream = container.streams.video[0]
        frames = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            if max_frames is not None and len(frames) >= max_frames:
                break
        container.close()
        if not frames:
            raise RuntimeError("empty video: %s" % filedir)
        if max_frames is not None and len(frames) < max_frames:
            raise RuntimeError("too few frames: %s" % filedir)
        return torch.from_numpy(np.ascontiguousarray(np.stack(frames, 0)))
    except Exception:
        if max_frames is None:
            rgb, _, _ = torchvision.io.read_video(filedir, pts_unit="sec")
        else:
            fps = 25.0
            rgb, _, _ = torchvision.io.read_video(
                filedir,
                start_pts=0,
                end_pts=(max_frames - 0.5) / fps,
                pts_unit="sec",
            )
            if rgb.shape[0] > max_frames:
                rgb = rgb[:max_frames]
            elif rgb.shape[0] < max_frames:
                rgb, _, _ = torchvision.io.read_video(filedir, pts_unit="sec")
                rgb = rgb[:max_frames]
        return rgb


def get_rgb(filedir, sampling_mode="full", video_length=150, max_frames=None):
    """
    Functions: convert a video to a sequence of images

    Input:
        filedir(str): filedir of video
        sampling_mode(str): 'full' 'rand'
        video_length: 150 = n_sec * fps
    Return:
        rgb(tensor): [T C H W ] in [-1,1]
        r(list): frame index
    """
    # print(filedir)
    if sampling_mode == "full":
        # Sequential PyAV decode of the prefix is faster than torchvision.read_video.
        rgb = _decode_mp4_prefix(filedir, max_frames)
        rgb = 2 * (rgb / 255) - 1  # (-1,1)
        rgb = rgb.permute(0, 3, 1, 2)  # T C H W 从这边入手
        r = list(range(rgb.shape[0]))
    else:
        t = (
            randint(0, int(0.15 * video_length))
            if sampling_mode == "rand"
            else int(0.15 * video_length)
        )
        # t = randint(0, int(0.15 * video_length)
        #             ) if sampling_mode == "rand" else 0 #
        # t = 0 # 参考帧为第一帧会好一点吗？
        r = [t, t + int(0.15 * video_length)]
        r = [1, int(0.2 * video_length)]
        # dist = randint(1, int(0.45* video_length)) if sampling_mode == "rand" else int(0.45* video_length)
        # r = [0, dist]
        # r = [0, t]
        # rgb, _, _ = torchvision.io.read_video(filedir, pts_unit="sec")
        # rgb = 2 * (rgb / 255) - 1  # (-1,1)
        # rgb = rgb.permute(0, 3, 1, 2)  # T C H W
        # rgb = rgb[r]
        capture = cv2.VideoCapture(filedir)

        list_rgb = []
        for i in r:  # 按帧索引读取部分图像
            # 允许我们取出视频中某个位置的帧  时间/帧号/比例
            capture.set(1, i)
            ret, frame = capture.read()
            # 色彩转换
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            list_rgb.append(frame)
        capture.release()
        rgb = np.stack(list_rgb, 0)  # [new_T H W C]
        # if random.random() > 0.5:
        #     rgb[0] = rgb[0, :, ::-1]
        #     rgb[1] = rgb[1,::-1]
        rgb = 2 * (rgb / 255) - 1
        rgb = rgb.astype(np.float32).transpose(0, 3, 1, 2)
        rgb = torch.FloatTensor(rgb)
    return rgb, r


def get_background(filedir, video_length):
    # 自创ing
    print("read %s" % filedir)
    result_dir = "fg_folder"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    capture = cv2.VideoCapture(filedir)
    # fbgb = cv2.createBackgroundSubtractorMOG2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.createBackgroundSubtractorKNN()
    fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
    out = cv2.VideoWriter("extract_fg.avi", fourcc, 25, (112, 112), True)
    background = cv2.imread("misc/backgrounds/plane_2_balls.png")
    for i in range(video_length):
        ret, frame = capture.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_ELLIPSE, kernel)
        bgmask = (background == frame).astype(int)
        cv2.imwrite(os.path.join(result_dir, str(i) + "_o.jpg"), frame)
        cv2.imwrite(os.path.join(result_dir, str(i) + "_b.jpg"), background)
        cv2.imwrite(
            os.path.join(result_dir, str(i) + "_substract.jpg"), frame - background
        )

        out.write(fgmask)
    capture.release()
    out.release()


def get_projection_matrix(scene):
    assert scene in ["ballsCF", "blocktowerCF"]
    if scene == "ballsCF":
        viewMatrix = (
            np.array(pb.computeViewMatrix([0, 0.01, 8], [0, 0, 0], [0, 0, 1]))
            .reshape((4, 4))
            .transpose()
        )
        # 视野/长宽比/near plane distance / far plance distance
        projectionMatrix = (
            np.array(pb.computeProjectionMatrixFOV(60, 1, 4, 20))
            .reshape((4, 4))
            .transpose()
        )
    elif scene == "blocktowerCF":
        viewMatrix = (
            np.array(pb.computeViewMatrix([0, -7, 4.5], [0, 0, 1.5], [0, 0, 1]))
            .reshape((4, 4))
            .transpose()
        )
        projectionMatrix = (
            np.array(pb.computeProjectionMatrixFOV(60, 112 / 112, 4, 20))
            .reshape((4, 4))
            .transpose()
        )
    return viewMatrix, projectionMatrix


def get_state(filedir):
    states = np.load(os.path.join(filedir, "states.npy"))
    scene = filedir.split("/")[-4]
    viewMatrix, projectionMatrix = get_projection_matrix(scene)
    positions = states[..., :3]
    # print(states.shape)
    pose_2d = []
    for t in range(positions.shape[0]):
        pose_2d.append([])
        for k in range(positions.shape[1]):
            if not np.all(positions[t, k] == 0):
                pose_2d[-1].append(
                    convert_to_2d(positions[t, k], viewMatrix, projectionMatrix, 112)
                )
            else:
                pose_2d[-1].append(np.zeros(2))
    # print(pose_2d)
    pose_2d = np.array(pose_2d)  # [T N 2]
    # pose.reshape(pose.shape[0],-1) #[T N*2(按序各目标的2D坐标)]
    return pose_2d


# filedir = 'Datasets/Cophy/blocktowerCF/0/ab'
# pose = get_state(filedir)
# print(pose)
# test_dataloader = DataLoader(
#     blocktowerCF_Video(mode='test', resolution=112,
#                        sampling_mode='full', env='blocktowerCF', load_state=True),
#     batch_size=1, num_workers=1, pin_memory=True, drop_last=True)
# print(len(test_dataloader))
# for i, x in enumerate(test_dataloader):
#     break
