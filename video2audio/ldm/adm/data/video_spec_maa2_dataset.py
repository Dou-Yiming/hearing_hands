import csv
import os
import os.path as osp
import pickle
import sys

import numpy as np
import random
import math
import json
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from ipdb import set_trace as st


class audio_video_spec_fullset_Dataset(torch.utils.data.Dataset):
    # Only Load audio dataset: for training Stage1: Audio Npy Dataset
    def __init__(
        self,
        split,
        dataset,
        use_hand_pose=True,
        use_clip_feat=True,
        use_clip_local_feat=True,
        transforms=None,
        sr=22050,
        duration=10,
        truncate=220000,
        fps=21.5,
        debug_num=False,
        fix_frames=False,
        hop_len=256,
        video_shape=(224, 224),
    ):
        super().__init__()
        self.split = split
        split_path = dataset.split_path
        self.use_multi_view = dataset.use_multi_view
        data_items = json.load(open(split_path, "r"))
        self.data_items = data_items[self.split]

        if not self.use_multi_view:
            self.data_items = [
                item for item in self.data_items if "_0_with" in item[1]]

        if self.split == "train":
            self.data_items *= 4
            if not self.use_multi_view:
                self.data_items *= 3

        data_dir = dataset.data_dir
        self.mel_list = [
            osp.join(data_dir, scene, "mel", item + ".npy")
            for scene, item in self.data_items
        ]
        self.video_list = [
            osp.join(data_dir, scene, "hamer_videos_truncated", item + ".mp4")
            for scene, item in self.data_items
        ]

        self.hand_pose_list = [
            osp.join(
                data_dir,
                scene,
                "hamer_hand_keypoints",
                (
                    "_".join(item.split("_")[:6])
                    if "2024-" in item
                    else "_".join(item.split("_")[:5])
                ),
                item + ".npy",
            )
            for scene, item in self.data_items
        ]
        self.clip_list = [
            osp.join(
                data_dir, scene, "clip_features", item + ".npy"
            )
            for scene, item in self.data_items
        ]
        self.clip_local_list = [
            osp.join(
                data_dir, scene, "clip_features_local", item + ".npy"
            )
            for scene, item in self.data_items
        ]
        assert (
            len(self.data_items)
            == len(self.mel_list)
            == len(self.video_list)
            == len(self.hand_pose_list)
            == len(self.clip_list)
            == len(self.clip_local_list)
        ), "Length of the lists are not equal"
        print("Split: {}  Sample Num: {}".format(split, len(self.data_items)))

        self.use_hand_pose = use_hand_pose
        self.use_clip_feat = use_clip_feat
        self.use_clip_local_feat = use_clip_local_feat

        # Default params:
        self.min_duration = 2
        self.sr = sr  # 22050
        self.duration = duration  # 10
        self.truncate = truncate  # 220000
        self.fps = fps
        self.fix_frames = fix_frames
        self.hop_len = hop_len
        print("Fix Frames: {}".format(self.fix_frames))

        self.img_transform = T.Compose(
            [
                T.Resize(video_shape),
                T.ToTensor(),
            ]
        )
        self.video_cache = {}

    def __len__(self):
        return len(self.data_items)

    def load_spec(self, mel_path):
        spec_raw = np.load(mel_path).astype(np.float32)  # channel: 1
        spec_len = self.sr * self.duration / self.hop_len
        if spec_raw.shape[1] < spec_len:
            spec_raw = np.tile(spec_raw, math.ceil(
                spec_len / spec_raw.shape[1]))
        spec_raw = spec_raw[:, : int(spec_len)]
        return spec_raw

    def load_hand_pose(self, hand_pose_path):
        hand_pose = np.load(hand_pose_path)
        hand_pose = hand_pose.reshape(hand_pose.shape[0], -1)
        return hand_pose.astype(np.float32)

    def load_clip_feat(self, clip_feat_path):
        clip_feat = np.load(clip_feat_path)
        return clip_feat.astype(np.float32)

    def mix_audio_and_feat(
        self,
        spec1=None,
        spec2=None,
        clip_feat1=None,
        clip_feat2=None,
        clip_local_feat1=None,
        clip_local_feat2=None,
        hand_pose1=None,
        hand_pose2=None,
        video_info_dict={},
        mode="single",
    ):
        """Return Mix Spec and Mix video feat"""
        if mode == "single":
            # spec1:
            start_idx = 0

            start_frame = int(self.fps * start_idx / self.sr)
            truncate_frame = int(self.fps * self.truncate / self.sr)
            truncate_hand_pose_frame = int(30 * self.truncate / self.sr)
            # Spec Start & Truncate:
            spec_start = int(start_idx / self.hop_len)
            spec_truncate = int(self.truncate / self.hop_len)

            spec1 = spec1[:, spec_start: spec_start + spec_truncate]

            video_feat1 = []
            if self.use_clip_feat:
                video_feat1.append(clip_feat1)
            if self.use_clip_local_feat:
                video_feat1.append(clip_local_feat1)
            if len(video_feat1) == 0:
                video_feat1.append(np.zeros((truncate_frame, 0)))
            video_feat1 = np.concatenate(video_feat1, axis=1)

            hand_pose1 = hand_pose1[
                start_frame: start_frame + truncate_hand_pose_frame
            ]

            # info_dict:
            video_info_dict["video_time1"] = (
                str(start_frame) + "_" + str(start_frame + truncate_frame)
            )  # Start frame, end frame
            video_info_dict["video_time2"] = ""
            return spec1, video_feat1, hand_pose1, video_info_dict

        elif mode == "concat":
            total_spec_len = int(self.truncate / self.hop_len)
            # Random Trucate len:
            spec1_truncate_len = random.randint(
                self.min_duration * self.sr // self.hop_len,
                total_spec_len - self.min_duration * self.sr // self.hop_len - 1,
            )
            spec2_truncate_len = total_spec_len - spec1_truncate_len

            # Sample spec clip:
            spec_start1 = random.randint(
                0, total_spec_len - spec1_truncate_len - 1)
            spec_start2 = random.randint(
                0, total_spec_len - spec2_truncate_len - 1)
            spec_end1, spec_end2 = (
                spec_start1 + spec1_truncate_len,
                spec_start2 + spec2_truncate_len,
            )
            # concat spec:
            spec1, spec2 = (
                spec1[:, spec_start1:spec_end1],
                spec2[:, spec_start2:spec_end2],
            )
            concat_audio_spec = np.concatenate([spec1, spec2], axis=1)

            # Concat Video Feat:
            start1_frame, truncate1_frame = (
                int(self.fps * spec_start1 * self.hop_len / self.sr),
                int(self.fps * spec1_truncate_len * self.hop_len / self.sr),
            )
            start2_frame, truncate2_frame = (
                int(self.fps * spec_start2 * self.hop_len / self.sr),
                int(self.fps * self.truncate / self.sr) - truncate1_frame,
            )
            start1_hand_pose_frame = int(
                np.around(30 * start1_frame / self.fps))
            start2_hand_pose_frame = int(
                np.around(30 * start2_frame / self.fps))
            truncate1_hand_pose_frame = int(
                np.around(30 * truncate1_frame / self.fps))
            truncate2_hand_pose_frame = 240 - truncate1_hand_pose_frame

            video_feat1 = []
            if self.use_clip_feat:
                video_feat1.append(
                    clip_feat1[start1_frame: start1_frame + truncate1_frame])
            if self.use_clip_local_feat:
                video_feat1.append(
                    clip_local_feat1[start1_frame: start1_frame + truncate1_frame])
            if len(video_feat1) == 0:
                video_feat1.append(np.zeros((truncate1_frame, 0)))
            video_feat1 = np.concatenate(video_feat1, axis=1)

            video_feat2 = []
            if self.use_clip_feat:
                video_feat2.append(
                    clip_feat2[start2_frame: start2_frame + truncate2_frame])
            if self.use_clip_local_feat:
                video_feat2.append(
                    clip_local_feat2[start2_frame: start2_frame + truncate2_frame])
            if len(video_feat2) == 0:
                video_feat2.append(np.zeros((truncate2_frame, 0)))
            video_feat2 = np.concatenate(video_feat2, axis=1)

            concat_video_feat = np.concatenate([video_feat1, video_feat2])

            hand_pose1, hand_pose2 = (
                hand_pose1[
                    start1_hand_pose_frame: start1_hand_pose_frame
                    + truncate1_hand_pose_frame
                ],
                hand_pose2[
                    start2_hand_pose_frame: start2_hand_pose_frame
                    + truncate2_hand_pose_frame
                ],
            )
            concat_hand_pose = np.concatenate([hand_pose1, hand_pose2])

            video_info_dict["video_time1"] = (
                str(start1_frame) + "_" + str(start1_frame + truncate1_frame)
            )  # Start frame, end frame
            video_info_dict["video_time2"] = (
                str(start2_frame) + "_" + str(start2_frame + truncate2_frame)
            )
            return (
                concat_audio_spec,
                concat_video_feat,
                concat_hand_pose,
                video_info_dict,
            )

    def __getitem__(self, idx):
        audio_name1 = self.data_items[idx][1]
        mel_path1 = self.mel_list[idx]
        video_path1 = self.video_list[idx]
        hand_pose_path1 = self.hand_pose_list[idx]
        clip_feat_path1 = self.clip_list[idx]
        clip_local_feat_path1 = self.clip_local_list[idx]
        # select other video:
        flag = False
        if self.split == "train" and random.uniform(0, 1) < 0.5:
            flag = True
            random_idx = random.randint(0, len(self.data_items) - 1)
            audio_name2 = self.data_items[random_idx]
            mel_path2 = self.mel_list[random_idx]
            video_path2 = self.video_list[random_idx]
            hand_pose_path2 = self.hand_pose_list[random_idx]
            clip_feat_path2 = self.clip_list[random_idx]
            clip_local_feat_path2 = self.clip_local_list[random_idx]
        # Load the Spec and Feat:
        spec1 = self.load_spec(mel_path1)
        hand_pose1 = self.load_hand_pose(hand_pose_path1)
        clip_feat1 = self.load_clip_feat(
            clip_feat_path1) if self.use_clip_feat else None
        clip_local_feat1 = self.load_clip_feat(
            clip_local_feat_path1) if self.use_clip_local_feat else None

        if flag:
            spec2 = self.load_spec(mel_path2)
            hand_pose2 = self.load_hand_pose(hand_pose_path2)
            clip_feat2 = self.load_clip_feat(
                clip_feat_path2) if self.use_clip_feat else None
            clip_local_feat2 = self.load_clip_feat(
                clip_local_feat_path2) if self.use_clip_local_feat else None
            video_info_dict = {
                "audio_name1": audio_name1,
                "audio_name2": audio_name2,
                "video_path1": video_path1,
                "video_path2": video_path2,
            }
            mix_spec, mix_video_feat, mix_hand_pose, mix_info = self.mix_audio_and_feat(
                spec1=spec1,
                spec2=spec2,
                clip_feat1=clip_feat1,
                clip_feat2=clip_feat2,
                clip_local_feat1=clip_local_feat1,
                clip_local_feat2=clip_local_feat2,
                hand_pose1=hand_pose1,
                hand_pose2=hand_pose2,
                video_info_dict=video_info_dict,
                mode="concat",
            )
        else:
            video_info_dict = {
                "audio_name1": audio_name1,
                "audio_name2": "",
                "video_path1": video_path1,
                "video_path2": "",
            }
            mix_spec, mix_video_feat, mix_hand_pose, mix_info = self.mix_audio_and_feat(
                spec1=spec1,
                clip_feat1=clip_feat1,
                clip_local_feat1=clip_local_feat1,
                hand_pose1=hand_pose1,
                video_info_dict=video_info_dict,
                mode="single",
            )

        output = {}
        output["mix_spec"] = mix_spec
        output["mix_video_feat"] = mix_video_feat
        output["mix_hand_pose"] = mix_hand_pose

        return output


class audio_video_spec_fullset_Dataset_Train(audio_video_spec_fullset_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split="train", **dataset_cfg)


class audio_video_spec_fullset_Dataset_Valid(audio_video_spec_fullset_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split="val", **dataset_cfg)


class audio_video_spec_fullset_Dataset_Test(audio_video_spec_fullset_Dataset):
    def __init__(self, dataset_cfg):
        super().__init__(split="test", **dataset_cfg)
