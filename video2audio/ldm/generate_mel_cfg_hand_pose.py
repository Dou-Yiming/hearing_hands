from ipdb import set_trace as st
import librosa.display
import librosa
from matplotlib import pyplot as plt
import soundfile as sf
import glob
from diff_foley.util import instantiate_from_config
from omegaconf import OmegaConf
import torch
import numpy as np
from tqdm import tqdm
import random
import os
import os.path as osp
import argparse
import sys
import json
import cv2
from PIL import Image
import torchvision.transforms as T

sys.path.append("/".join(os.getcwd().split("/")[:-2]))


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model


def load_video(video_path):
    img_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )
    cap = cv2.VideoCapture(video_path)
    video_frames = []
    while cap.isOpened():
        ret, video_frame = cap.read()
        if not ret:
            break
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        video_frame = img_transform(Image.fromarray(video_frame))
        video_frames.append(video_frame)
    cap.release()
    video_frames = torch.stack(video_frames)
    return video_frames


@torch.no_grad()
def generate_mel(ldm_config_path, ldm_log_dir, gpus, timesteps, solver):
    # Set Device:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device = torch.device("cuda")

    seed_everything(21)

    sample_num = 1  # Set sample_num to 1
    # Inference Param:
    cfg_scale = 4.5  # Classifier-Free Guidance Scale
    cg_scale = 50  # Classifier Guidance Scale
    steps = timesteps  # Inference Steps
    sampler = solver  # DPM-Solver Sampler
    truncate_len = 32

    ldm_ckpt_dir = osp.join(ldm_log_dir, "checkpoints")
    ldm_ckpt_path = sorted(glob.glob(f"{ldm_ckpt_dir}/epoch=*.ckpt"))[-1]

    save_mel_path = osp.join(ldm_log_dir, "mel")
    save_mel_img_path = osp.join(ldm_log_dir, "mel_images")
    os.makedirs(save_mel_path, exist_ok=True)
    os.makedirs(save_mel_img_path, exist_ok=True)

    # LDM Config:
    config = OmegaConf.load(ldm_config_path)

    # Loading LDM:
    latent_diffusion_model = load_model_from_config(config, ldm_ckpt_path)

    split_path = config.data.params.test.params.dataset.split_path
    data_items = json.load(open(split_path, "r"))["test"]
    data_dir = config.data.params.test.params.dataset.data_dir

    mel_path_list = [
        osp.join(data_dir, scene, "mel", item + ".npy") for scene, item in data_items
    ]
    # st()
    hand_pose_path_list = [
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
        for scene, item in data_items
    ]
    clip_list = [
        osp.join(data_dir, scene, "clip_features", item + ".npy")
        for scene, item in data_items
    ]
    clip_local_list = [
        osp.join(data_dir, scene, "clip_features_local", item + ".npy")
        for scene, item in data_items
    ]
    video_list = [
        osp.join(data_dir, scene, "hamer_videos_truncated", item + ".mp4")
        for scene, item in data_items
    ]

    for gt_mel_path, hand_pose_path, video_path, clip_path, clip_local_path in tqdm(
        zip(mel_path_list, hand_pose_path_list, video_list, clip_list,
            clip_local_list),
        desc="Data:",
        total=len(mel_path_list),
    ):

        cur_mel_path = osp.join(
            save_mel_path, gt_mel_path.split("/")[-1].replace(".npy", ".npy")
        )
        cur_mel_image_path = osp.join(
            save_mel_img_path, gt_mel_path.split("/")[-1].replace(".npy", ".png")
        )

        video_feat = []
        if config.model.params.cond_stage_config.params.use_clip_feat:
            clip_feats = np.load(clip_path)
            video_feat.append(clip_feats)
        if config.model.params.cond_stage_config.params.use_clip_local_feat:
            clip_local_feats = np.load(clip_local_path)
            video_feat.append(clip_local_feats)
        if len(video_feat) == 0:
            video_feat.append(np.zeros((truncate_len, 0)))
        video_feat = np.concatenate(video_feat, axis=1)
        video_feat = torch.Tensor(video_feat).unsqueeze(0).to(device)

        hand_pose = np.load(hand_pose_path)
        hand_pose = hand_pose.reshape(1, hand_pose.shape[0], -1)
        hand_pose = torch.Tensor(hand_pose.astype(np.float32)).to(device)
        if not config.model.params.cond_stage_config.params.use_hand_pose:
            hand_pose = torch.zeros_like(hand_pose)

        # Truncate the Video Cond:
        feat_len = video_feat.shape[1]
        window_num = feat_len // truncate_len

        mel_list = []  # [sample_list1, sample_list2, sample_list3 ....]
        for i in tqdm(range(window_num), desc="Window:", leave=False):
            start, end = i * truncate_len, (i + 1) * truncate_len
            video_feat = video_feat[:, start:end]

            cond = {
                "mix_video_feat": video_feat,
                "mix_hand_pose": hand_pose,
                # "mix_video_frames": video_frames
            }
            # 1). Get Video Condition Embed:
            embed_cond_feat = latent_diffusion_model.get_learned_conditioning(
                cond)

            # 2). CFG unconditional Embedding:
            uncond_cond = torch.zeros(embed_cond_feat.shape).to(device)

            # 3). Diffusion Sampling:
            audio_samples, _ = latent_diffusion_model.sample_param_cfg(
                embed_cond_feat,
                cfg_scale=cfg_scale,
                batch_size=sample_num,
                solver=sampler,
                timesteps=steps,
            )

            # 4). Decode Latent:
            audio_samples = latent_diffusion_model.decode_first_stage(
                audio_samples)
            # print(audio_samples.shape)
            if len(audio_samples.shape) == 4:
                audio_samples = audio_samples[:,
                                              0, :, :].detach().cpu().numpy()
            else:
                audio_samples = audio_samples.detach().cpu().numpy()
            mel_list.append(audio_samples)
        # Save Sample:
        current_mel_list = []
        for k in range(window_num):
            current_mel_list.append(mel_list[k][0])
        if len(current_mel_list) > 0:
            current_mel = np.concatenate(current_mel_list, 1)
            np.save(cur_mel_path, current_mel)

            mel_spectrogram = current_mel
            sr = 16000  # Replace with your actual sampling rate

            # Calculate the time axis manually based on the known duration (8.192 seconds)
            duration = 8  # duration in seconds
            time_axis = np.linspace(0, duration, mel_spectrogram.shape[1])

            gt_mel = np.load(gt_mel_path)

            # save gt and generated mel spectrogram respectively
            plt.figure(figsize=(10, 2))
            librosa.display.specshow(gt_mel, sr=sr, fmax=8000, cmap="coolwarm")
            plt.tight_layout()
            plt.savefig(
                cur_mel_image_path.split(".png")[0] + "_gt.png",
                format="png",
                dpi=800,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

            plt.figure(figsize=(10, 2))
            librosa.display.specshow(
                mel_spectrogram, sr=sr, fmax=8000, cmap="coolwarm")
            plt.tight_layout()
            # remove even more whitespace
            plt.savefig(
                cur_mel_image_path.split(".png")[0] + "_pred.png",
                format="png",
                dpi=800,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
        help="ldm model checkpoints path",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config/Stage2_LDM.yaml",
        help="ldm model config path",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="use gpu",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="euler",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    generate_mel(
        ldm_config_path=args.config,
        ldm_log_dir=args.log_dir,
        gpus=args.gpus,
        timesteps=args.timesteps,
        solver=args.solver,
    )
