from omegaconf import OmegaConf
import torch
import numpy as np
from tqdm import tqdm
import os
import os.path as osp
import argparse
import sys
import json
import glob
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip

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
        default="./configs/full.yaml",
        help="ldm model config path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)

    split_path = config.data.params.test.params.dataset.split_path
    data_items = json.load(open(split_path, "r"))["test"]
    scene_video_list = list(
        set(
            [
                (
                    i[0],
                    (
                        "_".join(i[1].split("_")[:6])
                        if "2024-" in i[1]
                        else "_".join(i[1].split("_")[:5])
                    ),
                )
                for i in data_items
            ]
        )
    )
    data_dir = config.data.params.test.params.dataset.data_dir
    wav_dir = osp.join(args.log_dir, "gen_wav_16k_80")

    output_dir = osp.join(args.log_dir, "videos_with_generated_audio")
    os.makedirs(output_dir, exist_ok=True)
    for scene, video in tqdm(scene_video_list):
        wav_file_list = glob.glob(osp.join(wav_dir, video + "*.wav"))
        wav_file_list.sort(key=lambda x: int(x.split("_")[-2]))
        combined_audio = AudioSegment.empty()
        for wav_file in wav_file_list:
            wav_path = os.path.join(wav_dir, wav_file.split("/")[-1])

            audio = AudioSegment.from_wav(wav_path)
            combined_audio += audio
        combined_audio.export("tmp.wav", format="wav")

        video_path = osp.join(
            data_dir, scene, "hamer_videos_with_aug", video + "_with_audio.mp4"
        )
        video_clip = VideoFileClip(video_path)
        video_clip = video_clip.subclip(0, combined_audio.duration_seconds)
        assert video_clip.duration == combined_audio.duration_seconds

        new_audio_clip = AudioFileClip("tmp.wav")
        output_video = video_clip.set_audio(new_audio_clip)
        output_video_path = osp.join(
            output_dir, video + f"_{scene}" + "_generated.mp4")
        output_video.write_videofile(
            output_video_path, codec="libx264", audio_codec="aac", threads=8, preset="ultrafast"
        )
