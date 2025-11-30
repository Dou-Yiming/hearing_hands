import sys
import os
import os.path as osp

sys.path.append("../../")
import numpy as np
import soundfile as sf
from vocoder.bigvgan.models import VocoderBigVGAN
import librosa
import glob
from tqdm import tqdm
from multiprocessing import Pool
import argparse


def inverse_op(spec, sr):
    n_fft = 1024
    fmin = 125
    fmax = 7600
    nmels = 80
    hoplen = 1024 // 4
    spec_power = 1

    # Inverse Transform
    spec = spec * 100 - 100
    spec = (spec + 20) / 20
    spec = 10**spec
    spec_out = librosa.feature.inverse.mel_to_stft(
        spec, sr=sr, n_fft=n_fft, fmin=fmin, fmax=fmax, power=spec_power
    )
    wav = librosa.griffinlim(spec_out, hop_length=hoplen)
    return wav


def process_data(mel_file, target_dir):
    file_name = mel_file.split("/")[-1][:-6]
    try:
        wav_file = os.path.join(
            target_dir, mel_file.split("/")[-1].replace(".npy", ".wav")
        )
        mel_spec = np.load(mel_file)
        sample = inverse_op(mel_spec, 22050)
        sf.write(wav_file, sample, 16000)
        return file_name, True
    except Exception as e:
        print(e)
        return file_name, False


success_list = []
err_list = []
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
    )
    args = parser.parse_args()

    vocoder = VocoderBigVGAN(
        "../bigvgan/checkpoints", "cuda"
    )

    log_dir = args.log_dir

    target_dir = osp.join(log_dir, "gen_wav_16k_80")
    os.makedirs(target_dir, exist_ok=True)
    for mel_file in tqdm(sorted(glob.glob(f"{log_dir}/mel/*.npy"))):
        wav_file = osp.join(target_dir, mel_file.split("/")
                            [-1].replace(".npy", ".wav"))
        mel_spec = np.load(mel_file)
        sample = vocoder.vocode(mel_spec)
        sf.write(wav_file, sample, 16000)
