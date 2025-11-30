experiment_name="sarf_full"

export CUDA_VISIBLE_DEVICES=0

python generate_mel_cfg_hand_pose.py \
    --log_dir ../logs/$experiment_name \
    --config ./configs/full.yaml \
    --timesteps 26 --solver euler

cd ./vocoder/generate_wav
python mel2wav_vocoder.py \
    --log_dir ../../../logs/$experiment_name
cd ../../

python combine_generated_audio_with_video.py \
        --log_dir ../logs/$experiment_name \
        --config ./configs/full.yaml