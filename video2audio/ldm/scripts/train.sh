python main.py \
    --base ./configs/full.yaml\
    -t --gpus 1 --stage 2 --epoch 40 \
    --wandb_project hearing_hands --scale_lr False\
    -l ../logs