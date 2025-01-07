#!/bin/bash
#SBATCH -o ./ccv_logs/train-genie35M-%j.out
#SBATCH -e ./ccv_logs/train-genie35M-%j.err

#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gpus=1


source venv/bin/activate

./venv/bin/python3 train.py \
    --genie_config genie/configs/genie35M_n32_h8_d256.json \
    --output_dir data/genie_model \
    --max_eval_steps 10 \
    --per_device_train_batch_size 10