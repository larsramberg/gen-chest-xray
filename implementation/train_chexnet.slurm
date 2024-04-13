#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=0-34:00:00     # 0 days and 15 minutes limit
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --job-name="chexnet"
#SBATCH --mem=32000
#SBATCH --cpus-per-task=6
#SBATCH --output=chexnet_slurm_log_hp.txt    # Log file
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
module load torchvision/0.13.1-foss-2022a-CUDA-11.7.0
module load matplotlib/3.7.2-gfbf-2023a
echo "Starting training"
python train_chexnet.py
echo "Training Completed"
