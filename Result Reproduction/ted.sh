#!/bin/bash
#SBATCH --job-name=ted                   # Job name
#SBATCH --output=ted_output.log           # Standard output log
#SBATCH --error=ted_error.log             # Standard error log
#SBATCH --partition=gpu                         # Use GPU partition
#SBATCH --gres=gpu:a100:1                       # Request 1 A100 GPU
#SBATCH --time=48:00:00                         # Time limit (48 hours)
#SBATCH --mail-user=tomledeakin@gmail.com       # Email for notifications
#SBATCH --mail-type=END,FAIL                    # Notify on job completion or failure


# Navigate to the project directory
cd "$HOME/BackdoorBox Research/backdoor-toolbox" || { echo "Directory not found"; exit 1; }

# Activate the Python virtual environment
source "my_env/bin/activate"

cd "$HOME/TED Research/ted" || { echo "Directory not found"; exit 1; }

# python train_SSDT.py --dataset mnist --attack_mode SSDT --n_iters 50
python TED_nv.py
python TED.py

echo "All tasks completed successfully."






