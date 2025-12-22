#!/bin/bash -eux
#SBATCH --job-name=training
#SBATCH --account=sci-rabl
#SBATCH --output=train.out
#SBATCH --partition=gpu 
#SBATCH --gpus=1 
#SBATCH --time=05:00:00
#SBATCH --mem=50gb
#SBATCH --nodelist=gx02

# uncomment to receive email when job fails
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smilla.fox@student.hpi.de

source .venv/bin/activate
python main.py model=resnet18 dataset=gtsrb training.train=true training.use_data_augmentation=true training.batch_size_train=256 training.batch_size_test=256 pruning.pruning_ratio=0.2 log_results=true run_name=train_gtsrb_resnet18_v0
