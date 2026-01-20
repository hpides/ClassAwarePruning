#!/bin/bash -eux
#SBATCH --job-name=training
#SBATCH --account=sci-rabl
#SBATCH --output=outputs/cap_%j.out
#SBATCH --error=outputs/cap_%j.err
#SBATCH --partition=gpu-batch
#SBATCH --gpus=1 
#SBATCH --time=05:00:00
#SBATCH --mem=50gb
#SBATCH --nodelist=gx02

# uncomment to receive email when job fails
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=jonas.schulze@hpi.de

export WANDB_API_KEY=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
export WANDB_ENTITY=sjoze
export WANDB_PROJECT=ClassAwarePruning

source .venv/bin/activate
##python main.py model=resnet18 dataset=gtsrb training.train=true training.use_data_augmentation=true training.batch_size_train=256 training.batch_size_test=256 pruning.pruning_ratio=0.2 log_results=true run_name=train_gtsrb_resnet18_v0
python main.py run_name=cap_%j
