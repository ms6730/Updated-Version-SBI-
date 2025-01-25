#!/bin/bash
#SBATCH --job-name=evaluate_ensemble
#SBATCH --output=/home/ms6730/c2_sbi_experiments_new/sbi_framework/%x_%j.out
#SBATCH --error=/home/ms6730/c2_sbi_experiments_new/sbi_framework/%x_%j.err
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1      # total number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=01:00:00
#SBATCH --array=1-100%4
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=ms6730@princeton.edu
# Calculate the task ID based on the SLURM_ARRAY_TASK_ID
task_id=$((SLURM_ARRAY_TASK_ID - 1))

#set path to json file with run settings
json_path='/home/ms6730/c2_experiment_sin/hydrogen-sbi/scripts/settings.json'
module purge
module load anaconda3/2021.11
conda activate sbi_new
# Run your Python script
python3 evaluate_ensemble.py "$task_id" "$json_path"
