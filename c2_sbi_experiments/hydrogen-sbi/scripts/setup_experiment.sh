#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --output=/home/at8471/c2_sbi_experiments/sbi_framework/%x_%j.out
#SBATCH --error=/home/at8471/c2_sbi_experiments/sbi_framework/%x_%j.err
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1     # total number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=01:00:00
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

json_path='/home/at8471/c2_sbi_experiments/hydrogen-sbi/scripts/settings.json' 

# Set up and do baseline run
module purge                     # machine learning purposes 
module load anaconda3/2021.11    # to manage software packages and environments 
conda activate sbi_new           # to activate the environment 
python3 setup_experiment.py "$json_path"
