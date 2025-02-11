#!/bin/bash
#SBATCH --job-name=sbp_bl
#SBATCH --output=/home/at8471/c2_sbi_experiments/sbi_framework/%x_%j.out
#SBATCH --error=/home/at8471/c2_sbi_experiments/sbi_framework/%x_%j.err
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=16      # total number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=06:00:00
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=ms6730@princeton.edu
json_path='/home/at8471/c2_sbi_experiments/hydrogen-sbi/scripts/settings.json'
module purge # machine learning 
module load anaconda3/2021.11 # environment version 
conda activate sbi_new  # activate the environment
# Set up and do baseline run
python3 setup_experiment.py "$json_path"

