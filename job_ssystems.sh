#!/bin/bash
#SBATCH --array=0-739
#SBATCH --cpus-per-task=16
#SBATCH --mem=36G
#SBATCH --time=01:00:00
#SBATCH --job-name=expes_rashomon
#SBATCH -o slurm_out/slurmout_%A.out
#SBATCH -e slurm_out/slurmout_%A.errarray

python3.10 ssystems/main_scoring.py --expe_id=$SLURM_ARRAY_TASK_ID