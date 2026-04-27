#!/bin/bash
#SBATCH -J biornn_runner
#SBATCH -N 1 --ntasks-per-node=16
#SBATCH --gres=gpu:1 
#SBATCH -t7:00:00
#SBATCH -ojob_reports/biornn_runner-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=avavilikolanu3@gatech.edu

cd $SLURM_SUBMIT_DIR

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate hml-lab1
which python
python --version

python BioRNN_RichLazy/runner.py --task_mode=2AF --task=CXT --var_name=rr