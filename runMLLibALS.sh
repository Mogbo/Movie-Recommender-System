#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=proj_%j
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=general
#SBATCH --mem=100Gb
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ojuba.e/Project/logs/exp_%j.out
#SBATCH --error=/scratch/ojuba.e/Project/logs/exp_%j.err

python -u MatrixFactorizationALS.py --saveTo /scratch/ojuba.e/Project/results/baseResults2_noNeg.csv