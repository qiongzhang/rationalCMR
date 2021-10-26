#!/usr/bin/env bash
#SBATCH -J 'p0'
#SBATCH -o 00.out
#SBATCH -p all
#SBATCH -t 400
#SBATCH -c 2

module load anacondapy
. activate myenv

python3 /usr/people/qiongz/submisions/optimalfreerecall/CMR_prob/main_optimalCMRprob_2modes_2stage.py --primacy 0

