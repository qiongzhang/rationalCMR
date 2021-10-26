#!/usr/bin/env bash
#SBATCH -J 'p9'
#SBATCH -o 00.out
#SBATCH -p all
#SBATCH -t 600
#SBATCH -c 2

module load anacondapy
. activate myenv

python3 /usr/people/qiongz/submisions/optimalfreerecall/CMR_prob/main_optimalCMRprob_rep_primacy.py --primacy 9

