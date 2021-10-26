#!/usr/bin/env bash
#SBATCH -J 'p9'
#SBATCH -o 00.out
#SBATCH -p all
#SBATCH -t 500
#SBATCH -c 2

module load anacondapy
. activate myenv

python3 /usr/people/qiongz/submisions/optimalfreerecall/CMR_prob/script_primacy_twostage.py --primacy 5

