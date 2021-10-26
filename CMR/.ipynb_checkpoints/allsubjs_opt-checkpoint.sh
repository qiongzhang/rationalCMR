#!/usr/bin/env bash
#SBATCH -J 'p10_75'
#SBATCH -o 00_.out
#SBATCH -p all
#SBATCH -t 200
#SBATCH -c 2

module load anacondapy
. activate myenv

python3 /usr/people/qiongz/submisions/optimalfreerecall/CMR_prob/main_optimalCMRprob_2modes.py --pos 10

