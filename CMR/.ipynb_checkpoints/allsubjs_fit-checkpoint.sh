#!/usr/bin/env bash
#SBATCH -J 'Z09_10'
#SBATCH -o Z09_10.out
#SBATCH -p all
#SBATCH -t 1200
#SBATCH -c 2

module load anacondapy
. activate myenv

python3 /usr/people/qiongz/submisions/optimalfreerecall/CMR_prob/main_fitCMRprob.py 

