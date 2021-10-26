#!/usr/bin/env bash
#SBATCH -J 'fitb'
#SBATCH -o fitb.out
#SBATCH -p all
#SBATCH -t 900
#SBATCH -c 2

module load anacondapy
. activate myenv

python3 /usr/people/qiongz/submisions/optimalfreerecall/CMR_prob/script_fitCMRprob.py

