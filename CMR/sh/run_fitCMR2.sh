#!/usr/bin/env bash
#SBATCH -J 'fitk-5n'
#SBATCH -o fitk.out
#SBATCH -p all
#SBATCH -t 1600
#SBATCH -c 2

module load anacondapy
. activate myenv

python3 /usr/people/qiongz/submisions/optimalfreerecall/CMR_prob/script_fitfullCMR2.py

