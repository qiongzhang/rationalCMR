#!/usr/bin/env bash
#SBATCH -J 'p0'
#SBATCH -o 00.out
#SBATCH -p all
#SBATCH -t 200
#SBATCH -c 2

module load anacondapy
. activate myenv

python3 /usr/people/qiongz/submisions/optimalfreerecall/CMR_prob/script_primacy.py --primacy 0

