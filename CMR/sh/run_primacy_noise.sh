#!/usr/bin/env bash
#SBATCH -J 'k3p0'
#SBATCH -o 00.out
#SBATCH -p all
#SBATCH -t 240
#SBATCH -c 2

module load anacondapy
. activate myenv

python3 /usr/people/qiongz/submisions/optimalfreerecall/CMR_prob/script_primacy_noise.py --primacy 0 --k 3 --enc 100

