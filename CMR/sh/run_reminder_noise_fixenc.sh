#!/usr/bin/env bash
#SBATCH -J 'p0'
#SBATCH -o 00.out
#SBATCH -p all
#SBATCH -t 250
#SBATCH -c 2

module load anacondapy
. activate myenv

python3 /usr/people/qiongz/submisions/optimalfreerecall/CMR_prob/script_reminder_noise_fixenc.py --beta 90 --pos 0 --k 3 --enc 100

