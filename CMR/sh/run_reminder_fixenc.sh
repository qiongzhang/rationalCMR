#!/usr/bin/env bash
#SBATCH -J 'B-b6p10'
#SBATCH -o 00.out
#SBATCH -p all
#SBATCH -t 240
#SBATCH -c 2

module load anacondapy
. activate myenv

python3 /usr/people/qiongz/submisions/optimalfreerecall/CMR_prob/script_reminder_fixenc.py --pos 10 --enc 60

