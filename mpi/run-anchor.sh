#!/bin/bash
#SBATCH -J anchor                 # job name
#SBATCH -o anchor.o%j             # output file name (%j expands to jobID)
#SBATCH -e anchor.e%j             # error file name (%j expands to jobID)
#SBATCH -n 1                      # number of cores (not nodes!)
#SBATCH -p cca                    # add to the CCA queue
#SBATCH -t 04:00:00               # run time (hh:mm:ss)

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/home/apricewhelan/software/lib/
cd /mnt/ceph/users/apricewhelan/projects/dr2-lmc-cluster/scripts

module load gcc

date

python run_anchor.py

date
