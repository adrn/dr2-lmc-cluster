#!/bin/bash
#SBATCH -J isochrones                 # job name
#SBATCH -o isochrones.o%j             # output file name (%j expands to jobID)
#SBATCH -e isochrones.e%j             # error file name (%j expands to jobID)
#SBATCH -n 160                        # number of cores (not nodes!)
#SBATCH -p cca                        # add to the CCA queue
#SBATCH -t 12:00:00                   # run time (hh:mm:ss)

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/home/apricewhelan/software/lib/
cd /mnt/ceph/users/apricewhelan/projects/dr2-lmc-cluster/scripts

module load gcc openmpi2

date

srun python sample-stellar-params.py --mpi --ncores=160

date
