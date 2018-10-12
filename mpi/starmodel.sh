#!/bin/bash
#SBATCH -J starmodel                 # job name
#SBATCH -o starmodel.o%j             # output file name (%j expands to jobID)
#SBATCH -e starmodel.e%j             # error file name (%j expands to jobID)
#SBATCH -n 112
#SBATCH -t 12:00:00             # run time (hh:mm:ss)

cd

module load gcc openmpi2

source activate scf

date

srun python field-of-streams.py -f ../../data/cosmohub-b5.fits -v --nworkers=1024 --mpi

date
