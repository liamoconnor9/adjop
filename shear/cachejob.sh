#PBS -S /bin/bash
#PBS -l select=1:ncpus=28:mpiprocs=28:model=bro
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -W group_list=s2276
file=${0##*/}
job_name="${file%.*}"
#PBS -N ${job_name}

bash Sort.sh