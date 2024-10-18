#!/bin/bash
file=${0##*/}
job_name="${file%.*}"
DEBUG=false
# RUNIT=true
if [ ! -z $1 ] 
then 
    if [ "$1" = "-q" ]; then
        RUNIT=false
        echo $RUNIT
    else
        if [ "$1" = "-d" ]; then
            RUNIT=false
            DEBUG=true
            # echo $RUNIT
        else
            echo "unrecognized argument. exitting..."
            exit 1
        fi
    fi
else
    RUNIT=true
    DEBUG=false
    echo $RUNIT
fi

run_func () {
    sleep 1
    if $RUNIT ; then
        echo "running!!!"
        bash $TEMPLATE
    else

        echo "queuing!!!"
        echo "please enter a message for the JOBLOG"
        read messg
        if $DEBUG ; then
            echo "DEBUGG!"
            OUT_SBATCH=$(sbatch -t $walltime --nodes=$nodes  --ntasks=$MPIPROC  $TEMPLATE )
            # OUT_QSUB=$(qsub -q debug -o logger.txt -v "suffix=$suffix" -l select=$nodes:ncpus=$ncpus:mpiprocs=$ncpus:model=$model -l walltime=$walltime -N "$suffix" $TEMPLATE)
        else
            OUT_SBATCH=$(sbatch -t $walltime --nodes=$nodes  --ntasks=$MPIPROC  $TEMPLATE )
            # OUT_QSUB=$(qsub -o logger.txt -v "suffix=$suffix" -l select=$nodes:ncpus=$ncpus:mpiprocs=$ncpus:model=$model -l walltime=$walltime -N "$suffix" $TEMPLATE)
        fi

        TIMESTAMP="$(date '+%d/%m/%Y %H:%M:%S')"
        echo ""
        echo "TIMESTAMP:  ${TIMESTAMP}"
        echo "JOBID:    ${OUT_BATCH}"
        echo "JOBNAME:  ${suffix}"

        echo "TIMESTAMP:  ${TIMESTAMP}"   >>  ~/JOBLOG.txt
        echo "JOBID:      ${OUT_BATCH}"    >>  ~/JOBLOG.txt
        echo "JOBNAME:    ${suffix}"      >>  ~/JOBLOG.txt
        echo "MESSAGE:    ${messg}"       >>  ~/JOBLOG.txt
    fi
}


export PATH=$HOME/scripts:$PATH
deactivate
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate dedalus3
# support lots of text output to stdio for analysis
export MPI_UNBUFFERED_STDIO=true

cd /home/lol7821/adjop/mri

FILE="$(readlink -f "$0")"
DIR="$(dirname "$(readlink -f "$0")")/"
CONFIG="new_config.cfg"
LOAD_IC="load_ic.py"
# SOLVER="mri.py"
TEMPLATE="template.sh"

source $CONFIG
echo "SUITE SUFIX SUPPLIED: $suffix"
# if [ ! -e "$load_cp" ]; then
#     if [ ! "$load_cp" = "default" ]; then
#         echo "DIRECTORY $suffix DOES NOT EXIST!!"
#         echo "cant load data that DNE bruh"
#         exit 1
#     else
#         echo "constructing default initial condition"
#     fi
# else 
#     echo "loading checkpoint"
#     echo $load_cp
# fi 

if [ -d "$suffix" ]; then
    echo "DIRECTORY $suffix ALREADY EXISTS!!"
    echo "PRESS ENTER TO OVERWRITE EXISTING SIMULATION SUITE"
    read -p " "
    rm -rf $suffix
fi


mkdir $suffix
# mkdir $suffix/data
cp $CONFIG $suffix
cp $SOLVER $suffix
cp $TARGET $suffix
cp $TEMPLATE $suffix
cd $suffix
ls

# sed -i "/#SBATCH -p ciera-std/ i #SBATCH --job-name $suffix " /home/lol7821/adjop/mri/$suffix/$TEMPLATE

run_func
exit 1