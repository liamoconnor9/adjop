#!/bin/bash

DIR="$(dirname "$(readlink -f "$0")")/"
qstat -u $USER > qstatout

function dispmenu() {
    echo "" 
    echo "SELECT FROM THE FOLLOWING:"
    echo "" 
    echo "1) VIEW OUTPUT / LOGGER"
    echo "2) OPEN CONFIG"
    echo "3) TWEAK CONFIG"
    echo "4) RESTART / FACTORY"
    echo "5) RUN INTERACTIVE"
    echo "6) PLOT"
    echo "7) KILL"
    echo "8) PURGE"
    echo "9) EXIT"
    echo ""
    read -p "" TASK
}

function dotask() {
    if [ $TASK -eq 1 ]; then
        if test -f "$suffix/logger.txt"; then
            eval $(code $suffix/logger.txt &> /dev/null &) &
        elif test -f "$suffix/output.txt"; then
            code $suffix/output.txt
        else
            echo "$suffix does not contain logger.txt or output.txt ..."
            exit 1
        fi
    elif [ $TASK -eq 2 ]; then
        code $suffix/$suffix.cfg
        exit 1
    elif [ $TASK -eq 3 ]; then
        bash CodeConfig.sh $suffix/$suffix.cfg
        exit 1
    elif [ $TASK -eq 4 ]; then
        bash JobFactory.sh $suffix/$suffix.cfg
        exit 1
    elif [ $TASK -eq 5 ]; then
        bash JobFactory.sh -f -d $suffix/$suffix.cfg
        exit 1
    elif [ $TASK -eq 6 ]; then
        echo "INPUT METRIC NAME"
        read -p "" name
        echo "INPUT FIG NAME"
        read -p "" figname
        echo "suffices = $suffices"
        echo "metric = $name"
        echo "fig = $figname"
        conda init bash
        conda activate dedalus3
        python3 plot_tracker.py $suffices $name $figname
        exit 1
    elif [ $TASK -eq 7 ]; then
        jobid=$($suffix/jobid)
        echo "HIT ENTER TO KILL JOB: $jobid"
        read -p ""
        qdel $jobid
        exit 1
    elif [ $TASK -eq 8 ]; then
        echo "HIT ENTER TO PURGE: $suffix"
        read -p ""
        bash Purge.sh $suffix
        bash menu.sh
    elif [ $TASK -eq 9 ]; then
        echo "EXITING..."
        exit 1
    else
        echo "unrecognized input"
    fi
}

function runmenu() {
    dispmenu
    dotask
}

function processSEL() {

    NEWLINENUM=1
    for d in */ ; do
        suffix=$(echo $d | sed 's:/*$::')
        if [[ "$suffix" == "__pycache__" ]] || [ "$suffix" == "cfgcache" ]; then
            continue
        fi
        pad=" "    
        if test -f "$DIR$suffix/$suffix.cfg"; then
            if [ $NEWLINENUM -eq $SEL ]; then
                echo ""
                break
            fi
        fi
        ((NEWLINENUM++))
    done
}

if ! [[ -z $1 ]]; then
    if test -d "$DIR$1/"; then
        suffix=$1
        runmenu
    else
        echo "arg menu.sh should be a run directory"
        exit 1
    fi
fi

while [[ 1 ]]; do

    echo ""
    echo "LISTING RUN DIRECTORIES:"
    echo ""

    LINENUM=1
    for d in */ ; do

        suffix=$(echo $d | sed 's:/*$::')
        pad=" "

        if (( $LINENUM > 9 )); then
            pad=""
        fi
        if [[ "$LINENUM" == "$SEL" ]]; then
            selsuffix=$suffix
        fi
    
        if test -f "$DIR$suffix/logger.txt"; then
            echo "[ D ]  ($LINENUM)$pad  $suffix"
        elif test -f "$suffix/$suffix.cfg"; then
            status=$(bash check.sh ${suffix} 1)
            echo "$status  ($LINENUM)$pad  $suffix"
        elif [[ "$suffix" == "__pycache__" ]] || [ "$suffix" == "cfgcache" ]; then
            # ((LINENUM++))
            continue
        else
            echo "$LINENUM)$pad   [MISSING .CFG]        $suffix"
        fi

        ((LINENUM++))

    done
    echo ""
    # echo "INPUT NUMBER TO SELECT RUN(S) OR PRESS ENTER TO SELECT MULTIPLE"
    # echo ""
    read -p "INPUT NUMBER TO SELECT RUN(S) OR PRESS ENTER TO SELECT MULTIPLE:   " SEL

    if [[ -z $SEL ]]; then
        # echo ""
        read -p "INPUT NUMBERS SEPERATED BY COMMAS:   " mult

        dispmenu

        sels=${mult//[[:blank:]]/}
        IFS=',' read -ra ADDR <<< "$sels"
        if [ $TASK -eq 6 ]; then
            suffices=""
            for SEL in "${ADDR[@]}"; do
                processSEL
                suffices="${suffices} ${suffix}"
            done

            dotask
        else
            for SEL in "${ADDR[@]}"; do
                processSEL
                dotask
            done
        fi
        exit 1
    fi

    processSEL
    runmenu
done