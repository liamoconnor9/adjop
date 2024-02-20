#!/bin/bash

DIR="$(dirname "$(readlink -f "$0")")/"

while [[ 1 ]]; do

    echo ""
    echo "LISTING ACTIVE JOBS:"
    echo ""

    qstat -u $USER > qstatout


    LINENUM=-2
    while read -r line; do
        name="$line"
        if (( LINENUM < 1 )); then
            echo "      $name"
        elif (( LINENUM > 9 )); then
            echo "$LINENUM)   $name"
        else
            echo "$LINENUM)    $name"
        fi
        ((LINENUM++))
    done < qstatout
    echo ""
    echo "INPUT NUMBER TO SELECT JOB:"
    read -p "" SEL

    LINENUM=-2
    while read -r line; do
        name="$line"

        if [ $LINENUM -eq $SEL ]; then
            echo ""
            echo "$LINENUM)   $name"
            break
        fi

        ((LINENUM++))
    done < qstatout

    # echo $PBS_QUEUE
    # echo $PBS_JOBNAME


    LINENUM=1
    for d in $DIR*/ ; do
        suffix=$(basename $d)

        if [[ $name == *" $suffix "* ]]; then
            # echo "SELECTED RUN: $suffix"
            bash ${DIR}menu.sh $suffix;
            break
        fi
    done
done