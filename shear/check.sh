#!/bin/bash

DIR="$(dirname "$(readlink -f "$0")")/"
suffix=$1

if [[ -z $2 ]]; then
    qstat -u $USER > qstatout
fi



LINENUM=-2
while read -r line; do
    name="$line"
    if [[ $name == *" $suffix "* ]]; then
        if [[ $name == *" Q "* ]]; then
            echo "[ Q ]"
            exit 1
        elif [[ $name == *" R "* ]]; then
            echo "[ R ]"
            exit 1
        fi
        # bash ${DIR}menu.sh $suffix;
        break
    fi

    ((LINENUM++))
done < qstatout

echo "[IDK]"
