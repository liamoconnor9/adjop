#!/bin/bash

arg=0
if [ -z "$1" ]
    then
        arg=1
fi

python3 multi.py 1 200 -2 6 $arg
python3 multi.py 0 200 0 6 $arg

# echo "###############################################"
# echo "PLOTTTING SBI INITIAL GUESS RESULTS"
# echo "###############################################"

# python3 multi.py 1 200 -3 6 $arg
# python3 multi.py 0 200 0 6 $arg