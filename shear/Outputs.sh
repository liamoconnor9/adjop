#!/bin/bash

RUNCOUNT=1
tac JOBLOG.txt | while read -r line; do
    if [[ $line == *"JOBNAME"* ]]; then
        name=$line
        ((RUNCOUNT++))
        suffix=${line:12}
        code $suffix/output.txt
    fi
    if (( RUNCOUNT > 5 )); then
        break
    fi
    echo "$line"
    # COMMAND using line
done