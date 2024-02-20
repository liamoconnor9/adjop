#!/bin/bash

function list () {
    while IFS='' read -d ' ' mtime && IFS='' read -r -d '' dirname; do
        outputmsg=""
        if [[ $dirname == "./__pycache__" ]]; then
            outputmsg=$(echo  "[$tag] $(date -r $dirname) $dirname")
        elif [[ $dirname == "./cfgcache" ]]; then
            outputmsg=$(echo  "[$tag] $(date -r $dirname) $dirname")
        elif [[ $dirname == "./savedruns" ]]; then
            outputmsg=$(echo  "[$tag] $(date -r $dirname) $dirname")
        elif [[ $dirname == "./PLT"* ]]; then
            outputmsg=$(echo  "[$tag] $(date -r $dirname) $dirname")
        fi

        if [[ $tag == "SKIPPED" ]] && [[ $outputmsg == "" ]]; then
            continue
        elif [[ $tag == "SKIPPED" ]]; then
            echo $outputmsg
        elif [[ $outputmsg == "" ]]; then
            outputmsg=$(echo "[$tag] $(date -r $dirname) $dirname")
            if $cacheruns; then
                echo "CACHING $outputmsg"
		rm -rf savedruns/$dirname
                mv $dirname savedruns/
            else
                echo $outputmsg
            fi
        fi
    done < <(find . -maxdepth 1 -mindepth 1 $timeoption -type d -printf '%T@ %p\0' | sort -z)
}

cacheruns=false
if test -z $1; then
    grace_days=20
else
    grace_days=$1
fi

echo "SKIPPED DIRECTORIES:"
timeoption=""
tag="SKIPPED"
list

timeoption=$(echo "-mtime -${grace_days}")
tag="INCLUDE"
echo ""
echo "RECENT RUNS WITHIN $grace_days DAY GRACE PERIOD:"
list

timeoption=$(echo "-mtime +${grace_days}")
tag="OLDERUN"
echo ""
echo "OBSOLETE RUNS OLDER THAN $grace_days DAY GRACE PERIOD:"
list
nruns=$(list | wc -l)

echo ""
read -p "PRESS ENTER TO MOVE THESE $nruns RUNS TO ./savedruns"
cacheruns=true
list 

# echo "yep"

# while IFS='' read -d ' ' mtime && IFS='' read -r -d '' dirname; do
#     if [[ $dirname == "./__pycache__" ]]; then
#         echo "[SKIPPED] $(date -r $dirname) $dirname"
#     elif [[ $dirname == "./savedruns" ]]; then
#         echo "[SKIPPED] $(date -r $dirname) $dirname"
#     elif [[ $dirname == "./PLT"* ]]; then
#         echo "[SKIPPED] $(date -r $dirname) $dirname"
#     else
#         echo "[INCLUDE] $(date -r $dirname) $dirname"
#         # printf '%(%FT%T%z)T\n' $mtime
#         # echo $(date -d $mtime)
#     fi
# done < <(find . -maxdepth 1 -mindepth 1 -mtime -${grace_days} -type d -printf '%T@ %p\0' | sort -z)

#   printf 'Processing file %q with timestamp of %s\n' "$dirname" "$mtime"
# for dirname in `ls -t */`; do
#     if [[ $dirname == "__pycache__/" ]]; then
#         echo "[SKIPPED] $dirname"
#     elif [[ $dirname == "PLT"* ]]; then
#         echo "[SKIPPED] $dirname"
#     else
#         echo "[INCLUDE] $dirname"
#     fi
# done

# RUNCOUNT=1
# tac JOBLOG.txt | while read -r line; do
#     if [[ $line == *"JOBNAME"* ]]; then
#         name=$line
#         ((RUNCOUNT++))
#         suffix=${line:12}
#         code $suffix/output.txt
#     fi
#     if (( RUNCOUNT > 5 )); then
#         break
#     fi
#     echo "$line"
#     # COMMAND using line
# done
