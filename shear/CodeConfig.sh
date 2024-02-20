#!/bin/bash

DIR="$(dirname "$(readlink -f "$0")")/"
template=$1
if [[ -z $template ]]; then
    echo "SPECIFY TEMPLATE CONFIG OR DIRECTORY!"
    echo "using new_config.cfg"
    template="new_config.cfg"

elif test -d "$DIR$template/"; then
    template=$template/$template.cfg
fi

echo "COPYING $template"
cp $template new_config.cfg

# if [[ "$PBS_JOBNAME" == "STDIN" ]]; then
if ! command -v code &> /dev/null ;
then
    echo "<the_command> could not be found"
    echo "VSCODE COMMAND LINE INTERFACE INACCESSIBLE, USING VIM (GOOD LUCK) ... "
    read -p ""
    vim new_config.cfg

else
    code new_config.cfg
fi


echo "HIT ENTER TO SEND JOB USING: new_config.cfg"
read -p ""
# source new_config.cfg &>/dev/null
# echo ""
# echo "SUFFIX READ FROM NEW CONFIG: $suffix"
# cp new_config.cfg $suffix.cfg

args=""
if [[ "$PBS_QUEUE" == "devel" ]]; then
    echo "ENTER ARGS FOR DEVEL?"
    read -p "" args
fi

echo ""
# echo "PRESS ENTER TO CONTINUE TO JOB FACTORY W/ ARGS: $args"
# read -p "" 
bash JobFactory.sh $args new_config.cfg
