#!/bin/bash

OLDSUFFIX=$1
NEWSUFFIX=$2
DIR="$(dirname "$(readlink -f "$0")")/"

echo "copying previous study $OLDSUFFIX into new study $NEWSUFFIX directory"
cp $DIR$OLDSUFFIX/checkpoints/* $DIR$NEWSUFFIX/checkpoints/
mkdir $DIR$NEWSUFFIX/checkpoint_target/
cp $DIR$OLDSUFFIX/checkpoint_target/*.h5 $DIR$NEWSUFFIX/checkpoint_target/
cp $DIR$OLDSUFFIX/tracker.pick $DIR$NEWSUFFIX/tracker.pick
cp $DIR$OLDSUFFIX/output.txt $DIR$NEWSUFFIX/output.txt
