#!/bin/bash

# $1 --> old_suffix
# $2 --> new_suffix

DIR="$(dirname "$(readlink -f "$0")")/"
if [ -z $2 ]; then
    echo "Insufficient args provided: $@"
    exit 1
    # CONFIG=${@: -1}
else
    old_suffix=$1
    new_suffix=$2
fi

if ! test -f $old_suffix/$old_suffix.cfg; then
    echo "Template directory $old_suffix/ does not contain config $old_suffix.cfg"
    exit 1

elif test -d $new_suffix; then
    echo "Directory named $new_suffix already exists!"
    echo "Press enter to overwrite?"
    read -p ""
    rm -rf $new_suffix
fi

mkdir $new_suffix
for filepath in $old_suffix/*.*; do
    filename=$(basename -- "$filepath")
    echo "copying file $filename"
    cp $old_suffix/$filename $new_suffix/$filename
done
for filepath in $old_suffix/*/; do
    filename=$(basename -- "$filepath")
    echo "making directory $new_suffix/$filename"
    mkdir $new_suffix/$filename
done

for filepath in $old_suffix/*/; do
    filename=$(basename -- "$filepath")

    count=$(find $filepath | wc -l)
    if [[ $filename == "checkpoints" ]] && [ $count -gt 100 ]; then
        echo $filepath has more than 50 files

        filecount=1
        for writefile in $( ls -r $filepath ); do
            echo copying $writefile
            cp $filepath/$writefile $new_suffix/checkpoints/
            ((filecount++))
            if (( filecount > 10 )); then
                break
            fi
        done

    elif [[ $filename == "snapshots_target" ]]; then
        echo "copying merged h5 files from snapshots_target.."
        cp  $old_suffix/$filename/*.h5 $new_suffix/$filename/
    else
        echo populating $filepath
        cp -r $old_suffix/$filename/* $new_suffix/$filename/
    fi

    # echo "populating directory $new_suffix/$filename"
    # cp -r $old_suffix/$filename/* $new_suffix/$filename/
done

if test -f $new_suffix/logger.txt; then
    echo "Renaming logger $new_suffix/logger.txt to $new_suffix/oldlogger.txt"
    mv $new_suffix/logger.txt $new_suffix/oldlogger.txt
else
    echo "No old logger found at path: $new_suffix/logger.txt"
fi

echo "Renaming config $new_suffix/$old_suffix.cfg to $new_suffix/$new_suffix.cfg"
mv $new_suffix/$old_suffix.cfg $new_suffix/$new_suffix.cfg

echo "Overwriting config parameter: from 'suffix=$old_suffix' to 'suffix=$new_suffix'"
sed -i -e "s/$old_suffix/$new_suffix/g" $new_suffix/$new_suffix.cfg

echo "Verifying..."
source $new_suffix/$new_suffix.cfg &>/dev/null

if [[ "$suffix" == "$new_suffix" ]]; then
    echo "Success! New run directory has been properly configured (I think)."
    bash CodeConfig.sh $new_suffix/$new_suffix.cfg
else
    echo "Yikes."
    echo "Looks like the new config $new_suffix/$new_suffix.cfg doesn't have suffix=$new_suffix"
fi