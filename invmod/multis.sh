#!/bin/bash

# for filename in appy/*/; do
# 	echo "plotting $filename"
# 	python3 multi.py $filename
# done

# for filename in qol/*/; do
#     echo "plotting $filename"
#     python3 multi.py $filename
# done

for filename in smol/*/; do
    echo "plotting $filename"
    python3 multi.py $filename
done
