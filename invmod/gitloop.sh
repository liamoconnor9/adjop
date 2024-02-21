#!/bin/bash

while [[ true ]]; do 
    git add . --all
    git commit -m 'live updates'
    git pull
    git push
    sleep 20
done
