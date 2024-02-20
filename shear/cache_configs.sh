#!/bin/bash

cache="cfgcache"
mkdir $cache &>/dev/null
echo "caching old configs:"

for i in *.cfg; do
    [ -f "$i" ] || break
    
    SUB='benchmark'
    DFLT='default.cfg'
    LOAD='shear_load_options.cfg'
    if [[ "$i" == *"$SUB"* ]]; then
        echo "skipped from cache $i"
    elif [[ "$i" == "default.cfg" ]] || [[ "$i" == "new_config.cfg" ]]; then
        echo "skipped from cache $i"
    else
        echo "cached $i to $cache/"
        mv $i $cache/$i
    fi
done