#!/usr/bin/env bash
# -*- encoding: utf-8 -*-

### Arguments list
### ----------------------------
# $CNT      :   simulation counter
# $RESDIR   :   results directory
# $FCONF    :   current configuration file
### ----------------------------

### Print current simulation start time [24H format]
CURRTIME=$(date +"%T")

echo "Running simulation $CNT with config file $FCONF at time $CURRTIME";

FNAME=$(basename $FCONF)
FNAME=$(echo $FNAME | cut -f 1 -d '.')

# python3 -c "from brian2 import *; clear_cache('cython');" > /dev/null 2>&1
command time -v python3 run_sim_dumb.py -p $FCONF -sd $OUTDIR > "$RESDIR/sim_${CNT}_${FNAME}.txt" 2>&1

CODE=$?
echo $CODE

RESULT=""
if [ $CODE = 0 ]; then
    RESULT="success"
else
    RESULT="failure"
fi

echo "Done + $RESULT"
echo "$CNT : $FCONF : $RESULT" >> "$RESDIR/total.txt"
