#!/usr/bin/env bash
# -*- coding: utf-8 -*-

RESDIR="./results"

if [ -d $RESDIR ]; then
	echo "Directory $RESDIR exists. Cleaning..."
	rm $RESDIR/*.txt
else
	echo "Directory $RESDIR does not exist. Creating..."
	mkdir $RESDIR
fi
echo "Creating txt output..."
touch "$RESDIR/total.txt"


CNT=0
RESULT=""
for FN_CONF in ./configs/*;
	do
		### Print current simulation start time [24H format] ###
		CURRDATE=$(date +'%T')
		
		echo "Running simulation $CNT with config file $FN_CONF at time $CURRDATE";
		
		python3 -c "from brian2 import *; clear_cache('cython');" > /dev/null 2>&1
		time python3 run_sim_dumb.py $FN_CONF > "./results/sim_res_$CNT.txt" 2>&1 && echo Done
		#python3 -run_sim_dumb.py > /dev/null 2>&1
		
		if [ "$?" = 0 ]; then
			RESULT="success"
		else
			RESULT="failure"
		fi
		
		echo "$CNT : $FN_CONF : $RESULT" >> "./results/total.txt"
		let "CNT+=1"
	done
