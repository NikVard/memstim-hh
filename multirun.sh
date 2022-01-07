#!/usr/bin/env bash
# -*- coding: utf-8 -*-

### Get the current date and start making the results directory
CURRDATE=$(date +"%FT%H%M")
RESBASE="./results"
RESDIR="$RESBASE"_"$CURRDATE"

### Check and make directory for saving results
if [ -d $RESDIR ]; then
	echo "Directory $RESDIR exists. Cleaning..."
	rm $RESDIR/*.txt
else
	echo "Directory $RESDIR does not exist. Creating..."
	mkdir $RESDIR
fi
echo "Creating txt output..."
touch "$RESDIR/total.txt"

### Make a subdirectory to store the processed used configuration files
BACKCONF="$RESDIR/configs"
if ! [ -d $BACKCONF ]; then
	echo "Directory $BACKCONF does not exist. Creating..."
	mkdir $BACKCONF
fi


CNT=0
RESULT=""
for FN_CONF in ./configs/*;
	do
		### Print current simulation start time [24H format] ###
		CURRTIME=$(date +"%T")

		echo "Running simulation $CNT with config file $FN_CONF at time $CURRTIME";

        FNAME=$(basename $FN_CONF)
        FNAME=$(echo $FNAME | cut -f 1 -d '.')

        python3 -c "from brian2 import *; clear_cache('cython');" > /dev/null 2>&1
		time python3 run_sim_dumb.py -p $FN_CONF > "$RESDIR/sim_${CNT}_${FNAME}.txt" 2>&1
		#python3 -run_sim_dumb.py > /dev/null 2>&1
        CODE=$? ; echo 'Done'

		if [ $CODE = 0 ]; then
			RESULT="success"
		else
			RESULT="failure"
		fi

		echo "$CNT : $FN_CONF : $RESULT" >> "$RESDIR/total.txt"
		let "CNT+=1"

		### Copy the config file used to results dir
		cp $FN_CONF $BACKCONF
	done
