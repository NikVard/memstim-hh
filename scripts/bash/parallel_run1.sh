#!/usr/bin/env bash
# -*- encoding: utf-8 -*-

### -------------------------------------------------------------- ###
# ARG $1 is the configuration file to be used; ARG $2 is the current count
run_sim() {
    FNAME=$(basename $1)
    FNAME=$(echo $FNAME | cut -f 1 -d '.')

    # python3 -c "from brian2 import *; clear_cache('cython');" > /dev/null 2>&1
	time python3 run_sim_dumb.py -p $1 > "$RESDIR/sim_$2_${FNAME}.txt" 2>&1 && echo 'Done'
	
	### Copy the config file used to results dir
	cp $1 $BACKCONF
	
	if [ "$?" = 0 ]; then
		RESULT="success"
	else
		RESULT="failure"
	fi

	echo "Simulation $2 ended - $RESULT"
	echo "$2 : $FN_CONF : $RESULT" >> "$RESDIR/total.txt"
}
### -------------------------------------------------------------- ###

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

### Clear Brian2 cache
python3 -c "from brian2 import *; clear_cache('cython');" > /dev/null 2>&1

### Run all simulations in parallel jobs
CNT=0
RESULT=""
for FN_CONF in ./configs/*;
	do
		### Current simulation start time [24H format] ###
		CURRTIME=$(date +"%T")

		echo "Running simulation $CNT with config file $FN_CONF at time $CURRTIME";
        
		# run simulation code here, detach, store PID in array
		run_sim "$FN_CONF" "$CNT" &
		PIDS[$CNT]=$!
		let "CNT+=1"
		
		echo "PID of simulation $CNT: $!"
		sleep 2
	done


# multi-wait here for all PIDs to end, append to "total.txt"
for pid in ${PIDS[*]}; do
    wait $pid
done
