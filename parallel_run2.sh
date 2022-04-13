#!/usr/bin/env bash
# -*- encoding: utf-8 -*-

### Get the current date and start making the results directory
CURRDATE=$(date +"%FT%H%M")
RESBASE="./results_opt"
RESDIRTOT="$RESBASE"_"$CURRDATE"


### Check and make directory for saving ALL the results
if [ -d $RESDIRTOT ]; then
	echo "Directory $RESDIRTOT exists. Cleaning..."
	rm $RESDIRTOT/*.txt
else
	echo "Directory $RESDIRTOT does not exist. Creating..."
	mkdir $RESDIRTOT
fi


### Get all config directories
#CONF_DIRS=$(find ./configs/ -maxdepth 1 -mindepth 1 -type d)
# ISTIM=$1
# CONF_DIRS="configs/${ISTIM}_nA"
CONF_DIRS="configs/opt_noise_EC"

### Go through the config directories and do the following:
for DIR in $CONF_DIRS;
    do
        ### Store the basename for the parameters i.e. 10nA
        PNAME=$(basename $DIR)
        echo "In $PNAME"

        ### 1. Make a subdirectory to store the used configuration files for this parameter set i.e. results_{date}/10nA/...
        CURRRES="$RESDIRTOT/$PNAME"
        if ! [ -d $CURRRES ]; then
        	echo "Directory $CURRRES does not exist. Creating..."
        	mkdir $CURRRES
        fi

        BACKCONF="$CURRRES/configs"
        if ! [ -d $BACKCONF ]; then
        	echo "Directory $BACKCONF does not exist. Creating..."
        	mkdir $BACKCONF
        fi


        ### 2. Make the current config set output logs
        echo "Creating txt output for current config set..."
        touch "$CURRRES/total.txt"


        ### 3. Queue the batch script
        ### Set the variables used as arguments
        CNT=0
        RESDIR=$CURRRES
        for FN_CONF in $DIR/*.json;
                do
                echo $FN_CONF

                ### Queue the simulation
                sbatch --job-name="SIM_${ISTIM}_${FN_CONF}" --time=60 --cpus-per-task=1 --mem-per-cpu=16G --ntasks=1 --export=FCONF=$FN_CONF,RESDIR=$RESDIR,CNT=$CNT run_sim.sh

                let "CNT+=1"

                sleep 1

                ### Copy the config file used to results dir
                cp $FN_CONF $BACKCONF
            done
    done
