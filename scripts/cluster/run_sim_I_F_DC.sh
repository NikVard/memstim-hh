#!/usr/bin/env bash
# -*- encoding: utf-8 -*-

### Print current simulation start time [24H format]
CURRTIME=$(date +"%T")

echo "Running simulation at time $CURRTIME";

command time -v python3 simulate_I_F_curves_CA1.py -p configs/LFP_test_sim_DC.json -sd /beegfs/nvardala/results_I_F_CA1_DC_PR > /beegfs/nvardala/results_I_F_CA1_DC_PR/sim_I_F_CA1.txt 2>&1

CODE=$?
echo $CODE

RESULT=""
if [ $CODE = 0 ]; then
    RESULT="success"
else
    RESULT="failure"
fi

touch /beegfs/nvardala/results_I_F_CA1_DC_PR/result.txt
echo "Done + $RESULT" >> /beegfs/nvardala/results_I_F_CA1_DC_PR/result.txt
