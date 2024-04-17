#!/usr/bin/env bash
# -*- encoding: utf-8 -*-

# Make directories
# TODO: results directories per phase
echo "[+] Making directories..."

# Phase 1: Noise opt
# ------------------
OUTDIR="results_opt_phase1"
JVAL="1000.0"
THRESHOLD="40.0"
CNT=0
TARGET=(1 1 1 1 1 1 1 1 1 6)
while [ "$(bc <<< "$JVAL > $THRESHOLD")" == "1" ] # while [ $JVAL -ge $THRESHOLD ]
do
    # P0: Generate a config file
    echo "[+] Generating configuration file"
    FCONF="confname.json"
    NOISE=(20 20 20 20)
    echo "python parameters_opt.py --noise-list $(echo ${NOISE[@]})" # TODO: add arguments for noise here [EC, DG, CA3, CA1]

    # P1: Run the simulation using the file
    echo "[*] Running simulation..."
    echo "command time -v python3 run_sim_dumb.py -p $FCONF -sd $OUTDIR > "$RESDIR/sim_${CNT}_${FNAME}.txt" 2>&1"
    echo "[*] Simulation done"

    # P2: TODO: Find the most recent results directory
    # RDIR=$(ls -td -- $(echo "$OUTDIR/None/") | head -n 1)
    # echo $RDIR

    # P3: Run the optimization python code
    echo "python3 optimization.py --target-vector $(echo ${TARGET[@]})" # TODO: add the goal vector and the location of the simulation results

    # P4: Read the J value and the FR values per area
    # tag=$( tail -n 1 optimization_CA1.csv | cut -d ',' -f 3-8 )
    JVAL="$( tail -n 1 optimization_CA1.csv | cut -d ',' -f4 )"
    # JVAL=$((JVAL-990))
    echo $JVAL

    # Update counter
    CNT=$((CNT+1))
done

exit 0

# Phase 2: Input opt
# ------------------
OUTDIR="results_opt_phase2"
JVAL=1000
THRESHOLD=40
while [ $JVAL -gt $THRESHOLD ]
do
    # P0: Generate a config file
    FCONF="confname"
    python3 parameters.py # TODO: add arguments for noise here [EC, DG, CA3, CA1]

done


# Phase 3: Connections opt
# ------------------------
OUTDIR="results_opt_phase3"
JVAL=1000
THRESHOLD=40
while [ $JVAL -gt $THRESHOLD ]
do
    # P0: Generate a config file
    FCONF="confname"
    python3 parameters.py # TODO: add arguments for noise here [EC, DG, CA3, CA1]

done
