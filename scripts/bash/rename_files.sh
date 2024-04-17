#!/bin/bash
for f in results/*
do
	datevar=$(date -r "$f" +"%a_%b_%d_%Y_%H_%M_%S")
	datevar2=$(date -r "$f" +"%Y_%m_%d_%H_%M_%S_")
	fname=$(basename "$f")
	mv "$f" "results/$datevar2$fname"
done
