#!/bin/bash

filename='path_list_reversed.txt'

while read i
do
	for j in {1..5}
	do
		~/Downloads/Programs/BusyFit-0.3.2/busyfit -c 1 2 -n rms -1 -noplot "$i"
	done
done < "$filename"
