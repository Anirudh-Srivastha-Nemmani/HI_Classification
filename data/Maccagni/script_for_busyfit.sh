#!/bin/bash

filename='path_list_reversed.txt'
exception='spectraArcHIve_reversed/J135217.88+312646.4_spec.txt'

while read i
do
	for j in {1..5}
	do
		
		if [ "$i" == "$exception" ]
	       	then
			~/Downloads/Programs/BusyFit-0.3.2/busyfit -c 1 2 -n rms -1 -noplot "$i"
		else
			~/Downloads/Programs/BusyFit-0.3.2/busyfit -c 1 3 -n rms 4 -noplot "$i"
		fi
	done
done < "$filename"
