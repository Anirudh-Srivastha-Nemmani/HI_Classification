#!/bin/bash

filename='path_list_reversed.txt'
exception='spectraArcHIve_reversed/J135217.88+312646.4_spec.txt'

while read i
do
	if [ "$i" == "$exception" ]
	then
		~/Downloads/Programs/BusyFit-0.3.2/busyfit -c 1 2 "$i"
	else
		~/Downloads/Programs/BusyFit-0.3.2/busyfit -c 1 3 "$i"
	fi
done < "$filename"
