#!/bin/bash

filename='path_for_non_fit.txt'

while read i
do
	for j in {1..5}
	do
		~/Downloads/Programs/BusyFit-0.3.2/busyfit -c 1 2 -noplot "$i"
	done
done < "$filename"