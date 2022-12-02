#!/bin/bash

filename='list_intervening_final.txt'

while read i
do
	~/Downloads/Programs/BusyFit-0.3.2/busyfit -c 1 2 "$i"
done < "$filename"