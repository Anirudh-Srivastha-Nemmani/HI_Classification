#!/bin/bash

filename='list_intervening.txt'

while read i
do
	~/Downloads/Programs/BusyFit-0.3.2/busyfit -c 1 2 -noplot "$i"
done < "$filename"