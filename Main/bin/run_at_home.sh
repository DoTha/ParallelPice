#!/bin/bash
mydate=$(date +%y_%m_%d_%H_%M_%S)
mpiexec -n 1 python $1.py $1 $mydate
echo copying $1.py to ../Data/binarchive/$1_$mydate.py
cp $1.py ../Data/binarchive/$1_$mydate.py
echo $1"_"$mydate >> ../Data/list_of_data.dat
echo "done"
