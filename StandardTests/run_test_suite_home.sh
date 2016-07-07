#!/bin/bash
mydate=$(date +%y_%m_%d_%H_%M_%S)
mpiexec -n 2 python test_1D_sequential.py test_1D_sequential $mydate
mpiexec -n 2 python test_2D_sequential.py test_2D_sequential $mydate
mpiexec -n 2 python test_1D_parallel.py test_1D_parallel $mydate
mpiexec -n 2 python test_2D_parallel.py test_2D_parallel $mydate
echo "done"
