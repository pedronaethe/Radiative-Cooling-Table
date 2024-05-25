#!/bin/bash

# Creating directories
mkdir -p build
mkdir -p tables
mkdir -p parameters

# Compile parameters.c and with the executable to the build directory
echo "Compiling parameters..."
gcc src/parameters.c -o build/parameters -lm
echo "Done!"

# Run parameters to generate parameter files
echo "Building parameters..."
cd build
./parameters
cd ..
echo "Done!"

# Compile cooling_table_OpenMPI.c and move the executable to the build directory
echo "Compiling table generation code..."
mpicc -o build/cooling_mpi src/cooling_table_OpenMPI.c -lm -fopenmp
echo "Done!"

# Get the number of processes from command line argument, default to 1 if not provided
if [ -z "$1" ]; then
    num_processes=1
else
    num_processes=$1
fi

# Run MPI program
echo "Running program with $num_processes MPI processes."

mpirun -np $num_processes build/cooling_mpi
echo "Done!"
