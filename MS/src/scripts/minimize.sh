#!/bin/sh

kind="solv_oct"

mpirun -np 7 sander -O -i minimize.in -o ../libs/1fk9_min_${kind}.out -p ../libs/1fk9_${kind}.prmtop -c ../libs/1fk9_${kind}.rst7  -r ../libs/1fk9_min_${kind}.ncrst
conda activate AmberTools21
ambpdb -p ../libs/1fk9_${kind}.prmtop -c ../libs/1fk9_min_${kind}.ncrst > ../libs/1fk9_min_${kind}.pdb
