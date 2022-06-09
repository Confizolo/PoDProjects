#!/bin/sh

mol="efz"
kind="_solv_oct"

mpirun -np 2 sander -O -i minimize.in -o ../libs/${mol}_min${kind}.out -p ../libs/${mol}${kind}.prmtop -c ../libs/${mol}${kind}.rst7  -r ../libs/${mol}_min${kind}.ncrst
conda activate AmberTools21
ambpdb -p ../libs/${mol}${kind}.prmtop -c ../libs/${mol}_min${kind}.ncrst > ../libs/${mol}_min${kind}.pdb
