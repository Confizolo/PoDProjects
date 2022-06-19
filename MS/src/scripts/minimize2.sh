#!/bin/sh

mol="chain_b_"
kind="opt_pbc"

mpirun -np 3 sander -O -i min.in -o ../out/${mol}${kind}_min.out -p ../data/${mol}${kind}.prmtop -c ../data/${mol}${kind}.rst7  -r ../res/${mol}${kind}_min.ncrst
ambpdb -p ../data/${mol}${kind}.prmtop -c ../res/${mol}${kind}_min.ncrst > ../res/${mol}${kind}_min.pdb
