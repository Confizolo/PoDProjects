#!/bin/sh

mol="chain_b_efz_"
kind="solv_box"

mpirun -np 3 sander -O -i min_pbc.in -o ../out/${mol}${kind}_min.out -p ../res/${mol}${kind}.prmtop -c ../res/${mol}${kind}.rst7  -r ../res/${mol}${kind}_min.ncrst
ambpdb -p ../res/${mol}${kind}.prmtop -c ../res/${mol}${kind}_min.ncrst > ../res/${mol}${kind}_min.pdb
