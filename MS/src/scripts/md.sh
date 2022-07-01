#!/bin/sh

mol="chain_a_efz_"
kind="solv_box"


mpirun -np 8 sander -O -i md_pbc.in -o ../out/${mol}${kind}_md.out -p ../res/${mol}${kind}.prmtop -c ../res/${mol}${kind}_heatpbc.rst  -r ../res/${mol}${kind}_md.rst -x ../res/${mol}${kind}_md.nc
