#!/bin/sh


mol="chain_b_efz_"
kind="solv_box"

mpirun -np 6 sander -O -i heat_pbc.in -o ../out/${mol}${kind}_heatpbc.out -p ../res/${mol}${kind}.prmtop -c ../res/${mol}${kind}_min.ncrst  -r ../res/${mol}${kind}_heatpbc.rst -x ../res/${mol}${kind}_heatpbc.nc 
