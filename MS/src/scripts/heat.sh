#!/bin/sh

sander -O -i heat.in -o hivrt_efz_heat.out -p ../P2/hivrt_efz.prmtop -c ../P2/hivrt_efz_min.ncrst  -r hivrt_efz_heat.rst -x hivrt_efz_heat.nc &
