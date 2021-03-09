#!/bin/bash
cd UART_T 
cp *.vhd ../
cd ../UART_R 
cp *.vhd ../
cd ../
ghdl -a *.vhd
ghdl -e testbench
ghdl -r testbench  --wave=wave.ghw --stop-time=10000us                                                                                 
