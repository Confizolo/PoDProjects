# Management and Analysis of Physics Dataset - Part A (MAPD-A)

## Overview
This directory contains VHDL hardware design projects and digital signal processing implementations, focusing on FPGA development and FIR filter design.

## Contents

### Main Projects
- **FIR Filter Implementation**: Multiple versions of Finite Impulse Response filter designs
  - FIR/, Fir-Filter/, Fir-Filter5/, FIR5/: Various FIR filter implementations in VHDL
  - Python notebooks for FIR filter design and frequency analysis
  - Test vectors and verification results

### Digital Design Components
- **Adder**: Basic arithmetic unit in VHDL
- **Multiplexer**: Data selection logic
- **Flip-Flops**: Sequential logic elements (Flip-Flop/, SyncFlipFlop/, ToggleFlipFlop/)
- **State Machines**: FSM implementations (StateMachine/, StateMachine2/, StateMachine3/)
- **UART**: Serial communication protocol with baud rate generator
  - Baudrate/, Baudrate_SM/: Baud rate generation circuits
  - UART/: Complete UART implementation

### Signal Processing
- **fir_implementation.ipynb**: Python notebook for FIR filter design
- **Freq_Analysis.ipynb**: Frequency domain analysis
- **io_test.py**: I/O testing script

### Final Report
- **MAPD_A_lab_report.pdf**: Final report on the work in the lab on FIR filters on FPGA

### Other Components
- **HeartBeat**: LED blinking circuit
- **Hello**: Basic VHDL introduction project
- **Or_gate**: Simple logic gate implementation
- **Pulse/**: Pulse generation circuit
- **Sampling circuits**: samp.vhd, samp_del.vhd, samp_sm.vhd

## Technologies
- **HDL**: VHDL
- **Languages**: Python
- **Libraries**: NumPy, SciPy, Matplotlib
- **Tools**: FPGA development tools, testbenches

## Files
- **Appunti.txt**: Notes and documentation
- **.vhd files**: VHDL source code
- **testbench.vhd**: Test benches for verification
- **input_vectors.txt, output_results*.txt**: Test data
