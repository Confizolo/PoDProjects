import serial


def twos_comp(val, bits, pos = False):
    if (val & (1 << (bits - 1))) != 0 or pos : # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)        # compute negative value
    return val

ser = serial.Serial('/dev/ttyUSB1', baudrate=115200)
print("Serial port:", ser.name)

signal_input = []
signal_output = []
input_file = open("input_vectors.txt", "r")

for line in input_file:
    # Appending values in the input list as the char corresponding to the input valueio_test
    if int(line) < 0:
        temp = - twos_comp(- int(line),8, True)
    else:
        temp = int(line)
    signal_input.append(chr(temp))
for i,signal in enumerate(signal_input):
    if i%50 == 0:
        ser.close()
        ser.open()
    # Writing the ascii byte in the serial port
    ser.write(signal)
    # Reading the serial port output and converting binary (char) to int
    signal_temp = ord(ser.read())
    # Bitwise left shift (+2 zeros to the right) for the output value
    signal_temp = signal_temp << 2
    # If signal_temp > 511, it represents a negative number
    if signal_temp > 511:
        # 2 Complement to get the corresponding negative number
        signal_temp = twos_comp(signal_temp, 10)
    signal_output.append(signal_temp)

# Writing to output file
output_file = open("output_vectors.txt", "w")

for signal in signal_output:
    output_file.write(str(signal))
    output_file.write("\n")
output_file.close()
