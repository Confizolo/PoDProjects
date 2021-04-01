import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin

#from unsigned to signed
def convert_tosigned(x, nbits):
    if x >= 2**(nbits-1):
        x -= 2**nbits
    return x

def countTotalBits(x):
     x = int(x)
     # convert number into it's binary and 
     # remove first two characters 0b.
     binary = bin(x)[2:]
     return len(binary)

def resize(x, nbins):
     length = countTotalBits(x)
     
     x = bin(int(x))
     print(x)
     if (length < nbins):
          for i in range(nbins-length):
               x = x[:2] + '0' + x[2:] 
          #x = int(x, 2)
               
     else:
          x = int(x, 2)
          #print(x)
          x = ( x >> ( length - nbins ) & 0b11111111 ) 
          #print(x)
          x = bin(x)
          print(x)

     return convert_tosigned(int(x, 2), 8)



#Compute the coefficient of the FIR filter
N = 5
fs = 11025
fcut = 0.1*fs/2
coeff = np.asarray(firwin(N, fcut, fs=fs))
coeff_int = []
print("FIR filter coefficients: ", coeff)
coeff= 2**7 * coeff
for i in range(N):
    coeff_int.append(round(coeff[i], 0))

coeff_int = np.asarray(coeff_int)

coeff_bin = []
for i in range(N):
    coeff_bin.append(bin(int(coeff_int[i])))

print(coeff_int, coeff_bin)

#Read the input file and put data in an array
ifile = open("input_vectors.txt", 'r')
lines = ifile.readlines()
x = np.zeros(len(lines))
for i in range(len(lines)):
    line = convert_tosigned(int(lines[i]), 8)
    print(line)
    x[i] = line
    

#Filter
index = int((N-1)/2)
y = np.zeros_like(x)
for i in range(x.size-N+1):
    y[i+index] = np.dot(x[i:N+i], coeff_int[::-1])
    signed_binary = np.binary_repr(int(y[i+index]), 18)
    print (signed_binary,  y[i+index])



    
    
    
    

#Plot the signals
plt.stem(x, linefmt='b-', markerfmt='bo', basefmt= 'k--')
plt.stem(y, linefmt='r-', markerfmt='ro', basefmt='k--')
plt.show()


