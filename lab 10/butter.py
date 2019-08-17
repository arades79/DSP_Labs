
"""
Skyelar Craver
DSP | Lab 10
Summer 2019
"""

#%% imports
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

plt.style.use('classic')

#%% part 1: continuous
b, a = signal.butter(4, 100, 'low', analog=True)
w, h = signal.freqs(b, a)
plt.plot(w, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='red') # cutoff frequency
plt.savefig('lab 10/images/cont_butter.png')
plt.show()

"""
Q1:
The analog argument of signal.butter creates a continuous response,
thus the filter is continuous.
Q2:
Since frequency domain is in the form of e^wt,
all changes map to order of magnitude changes.
"""

#%% part 2: discrete
b, a = signal.butter(2, 0.703/np.pi)
b1, a2 = signal.butter(8, 0.703/np.pi)
w, h = signal.freqz(b, a)
w2, h2 = signal.freqz(b1, a2)
plt.plot(w, abs(h))
plt.plot(w2,abs(h2))
plt.title('Discrete Butterworth filter frequency response')
plt.xlabel('w_hat')
plt.ylabel('Amplitude')
plt.margins(0, 0.1)
plt.legend(['N=2','N=8'])
plt.grid(which='both', axis='both')
plt.axvline(0.703, color='red') # cutoff frequency
plt.savefig('lab 10/images/disc_butter.png')
plt.show()

#%% part 3: type 1 chebychev
b, a = signal.cheby1(2, 0.1, 0.703/np.pi)
b1, a2 = signal.cheby1(8, 0.1, .703/np.pi)
w, h = signal.freqz(b, a)
w2, h2 = signal.freqz(b1, a2)
plt.plot(w, abs(h))
plt.plot(w2,abs(h2))
plt.title('Discrete Chebychev1 filter frequency response')
plt.xlabel('w_hat')
plt.ylabel('Amplitude')
plt.margins(0, 0.1)
plt.legend(['N=2','N=8'])
plt.grid(which='both', axis='both')
plt.axvline(0.703, color='red') # cutoff frequency
plt.savefig('lab 10/images/cheby_1.png')
plt.show()


#%% part 4: type 2 chebychev
b, a = signal.cheby2(2, 20, 0.703/np.pi)
b1, a2 = signal.cheby2(8, 20, .703/np.pi)
w, h = signal.freqz(b, a)
w2, h2 = signal.freqz(b1, a2)
plt.plot(w, abs(h))
plt.plot(w2,abs(h2))
plt.title('Discrete Chebychev2 filter frequency response')
plt.xlabel('w_hat')
plt.ylabel('Amplitude')
plt.margins(0, 0.1)
plt.legend(['N=2','N=8'])
plt.grid(which='both', axis='both')
plt.axvline(0.703, color='red') # cutoff frequency
plt.savefig('lab 10/images/cheby_2.png')
plt.show()

"""
Q1:
Type 1 chebychev has ripple in the pass band,
while type 2 has ripple in the stop band
Q2:
The critical frequency parameter (Wn) for chebychev 1
is based on the pass band for cutoff, where the chebychev 2
critical frequencies are based on the edge of the stop band
"""


#%% part 5: elliptic
b, a = signal.ellip(2, 0.1, 20, 0.703/np.pi)
b1, a2 = signal.ellip(8, 0.1, 20, 0.703/np.pi)
w, h = signal.freqz(b, a)
w2, h2 = signal.freqz(b1, a2)
plt.plot(w, abs(h))
plt.plot(w2,abs(h2))
plt.title('Discrete Elliptic filter frequency response')
plt.xlabel('w_hat')
plt.ylabel('Amplitude')
plt.margins(0, 0.1)
plt.legend(['N=2','N=8'])
plt.grid(which='both', axis='both')
plt.axvline(0.703, color='red') # cutoff frequency
plt.savefig('lab 10/images/ellip.png')
plt.show()

"""
Q1:
Elliptic filter provides the shortest transition band by a significant amount,
making it good for applications needing precise frequencies.
"""

#%% part 6: real sh*t
x =  np.arange(0, 1, 1e-4)
s = np.sin(7030 * x)
b, a = signal.butter(8, 0.703/np.pi)
y = signal.lfilter(b, a, s)
plt.plot(x, s)
plt.plot(x, y)
plt.title('Sine wave applied to Butterworth')
plt.xlabel('w_hat')
plt.xlim(0,1)
plt.legend(['unfiltered', 'filtered'])
plt.savefig('lab 10/images/sin.png')





#%%
