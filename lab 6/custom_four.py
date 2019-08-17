
"""
Skyelar Craver
DSP - Lab 6
Summer 2019
"""

#%% import packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile
from typing import Union
import time

#%% define functions
# this is an implementation using numpy parallelization
# and other optimizations without changing the core algorithm
# n = 1000 samples takes 0.07 seconds
def npDFT(u: np.array) -> np.array:
    N = u.size
    k = np.arange(u.size)
    @np.vectorize
    def DFT_elem(n: int):
        exponent = np.exp((-1j * 2 * np.pi * k * n) / N)
        series = u * exponent
        return np.abs(np.sum(series))
    U = DFT_elem(np.arange(0, N))
    return U

# here's a vanilla implementation
def DFT(u: np.array) -> np.array:
    N = u.size
    U = []
    for n in np.arange(u.size):
        term = 0.0
        for k in np.arange(u.size):
            exponent = np.exp((-1j * 2 * np.pi * k * n) / N)
            term += u[k] * exponent
        U.append(abs(term)) 
    return np.array(U)

def timer(func, n):
    start = time.perf_counter()
    func(n)
    return (time.perf_counter() - start)

def wav_DFT(path: str):
    rate, u = wavfile.read(path)
    time_span = u.size / rate
    labels = np.arange(u.size) / time_span
    return labels, npDFT(u)


#%% Q1
dft_test = lambda f: npDFT(np.sin(f * np.linspace(0, 2 * np.pi, 1000)))

# adjusting the indecies to be hz is unecessary due to the sample
# being only 1 second long resulting in the conversion being 
# (output index) / 1, which is equivalent to (output index)
plt.plot(dft_test(10))
plt.xlabel('Frequency (Hz)')
plt.savefig('lab 6/images/fullsize10hz.png')
# this looks dumb though, DFT/FFT will always produce symmetrical results,
# but anything over half the sampling frequency is just an alias
# (something about nyquist's theorum)
# so for the rest of the charts, the axis are cut to focus on the good bits
plt.xlim(0, 50)
plt.savefig('lab 6/images/10hzsin.png')
plt.close()

plt.plot(dft_test(20))
plt.xlim(0, 50)
plt.xlabel('Frequency (Hz)')
plt.savefig('lab 6/images/20hzsin.png')
plt.close()

plt.plot(dft_test(30))
plt.xlim(0, 50)
plt.xlabel('Frequency (Hz)')
plt.savefig('lab 6/images/30hzsin.png')
plt.close()

#%% Q2
in_sin = np.sin(10 * np.linspace(0, 2 * np.pi, 1000))
n_samples = [t for t in range(100, 1001, 100)]
times = [timer(DFT, in_sin[:t]) for t in n_samples]
plt.plot(n_samples, times)
plt.title('Execution Time Plot')
plt.xlabel('Number of Samples')
plt.ylabel('time (seconds)')
plt.savefig('lab 6/images/time_complexity_vanilla.png')
plt.close()

# this is where you learn that for loops are suboptimal,
# and that numpy is very fast
in_sin = np.sin(10 * np.linspace(0, 2 * np.pi, 10000))
n_samples = [t for t in range(1000, 10001, 1000)]
times = [timer(npDFT, in_sin[:t]) for t in n_samples]
plt.plot(n_samples, times)
plt.title('Execution Time Plot')
plt.xlabel('Number of Samples')
plt.ylabel('time (seconds)')
plt.savefig('lab 6/images/time_complexity_np.png')
plt.close()

#%% Q3
_, data = wavfile.read('lab 5/kpt.wav')
ds = data.size
n2 = ds ** 2
vanilla = n2 * (1 / 400)
numpy = n2 * (1 / 4000)
print(f'number of samples in kpt.wav: {ds}\n' 
      f'complexity of taking the DFT: {n2}\n'
      f'estimated time of execution for vanilla DFT: {vanilla}\n'
      f'estimated time of execusion for numpy DFT: {numpy}')
# number of samples in kpt.wav: 561152
# complexity of taking the DFT: 314891567104
# estimated time of execution for vanilla DFT: 787228917.76
# estimated time of execusion for numpy DFT: 78722891.77600001

#%% Q4
freqs, DFT_out = wav_DFT('lab 6/kpt1note2k.wav')
plt.plot(freqs, DFT_out)
plt.xlim(0, 4000)
plt.title('KPT One Note DFT')
plt.xlabel('Frequency (Hz)')
# now lets pick the principal frequency and plot that too
max_loc = np.argmax(DFT_out[:50])
fmax = freqs[max_loc]
dmax = DFT_out[max_loc]
plt.plot(fmax, dmax, marker='o', label=f'{fmax}Hz (A4#)')
plt.legend()
plt.savefig('lab 6/images/kptnote.png')
plt.close()

#%%
