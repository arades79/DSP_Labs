"""
Skyelar Craver
DSP | Lab 8
Summer 2019
"""

#%% package imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

#%% setup
plt.style.use('classic')

#%% problem 1: example spectrogram
fs = 10e3
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500*np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*3e3*time + mod)
noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise
f, t, Sxx = signal.spectrogram(x, fs)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.savefig('lab 8/images/ex_spectrum.png')
plt.close()

#%% problem 2: PK spectrogram
rate, raw_data = wavfile.read('lab 8/PK.wav')
data = (raw_data[:, 0] + raw_data[:, 1]) / 2
f, t, sx = signal.spectrogram(data, rate, nfft=1024, scaling='spectrum')
plt.pcolormesh(t, f, sx)
plt.ylim(0, 4000)
plt.xlim(0, 60)
plt.title('Spectrogram of PK')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.savefig('lab 8/images/PK_spectrum.png')
plt.close()

#%% problem 3:
sx_const = sx.copy()
sx_const[sx_const < (sx.max() * 0.15)] = 0
plt.pcolormesh(t, f, sx_const)
plt.ylim(0, 4000)
plt.xlim(0, 60)
plt.title('Constelation Graph of PK')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.savefig('lab 8/images/PK_const.png')
plt.close()


#%% question 1:
# the Spotify version doesn't store the full sized spectrogram,
# instead it only stores the coordinates of the significant points.
# This could be implemented by only storing the indecies of star points.

