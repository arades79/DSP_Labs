
"""
Skyelar Craver
DSP - Lab 5
Summer 2019
"""

#%% import packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile

#%% define functions
def doTheWork(file_name: str):
    wav_file = f'lab 5/{file_name}.wav'
    wav_rate, wav_dat = wavfile.read(wav_file)
    fft_out = fft(wav_dat)
    abs_fft = abs(fft_out)
    time_span = wav_dat.size / wav_rate
    wav_index = np.arange(wav_dat.size)
    max_rate = 4186
    freq_label = wav_index / time_span
    plt.plot(freq_label, abs_fft)
    plt.xlim(0, max_rate)
    plt.xlabel('frequency (Hz)')
    plt.title(f'FFT of {file_name}.wav')
    plt.savefig(f'lab 5/{file_name}_fft.png')
    plt.show()


    


#%% run it
doTheWork('kpt')
doTheWork('hkp')

#%%
