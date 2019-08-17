
"""
Skyelar Craver
DSP | Lab 9
Summer 2019
"""

#%% imports
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import pandas as pd

#%% setup
plt.style.use('classic')

#%% import dog paw
raw_paws = np.loadtxt('lab 9/paws.txt').reshape(4, 11, 14)
paws = [p.squeeze() for p in np.vsplit(raw_paws, 4)]

#%% define filter function
def find_peaks_paws(img: np.ndarray):
    neighborhood = generate_binary_structure(2, 2)
    local_max = (maximum_filter(img, footprint=neighborhood) == img)
    background = (img == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    peaks = (local_max ^ eroded_background)
    return peaks

def find_peaks_song(img: np.ndarray):
    neighborhood = generate_binary_structure(2, 2)
    local_max = (maximum_filter(img, size=10) == img)
    background = (img == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    peaks = (local_max ^ eroded_background)
    return peaks

#%% plot those paws

f, axs = plt.subplots(4, 2)

axs[0][0].imshow(paws[0])
axs[0][1].imshow(find_peaks_paws(paws[0]))
axs[1][0].imshow(paws[1])
axs[1][1].imshow(find_peaks_paws(paws[1]))
axs[2][0].imshow(paws[2])
axs[2][1].imshow(find_peaks_paws(paws[2]))
axs[3][0].imshow(paws[3])
axs[3][1].imshow(find_peaks_paws(paws[3]))
f.savefig('lab 9/images/paw_pics.png')

#%% song spectrogram
rate, raw_data = wavfile.read('lab 9/kkb.wav')
data = (raw_data[:, 0] + raw_data[:, 1]) / 2
f, t, sx = signal.spectrogram(data, rate, nfft=1024, scaling='spectrum')
plt.pcolormesh(t, f, sx)
plt.ylim(0, 4000)
plt.title('Spectrogram of Flamingo')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.savefig('lab 9/images/kkb_spectrum.png')
plt.close()


#%% filter of song
kkb_const = find_peaks_song(sx)
np.savetxt('lab 9/kkb.txt', sx, fmt='%0.4f', delimiter='    ')
np.savetxt('lab 9/kkb_const.txt', kkb_const, fmt='%0.4f', delimiter='    ')
f, ax = plt.subplots()
ax.imshow(kkb_const, aspect=10)
ax.set_title('Constelation Graph of Flamingo')
f.tight_layout()
f.savefig('lab 9/images/kkb_const.png')

#%%
