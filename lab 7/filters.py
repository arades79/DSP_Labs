"""
Skyelar Craver
DSP | Lab 7
Summer 2019
"""

#%% package imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import lfilter, freqz

#%% change plot style
plt.style.use('classic')

#%% part 1: fft and ifft
f, axs = plt.subplots(3, figsize=(8, 6))

# original signal generation and plot
t = np.linspace(0, 2, 1000)
xt = (np.cos(20 * np.pi * t) +
      np.cos(200 * np.pi * t) +
      np.cos(400 * np.pi * t))
axs[0].plot(t, xt)
axs[0].set_title('Original Signal')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Volts (V)')

# generate FFT and plot frequencies
fft_xt = fft(xt)
fs = np.linspace(0, 500, 1000)
axs[1].plot(fs, abs(fft_xt))
axs[1].set_xlim(0, 250)
axs[1].set_title('FFT of Original Signal')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Amplitude')

# do inverse FFT and plot
ifft_fft_xt = ifft(fft_xt)
axs[2].plot(t, ifft_fft_xt)
axs[2].set_title('Original Signal Reconstructed From FFT')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Volts (V)')

# save figure and cleanup
f.tight_layout()
#f.show()
#f.savefig('lab 7/images/fft_ifft.png')
#plt.close(f)

#%% part 2: Time Domain Filtering
f, axs = plt.subplots(2, figsize=(8, 4))

N = 15
b = (1/N) * np.ones(N)
filter_out = lfilter(b, 1, xt)

axs[0].plot(t, xt)
axs[0].set_title('noisy signal')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Volts (V)')

axs[1].plot(t, filter_out)
axs[1].set_title('Filter Output')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Volts (V)')

f.tight_layout()
# f.show()
# f.savefig('lab 7/images/time_filter.png')
# plt.close(f)


#%% Part 3: Frequency Domain Filtering
f, axs = plt.subplots(3, figsize=(8, 6))

axs[0].plot(fs, abs(fft_xt))
axs[0].set_xlim(0, 250)
axs[0].set_title('Amplitude Spectrum (real)')
axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('Amplitude')

hz, H = freqz(b, 1, worN=(fs * 2 * np.pi / 500))
true_hz = hz * 500 / (2 * np.pi)

axs[1].plot(true_hz, H)
axs[1].set_title('Frequency Response of Filter')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Amplitude')
axs[1].set_xlim(0, 250)


filtered_fft = H * fft_xt
axs[2].plot(true_hz, filtered_fft)
axs[2].set_title('Filter Output (FFT)')
axs[2].set_xlabel('Frequency (Hz)')
axs[2].set_ylabel('Amplitude')
axs[2].set_xlim(0, 250)


f.tight_layout()
# f.show()
# f.savefig('lab 7/images/frequency_filter.png')
# plt.close(f)

#%% Part 3: Final Plot
f, axs = plt.subplots(4, figsize=(8, 8))

axs[0].plot(t, filter_out)
axs[0].set_title('Time Domain Filter Output')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Volts (V)')

conv_fft = fft(filter_out)
axs[1].plot(true_hz, conv_fft)
axs[1].set_title('FFT of Time Domain Filter')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Amplitude')
axs[1].set_xlim(0, 250)

axs[2].plot(true_hz, filtered_fft)
axs[2].set_title('Frequency Domain Filter Output')
axs[2].set_xlabel('Frequency (Hz)')
axs[2].set_ylabel('Amplitude')
axs[2].set_xlim(0, 250)

filter_ifft = ifft(filtered_fft)
axs[3].plot(t, filter_out)
axs[3].set_title('IFFT of Frequency Domain Filter Output')
axs[3].set_xlabel('Time (s)')
axs[3].set_ylabel('Volts (V)')

f.tight_layout()
# f.show()
# f.savefig('lab 7/images/filter_comparison.png')
# plt.close(f)


#%%
