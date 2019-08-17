"""
Skyelar Craver
DSP - Lab 4
Summer 2019
"""

#%% imports
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import glob
#%% functions

# sawtooth function generator
# returns a function of t with N series terms included
def fourier_sawtooth(N: int):
    def series_func(t):
        def term(n):
            costerm = np.cos(n * np.pi)
            sinterm = np.sin(n * t)
            return (costerm * sinterm) / n
        term = np.vectorize(term)
        series = term(np.arange(1, N+1))
        summation = np.sum(series) * (-2 / np.pi)
        return summation
    return np.vectorize(series_func)

# square function generator
def fourier_problem_2(N: int):
    def series_func(t):
        def term(n):
            var = n * t * np.pi
            sinterm = np.sin(var / np.e)
            invn = 1 / n
            return invn * sinterm
        term = np.vectorize(term)
        series = term(np.arange(1, N+1, 2))
        summation = np.sum(series) * (4 / np.pi)
        return summation
    return np.vectorize(series_func)

# triangle function generator
def fourier_problem_3(N: int):
    def series_func(t):
        def term(n):
            power = (-1) ** ((n - 1) / 2)
            frac = power / (n ** 2)
            sinterm = np.sin((n * np.pi * t) / np.e)
            return sinterm * frac
        term = np.vectorize(term)
        series = term(np.arange(1, N+1, 2))
        coef = (8 / np.pi ** 2)
        summation = np.sum(series) * coef
        return summation
    return np.vectorize(series_func)

# create a plotting/saving function for a given function
def plotter(func, name):
    if not os.path.exists(f"lab 4/images/{name}"):
        os.makedirs(f"lab 4/images/{name}")
    def save(n):
        T = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
        f, ax = plt.subplots()
        ax.plot(func(n)(T))
        ax.set_title(f'N = {n}')
        ax.set_ylim([-2, 2])
        f.savefig(f"lab 4/images/{name}/{n + 10}.png")
        plt.close(f)
    return np.vectorize(save)

def make_gif(name):
    # get the full file names for all images in the folder
    image_files = glob.glob(f"lab 4/images/{name}/*.png")
    # read in the images to keep them as bitmaps in memory
    images = [imageio.imread(filename) for filename in image_files]
    # create a gif from the pngs 
    imageio.mimsave(f'lab 4/dope{name}.gif', images, duration=0.08)

#%% plot
T = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
sawtooth = fourier_sawtooth(1000)
print(plt.style.available)
plt.style.use('dark_background') 
plt.plot(sawtooth(T))
plt.suptitle('N = 50')
plt.show()


#%% save the plots
# generate functions to make all plots and save them
sawtooth_saver = plotter(fourier_sawtooth, 'sawtooth')
prob2_saver = plotter(fourier_problem_2, 'problem2')
prob3_saver = plotter(fourier_problem_3, 'problem3')
# save all figures to the path with its sequence number
plt.close('all')
sawtooth_saver(np.arange(1, 51))
odds = np.arange(1, 90, 2)
prob2_saver(odds)
prob3_saver(odds)

#%% create gifs from images
make_gif('sawtooth')
make_gif('problem2')
make_gif('problem3')


#%%
