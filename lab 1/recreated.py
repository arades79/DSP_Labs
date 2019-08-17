"""
Skyelar Craver
DSP - Lab 1
"""

#%% import packages 
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from itertools import count

#%% configure plot style
plt.style.use('fivethirtyeight')


#%% define function
def plot_the_thing(T):
    # reset the current figure
    plt.figure() 

    # create two arrays using T as a base for an exponential function
    num = np.array([10-10*np.exp(-T)])
    den = np.array([1, 10-11*np.exp(-T)])

    # turn the two arrays above into a transfer function
    dtf = sig.dlti(num,den)
    # generate axes for plot
    t,y = sig.dstep((dtf),n=1000)

    # generate values for lower circle plot
    tCirc = np.linspace(0,2*np.pi,100)
    xCirc = np.cos(tCirc)
    yCirc = np.sin(tCirc)

    # obtain the roots of the function to determine the poles
    poleLocation = np.roots(den)
    # style the indicators for the pole
    colorIndicator = "b"
    markerIndicator = "o"

    # restyle the pole locator if outside the unit circle
    if np.abs(poleLocation)>1:
        colorIndicator = "r"
        markerIndicator = "x"
    # tell user location of pole in the terminal
    print("pole location",poleLocation)

    # get axes objects to draw the two plots to
    f,xarr = plt.subplots(2)

    # plot the upper graph showing the output function
    xarr[0].set_title("T = %s"%T)
    xarr[0].plot(t,y[:][0])
    # plot the lower graph showing the pole relative to the unit circle
    xarr[1].plot(1,0,c="g",marker="o")
    xarr[1].plot(xCirc,yCirc)
    xarr[1].scatter(poleLocation,0,c=colorIndicator, marker=markerIndicator)
    xarr[1].set_xlabel("pole magnitude is %s"%poleLocation)
    xarr[1].set_xlim([-2,2])

    # return the figure object to save images of plots
    return f

#%% make plots
# make an array of all Ts to use to draw plots
Tx = Tx = np.linspace(-0.08, 0.32, 30)
# run the plotting function for all values T and keep the figures
figures = [plot_the_thing(T) for T in Tx]

#%% save the plots
# set a deeper path to keep images seperate from script
path = 'images/'
# create an automatic counter to name the files sequentially
i = lambda c=count(): next(c)
# save all figures to the path with it's sequence number
[f.savefig(f'{path}{i()}.png') for f in figures]
# close all plots to free system resources
plt.close('all')

#%% create gif from images
# import more packages
import imageio
import glob
# get the full file names for all images in the folder
image_files = glob.glob("images/*.png")
# read in the images to keep them as bitmaps in memory
images = [imageio.imread(filename) for filename in image_files]
# create a gif from the pngs 
imageio.mimsave('dopegif.gif', images, duration=0.1)
