import matplotlib
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt
import math


fm.fontManager.addfont('../../../dataset/Helvetica Neue Bold.ttf')
matplotlib.rc('font', family='Helvetica Neue')

# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica Neue']})
# ## for Palatino and other serif fonts use:
# #matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})
# matplotlib.rc('text', usetex=True)


#plt.rc('font', family='sans-serif')
#plt.rcParams['font.family'] = u'Helvetica Neue'


plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] ='bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14

def ellenify(xlabel, ylabel, xlabelpad = -10, ylabelpad = -20, minXticks = True, minYticks = True):
    plt.set_xlabel(xlabel, labelpad = xlabelpad)
    plt.set_ylabel(ylabel, labelpad = ylabelpad)

    if minXticks:
        plt.xticks(plt.xlim())
        rang, labels = plt.xticks()
        labels[0].set_horizontalalignment("left")
        labels[-1].set_horizontalalignment("right")

    if minYticks:
        plt.yticks(plt.ylim())
        rang, labels = plt.yticks()
        labels[0].set_verticalalignment("bottom")
        labels[-1].set_verticalalignment("top")


import numpy as onp
from matplotlib import pyplot as plt
from jax import random, grad, jit, vmap
import jax
import jax.numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image
from coordgrid import meshgrid_from_subdiv, flatten_all_but_lastdim
import nibabel as nib
from numpy.fft import fft2, ifft2, fftshift, ifftshift



colors = plt.cm.plasma(np.linspace(0,.8,4))

"""
fig, axs = plt.subplots(1, 4, figsize=(11, 3))

for i, t in enumerate(["100", "200", "500", "200-4capas"]):
    for j, w in enumerate(["30","90","180","300"]):
        path = 'w'+w+'-m'+t
        with open('../finalresults/'+path+'.txt', 'r') as f:
            u = onp.loadtxt(f)
        axs[i].plot(np.linspace(0,60,120), u, color = colors[j], label = '$\sigma_w$ = '+w+'$/2\pi$') 

    axs[i].plot([0 for i in range(120)], color = "black")
    axs[i].set_xlabel('$\\xi$', labelpad=-15, fontsize=16)

    axs[i].set_xticks([0,60])
    axs[i].set_xlim([0,60])
    axs[i].set_xticks(axs[i].get_xlim())  # Use the current x-axis limits for ticks

    # Get tick positions and labels
    ticks = axs[i].get_xticks()
    labels = axs[i].set_xticklabels([f"{tick:.0f}" for tick in ticks],
    fontsize=14)  # Set custom tick labels

    # Adjust alignment of the first and last ticks
    labels[0].set_horizontalalignment("left")  # Left-align the first tick
    labels[-1].set_horizontalalignment("right")
    if i==0:
        axs[i].set_yticks([-0.1,0.5])
        axs[i].set_ylim([-0.1,0.5])
    if i==1:
        axs[i].set_yticks([-0.2,1])
        axs[i].set_ylim([-0.2,1])
    if i==2:
        axs[i].set_yticks([-0.5,2.5])
        axs[i].set_ylim([-0.5,2.5])
    if i==3:
        axs[i].set_yticks([-1,5])
        axs[i].set_ylim([-1,5])
    axs[i].set_yticks(axs[i].get_ylim())  # Use the current x-axis limits for ticks

    # Get tick positions and labels
    ticks = axs[i].get_yticks()
    #ticks = [-1, 6]
    labels = axs[i].set_yticklabels([f"{tick:.1f}" for tick in ticks],
    fontsize=14)  # Set custom tick labels

    # Adjust alignment of the first and last ticks
    labels[0].set_verticalalignment("bottom")  # Left-align the first tick
    labels[-1].set_verticalalignment("top")
    

axs[0].set_title('2-layers, m = 100', fontsize=12)
axs[1].set_title('2-layers, m = 200', fontsize=12)
axs[2].set_title('2-layers, m = 500', fontsize=12)
axs[3].set_title('4-layers, [1,2000,200,200,1]', fontsize=12)
#axs[0].set_yticks([-1,5])


axs[0].set_ylabel('FLR', labelpad=-20, fontsize=16)





axs[3].legend(frameon=False,loc='upper right',fontsize=10)


plt.tight_layout()
fig.subplots_adjust(wspace=0.25)
plt.savefig('results/NTKRobustnessS.pdf', bbox_inches='tight')
plt.show()

"""


fig, axs = plt.subplots(1, 3, figsize=(8, 3))

for i, t in enumerate(["100", "200", "200-4capas"]):
    for j, w in enumerate(["30","90","180","300"]):
        path = 'w'+w+'-m'+t
        with open('../finalresults/'+path+'.txt', 'r') as f:
            u = onp.loadtxt(f)
        axs[i].plot(np.linspace(0,60,120), u, color = colors[j], label = '$\sigma_w$ = '+w+'$/2\pi$') 

    axs[i].plot([0 for i in range(120)], color = "black")
    axs[i].set_xlabel('$\\xi$', labelpad=-15, fontsize=12)

    axs[i].set_xticks([0,60])
    axs[i].set_xlim([0,60])
    axs[i].set_xticks(axs[i].get_xlim())  # Use the current x-axis limits for ticks

    # Get tick positions and labels
    ticks = axs[i].get_xticks()
    labels = axs[i].set_xticklabels([f"{tick:.0f}" for tick in ticks],
    fontsize=14)  # Set custom tick labels

    # Adjust alignment of the first and last ticks
    labels[0].set_horizontalalignment("left")  # Left-align the first tick
    labels[-1].set_horizontalalignment("right")
    if i==0:
        axs[i].set_yticks([-0.1,0.5])
        axs[i].set_ylim([-0.1,0.5])
    if i==1:
        axs[i].set_yticks([-0.2,1])
        axs[i].set_ylim([-0.2,1])
    if i==2:
        axs[i].set_yticks([-1,5])
        axs[i].set_ylim([-1,5])
    axs[i].set_yticks(axs[i].get_ylim())  # Use the current x-axis limits for ticks

    # Get tick positions and labels
    ticks = axs[i].get_yticks()
    #ticks = [-1, 6]
    labels = axs[i].set_yticklabels([f"{tick:.1f}" for tick in ticks],fontsize=12)  # Set custom tick labels
    

    # Adjust alignment of the first and last ticks
    labels[0].set_verticalalignment("bottom")  # Left-align the first tick
    labels[-1].set_verticalalignment("top")
    

axs[0].set_title('2-layers, m = 100', fontsize=12)
axs[1].set_title('2-layers, m = 200', fontsize=12)
#axs[2].set_title('2-layers, m = 500', fontsize=12)
axs[2].set_title('4-layers, [1,2000,200,200,1]', fontsize=12)
#axs[0].set_yticks([-1,5])


axs[0].set_ylabel('FLR', labelpad=-25, fontsize=12)





axs[2].legend(frameon=False,loc='upper right',fontsize=10)


plt.tight_layout()
fig.subplots_adjust(wspace=0.25)
plt.savefig('results/NTKRobustness2.pdf', bbox_inches='tight')
plt.show()


