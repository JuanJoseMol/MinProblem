import matplotlib
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt


fm.fontManager.addfont('../../../dataset/Helvetica Neue Bold.ttf')
matplotlib.rc('font', family='Helvetica Neue')

plt.rcParams['font.size'] = 15
plt.rcParams['font.weight'] ='bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 15


import numpy as np

def func(x):
  freq = 2.1
  sin = np.sin(2*np.pi*freq*x)
  return np.sign(sin)*(np.abs(sin) > 0.5)

def plot():


    fig, axs = plt.subplots(1, 3, figsize =(11, 4))

    colors = plt.cm.plasma(np.linspace(0,.8,5))
    
    x = np.linspace(-1,1,240)
    xf = np.linspace(0,60,120)


    axs[0].plot(x, func(x), "black")
    axs[0].set_title("target function",fontsize=18)
    axs[0].set_xticks([-1,1])
    axs[0].set_xlim([-1.01,1.01])
    axs[0].set_ylabel('$\widetilde{f}(x)$',fontsize=18)
    axs[0].set_xlabel('$x$', fontweight='bold',fontsize=18)

    axs[0].set_xticks(axs[0].get_xlim())  # Use the current x-axis limits for ticks

    # Get tick positions and labels
    ticks = axs[0].get_xticks()
    ticks = [-1, 1]
    print (ticks)
    #ticks = [-1, 1]
    labels = axs[0].set_xticklabels([f"{tick:.0f}" for tick in ticks])  # Set custom tick labels

    # Adjust alignment of the first and last ticks
    labels[0].set_horizontalalignment("left")  # Left-align the first tick
    labels[-1].set_horizontalalignment("right")

    

    axs[0].set_ylim([-1,1])
    axs[0].set_yticks([-1.05,1.05])

     

    axs[0].set_yticks(axs[0].get_ylim())  # Use the current x-axis limits for ticks

    # Get tick positions and labels
    ticks = axs[0].get_yticks()
    #ticks = [-1, 1]
    labels = axs[0].set_yticklabels([f"{tick:.0f}" for tick in ticks])  # Set custom tick labels

    # Adjust alignment of the first and last ticks
    labels[0].set_verticalalignment("bottom")  # Left-align the first tick
    labels[-1].set_verticalalignment("top")


    for k, tipo in enumerate(["norm", "uni"]):
        if tipo == "norm":
            SW = [30,90,180,300]
        else:
            SW = [60, 120, 240, 360]

        for j, w in enumerate(SW):
            with open('old/rate-'+tipo+'-w'+str(w)+'.txt', 'r') as f:
                y = np.loadtxt(f)
            if tipo == "norm":
                axs[k+1].plot(xf, y,  color = colors[j], label = '$\sigma_w = '+str(int(w))+'/2\pi$')
            else:
                axs[k+1].plot(xf, y,  color = colors[j], label = '$R= '+str(int(w))+'/2\pi$')


            
        #axs[k+1].set_yscale("log")
        
        
        axs[k+1].set_xticks([0,60])
        axs[k+1].set_xlim([0,60])
        axs[k+1].set_xticks(axs[k+1].get_xlim())  # Use the current x-axis limits for ticks

        # Get tick positions and labels
        ticks = axs[k+1].get_xticks()
        labels = axs[k+1].set_xticklabels([f"{tick:.0f}" for tick in ticks])  # Set custom tick labels

        # Adjust alignment of the first and last ticks
        labels[0].set_horizontalalignment("left")  # Left-align the first tick
        labels[-1].set_horizontalalignment("right")
        

        axs[k+1].set_yticks([-1,6])

        axs[k+1].set_ylim([-1,6])
        axs[k+1].set_yticks(axs[k+1].get_ylim())  # Use the current x-axis limits for ticks

        # Get tick positions and labels
        ticks = axs[k+1].get_yticks()
        #ticks = [-1, 6]
        labels = axs[k+1].set_yticklabels([f"{tick:.0f}" for tick in ticks])  # Set custom tick labels

        # Adjust alignment of the first and last ticks
        labels[0].set_verticalalignment("bottom")  # Left-align the first tick
        labels[-1].set_verticalalignment("top")

        axs[k+1].legend(loc='upper right',frameon=False,fontsize=15)
        

    
    
    #axs[0,0].set_yticks([1e-4,1e0])
    
    
    
    
    axs[1].set_xlabel('$\\xi$', fontweight='bold',fontsize=18) 
    axs[2].set_xlabel('$\\xi$', fontweight='bold',fontsize=18) 
    
    axs[0].yaxis.set_label_coords(-0.01, 0.5)
    axs[1].yaxis.set_label_coords(-0.01, 0.5)
    axs[2].yaxis.set_label_coords(-0.01, 0.5)

    axs[0].xaxis.set_label_coords(0.5, -0.02)
    axs[1].xaxis.set_label_coords(0.5, -0.02)
    axs[2].xaxis.set_label_coords(0.5, -0.02)
    

    axs[1].set_ylabel('FLR',fontsize=18)
    axs[2].set_ylabel('FLR',fontsize=18)
    axs[1].set_title("normal distribution",fontsize=18)
    axs[2].set_title("uniform distribution",fontsize=18)
    axs[1].plot([0 for i in range(120)], color = "black")
    axs[2].plot([0 for i in range(120)], color = "black")
    axs[0].tick_params(axis='both', labelsize=18)
    axs[1].tick_params(axis='both', labelsize=18)
    axs[2].tick_params(axis='both', labelsize=18)
    plt.tight_layout()
    plt.savefig('results/flr-uniform2.pdf', bbox_inches='tight')
    plt.show()

plot()