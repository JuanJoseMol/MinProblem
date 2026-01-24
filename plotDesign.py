import jax.numpy as np
from jax import random, grad, jit, vmap
import jax
import numpy as onp
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.color import rgb2gray
import os
import pickle
from randomSampling3 import pdf3dRhoMean, ppdfBrainRhoMean, pdfHumanRhoMean
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from coordgrid import meshgrid_from_subdiv, flatten_all_but_lastdim

fm.fontManager.addfont('../../../dataset/Helvetica Neue Bold.ttf')
matplotlib.rc('font', family='Helvetica Neue')

# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica Neue']})
# ## for Palatino and other serif fonts use:
# #matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})
# matplotlib.rc('text', usetex=True)


#plt.rc('font', family='sans-serif')
#plt.rcParams['font.family'] = u'Helvetica Neue'


plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] ='bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 18


@jit
def forward_passJJ(H, params):
  Ww = params[0]
  Wa = params[1]
  #print ("asdasd", Ww.shape, Wa.shape, H.shape)
  H = np.matmul(H, Ww)
  #print ("fdhgdf", H.shape)
  H = np.concatenate((np.sin(H), np.cos(H)),axis = -1)
  #print ("cvbvc", H.shape)

  Y = np.matmul(H,Wa)
  return Y


base_dir = "../../../dataset/testbrain"
for root, dirs, files in os.walk(base_dir):

    for k, file in enumerate(files):
        print (k)
        if k==17:
            break
        if file.endswith('T1w.nii.gz'):
            gz_path = os.path.join(root, file)

            im = nib.load(gz_path).get_fdata()#np.swapaxes(nib.load(gz_path).get_fdata()[:,:,120],0,1)
            im = np.swapaxes(im[7:187,16:216,120], 0, 1)
            maximun = 255#np.max(im)
            size = im.shape
            print (size)
            im = im/maximun
            im -= im.mean()
            layers = [2,15000,1]
            grilla = meshgrid_from_subdiv(im.shape, (-1,1))
            x_train = grilla
# Generate example 2D functions for demonstration
x = onp.linspace(-50, 50, size[0])
y = onp.linspace(-45, 45, size[1])
X, Y = onp.meshgrid(x, y)

mean = 0
std = 5*onp.pi
Z_normal = (1 / (2 * np.pi * std**2)) * np.exp(-((X - mean)**2 + (Y - mean)**2) / (2 * std**2))

# Create the 2x3 subplot
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
#fig.suptitle('Reconstructons', fontsize=16)

# Plot each function in a subplot
im = axes[1,0].imshow(
        im, 
        cmap='gray'  # Red to blue color map
    )
axes[1,0].set_title(f'Original')
axes[1,0].set_xticks([])
axes[1,0].set_yticks([])
#axes[0,0].set_xlabel('X')
#axes[0,0].set_ylabel('Y')

im = axes[0,1].imshow(
        Z_normal, 
        extent=[-1, 1, -1, 1], 
        origin='lower', 
        cmap='coolwarm'  # Red to blue color map
    )
axes[0,1].set_title(f'Normal')
axes[0,1].set_xticks([])
axes[0,1].set_yticks([])
cbar = fig.colorbar(im, ax=axes[0,1], shrink=0.8)
cbar.set_ticks([np.min(Z_normal), np.max(Z_normal)])
cbar.ax.tick_params(labelsize=8)
#cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#axes[0,1].set_xlabel('X')
#axes[0,1].set_ylabel('Y')

x = onp.linspace(-1, 1, size[0])
y = onp.linspace(-1, 1, size[1])
X, Y = onp.meshgrid(x, y)

pdf = lambda x,y: ppdfBrainRhoMean(x,y,13)

im = axes[0,2].imshow(
        pdf(X,Y), 
        extent=[-1, 1, -1, 1], 
        origin='lower', 
        cmap='coolwarm'  # Red to blue color map
    )
axes[0,2].set_title(f'Design')
axes[0,2].set_xticks([])
axes[0,2].set_yticks([])
cbar = fig.colorbar(im, ax=axes[0,2], shrink=0.8)
cbar.set_ticks([np.min(pdf(X,Y)), np.max(pdf(X,Y))])
cbar.ax.tick_params(labelsize=8)
#cbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#axes[0,2].set_xlabel('X')
#axes[0,2].set_ylabel('Y')

"""
with open('speedres/1ejbrainUni.pickle', 'rb') as file:
    data = pickle.load(file)

params = data['params']
"""

with open('speedres/1ejbrainNormal.pickle', 'rb') as file:
    data = pickle.load(file)

params1 = data['params']
loss1 = data['loss']


with open('speedres/1ejbrainRhoMean.pickle', 'rb') as file:
    data = pickle.load(file)

params2 = data['params']
loss2 = data['loss']

axes[0,0].plot(np.linspace(0,1000,1000), loss1,  label = "Normal")
axes[0,0].plot(np.linspace(0,1000,1000), loss2,  label = "Design")
axes[0,0].set_title(f'Uniform')
axes[0,0].set_xlabel('iterations')
axes[0,0].set_ylabel('Error')
axes[0,0].set_yscale("log")
axes[0,0].set_yticks([1e0,1e-4])
axes[0,0].set_xticks([0,1000])
axes[0,0].legend()

#axes[1,0].set_xlabel('X')
#axes[1,0].set_ylabel('Y')



im = axes[1,1].imshow(
        forward_passJJ(x_train, params1),
        cmap='gray'  # Red to blue color map
    )
axes[1,1].set_title(f'Normal')
axes[1,1].set_xticks([])
axes[1,1].set_yticks([])
#axes[1,1].set_xlabel('X')
#axes[1,1].set_ylabel('Y')


im = axes[1,2].imshow(
        forward_passJJ(x_train, params2), 
        cmap='gray'  # Red to blue color map
    )
axes[1,2].set_title(f'Design')
axes[1,2].set_xticks([])
axes[1,2].set_yticks([])
#axes[1,2].set_xlabel('X')
#axes[1,2].set_ylabel('Y')





#fig.colorbar(im, ax=ax, shrink=0.8)  # Add color bar to each subplot

#plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
plt.savefig('speedres/ejbrain.pdf')
plt.show()