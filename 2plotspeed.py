

######################################################
#######################################


import matplotlib
import matplotlib.font_manager as fm
from matplotlib import pyplot as plt

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

def ellenify(xlabel, ylabel, xlabelpad = -10, ylabelpad = -20, minXticks = True, minYticks = True):
    plt.xlabel(xlabel, labelpad = xlabelpad)
    plt.ylabel(ylabel, labelpad = ylabelpad)

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
import jax.numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image
from coordgrid import meshgrid_from_subdiv, flatten_all_but_lastdim
import nibabel as nib

v = np.pi
#v = 3.2

fig, axs = plt.subplots(2, 4, figsize=(11, 5))

tipo2 = "numbers"
from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist
(_, _), (traind,_ ) = mnist.load_data()
traind = traind.astype('float32') / 255.
im = traind[100,:,:]
axs[0, 0].imshow(im)
axs[0, 0].set_title('100 test-mnist')
axs[0, 0].axis('off')
"""
ma =[]
mi =[]
for j, w in enumerate([40, 60, 80, 100]):
    for i, t in enumerate(["Uni","Const", "PDE"]):
        with open('speedres/'+t+'numberslr0.0001-w'+str(w)+'a0.014.txt', 'r') as f:
            u = onp.loadtxt(f)
            ma.append(np.max(u))
            mi.append(np.min(u))
maximo = max(ma)
minimo = min(mi)
"""
colors = plt.cm.plasma(np.linspace(0,.8,3))
for j, w in enumerate([14*v]):
    for i, t in enumerate(["Uni","Const", "PDE"]): #, "PDENoU"
        with open('speedres/Mean'+t+'numberslr0.0001-w'+str(w)+'a0.020000001.txt', 'r') as f:
            u = onp.loadtxt(f)
            print (u.shape)
            #ma.append(np.max(u))
            #mi.append(np.min(u))
            
        axs[j+1, 0].plot([i for i in range(u.shape[0])], u, color=colors[i], label = t+' design')
        axs[j+1, 0].set_title('R = '+str(round(w,3)))
    #plt.ylim([-0.5, 2])
    axs[j+1, 0].set_yscale('log')
    #axs[j+1, 0].set_ylim(1.4651563e-14, 2.1967256)
    #axs[j+1, 0].legend()







tipo2 = "brain"
import nibabel as nib
#../../../
A = nib.load("../../../dataset/sub-r039s002_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz").get_fdata()
im = onp.swapaxes(A[7:187,16:216,120], 0, 1)
axs[0, 1].imshow(im, cmap="gray")
axs[0, 1].set_title('R039s002 ATLAS')
axs[0, 1].axis('off')
size = im.shape
print (size)
#im -= im.mean()
#im = im[:,:,None]
del A
layers = [2,10000,1]

ft = np.fft.fftshift(np.fft.fft2(im))
print (ft.shape)

aN = np.sqrt(2/(layers[-1] + layers[-2]))
lr =1e-05
it=10
tipo = "norm"
SW = [60,90,120]

for j, w in enumerate([90*v]):
    for i, t in enumerate(["Uni","Const", "PDE"]): #, "PDENoU"
        with open('speedres/Mean'+t+'brainlr0.001-w'+str(w)+'a0.012.txt', 'r') as f:
            u = onp.loadtxt(f)
        axs[j+1, 1].plot([i for i in range(u.shape[0])], u, color=colors[i], label = t+' design')
        axs[j+1, 1].set_title('R = '+str(round(w,3)))
    axs[j+1, 1].set_yscale('log')
    #axs[j+1, 1].set_ylim(0.64477, 494.3083)
    #axs[j+1, 1].legend()





tipo2 = "human"
from skimage.color import rgb2gray
#../../../
img = Image.open('../../../dataset/1 (1487).jpg')
# Convert the image to a Numpy array
img_array = onp.array(img)[:,1:,:]
s = (159,270)
im1 = Image.fromarray(img_array[:,:,0])
resized1 = im1.resize(s, Image.LANCZOS)
im2 = Image.fromarray(img_array[:,:,1])
resized2 = im2.resize(s, Image.LANCZOS)
im3 = Image.fromarray(img_array[:,:,2])
resized3 = im3.resize(s, Image.LANCZOS)
im = onp.concatenate((np.array(resized1)[:,:,None], np.array(resized2)[:,:,None], np.array(resized3)[:,:,None]), axis=2)

axs[0, 2].imshow(im)
axs[0, 2].set_title('1487 HUMAN FACES')
axs[0, 2].axis('off')
im = im/255.0
#im -= im.mean()
size = im.shape
#im = im[:,:,None]
print (im.shape)
del img
del img_array
del resized1
del resized2
del resized3
del im1
del im2
del im3
layers = [2,10000,3]

ft = np.fft.fft2(im, axes=(0,1))
print (ft.shape)

aN = np.sqrt(2/(layers[-1] + layers[-2]))
lr =1e-05
it=10
tipo = "norm"
SW = [7,14,20]

for j, w in enumerate([79.5*v]):
    for i, t in enumerate(["Uni","Const", "PDE"]): #, "PDENoU"
        with open('speedres/Mean'+t+'humanlr0.001-w'+str(w)+'a0.012.txt', 'r') as f:
            u = onp.loadtxt(f)
        axs[j+1, 2].plot([i for i in range(u.shape[0])], u, color=colors[i], label = t+' design')
        axs[j+1, 2].set_title('R = '+str(round(w,3)))
    axs[j+1, 2].set_yscale('log')
    #axs[j+1, 2].set_ylim(0.00028639878, 2.2920856)
    #axs[j+1,2].legend()



tipo2 = "3d"
from skimage.color import rgb2gray
#../../../
import nibabel as nib
from skimage.transform import resize
A = nib.load("../../../dataset/sub-r039s003_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz").get_fdata()

im =  np.swapaxes(A[7:187,16:216,98:108], 0, 1)
mip= Image.open('../../../dataset/3d-2.png')

axs[0, 3].imshow(mip)
axs[0, 3].set_title('3D r039s003 ATLAS')
axs[0, 3].axis('off')
im = resize(im, (100,90,10))
im -= im.mean()
size = im.shape

#im = im[:,:,None]
del A
layers = [3,15000,1]
for j, w in enumerate([22.5*v]):
    for i, t in enumerate(["Uni","Const", "PDE"]): #, "PDENoU"
        with open('speedres/Mean'+t+'3dlr0.01-w'+str(w)+'a0.012.txt', 'r') as f:
            u = onp.loadtxt(f)
            print (u.shape)
        axs[j+1, 3].plot([i for i in range(1000)], u, color=colors[i], label = t+' design')
        axs[j+1, 3].set_title('R = '+str(round(w,3)))
    axs[j+1, 3].set_yscale('log')
    #axs[j+1, 3].set_ylim(1.3970686, 577.69666)
    #axs[j+1, 3].legend()

#plt.tight_layout()
#handles, labels = axs[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3)
#plt.tight_layout(rect=[0, 0.15, 1, 1])
handles, labels = axs[1,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3)
plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig('speedres/Meanloss2.pdf')
plt.show()


