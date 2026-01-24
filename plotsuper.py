import nibabel as nib
import jax
import jax.numpy as np
import numpy as onp
from flax import linen as nn
from jax import random
import matplotlib.pyplot as plt
import pickle
from skimage.transform import resize
from coordgrid import meshgrid_from_subdiv, flatten_all_but_lastdim
from jax import random, grad, jit, vmap

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

def plotsuper2(tipo, tipo2, tipo3):

    gz_path = "../../../dataset/sub-r039s003_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz"

    im2 = nib.load(gz_path).get_fdata()
    im2 = np.swapaxes(im2[7:187,16:216,120], 0, 1)
    im2 = im2/255.0
    im2 -= im2.mean()
    im = resize(im2, (100,90))

    grilla_x = meshgrid_from_subdiv(im2.shape, (-1,1))
    grilla_y = im2[:,:,None]
    x_train = grilla_x
    y_train = im[:,:,None]
    v = np.pi 
    m = 15000
    lis = [1e-3]#[1e-1, 1e-2, 1e-3]
    if tipo3 == "1":
        wlist = [10*v,20*v,30*v,40*v,50*v]
    else:
        wlist = [45*v, 90*v]#[60*v,70*v,80*v,90*v,100*v]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for lr in lis:
        for i, w in enumerate(wlist):#,:
        
            with open('super/2Low'+tipo+tipo2+'-lr'+str(lr)+'w-'+str(round(w,3))+'m'+str(m)+'.pickle', 'rb') as input_file:
                res = pickle.load(input_file)
            
            with open('super/2'+tipo+'uniform-lr'+str(lr)+'w-'+str(round(w,3))+'m'+str(m)+'.pickle', 'rb') as input_file:
                res2 = pickle.load(input_file)
            
            loss = res['train_loss']
            lossdev = res['dev_loss']
            params = res['best_param']
            lossdevuni = res2['dev_loss']

            
            axes[i, 0].plot([i for i in range(1000)], loss, label = 'train')
            axes[i, 0].plot([i for i in range(1000)], lossdev, label = 'dev')
            axes[i, 0].plot([i for i in range(1000)], lossdevuni, label = 'devuni')
            axes[i, 0].set_title('w = '+str(round(w,3))+', lr = '+str(lr))
            axes[i, 0].set_yscale("log")
            axes[i, 0].legend(fontsize=8)



            im1_plot = axes[i, 1].imshow(im, cmap="gray")
            axes[i, 1].set_title("low-resolution")
            fig.colorbar(im1_plot, ax=axes[i, 1])

            im2_plot = axes[i, 2].imshow(im2, cmap="gray")
            axes[i, 2].set_title("original")
            fig.colorbar(im2_plot, ax=axes[i, 2])

            im3_plot = axes[i,3].imshow(forward_passJJ(grilla_x, params), cmap="gray")
            axes[i, 3].set_title("prediction")
            fig.colorbar(im3_plot, ax=axes[i, 3])
            
    plt.tight_layout()
    plt.savefig('super/'+tipo+'-'+tipo2+'-super2-'+tipo3+'.pdf')
    plt.show()

def plotsuper1(tipo, tipo2, tipo3):

    gz_path = "../../../dataset/sub-r039s003_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz"

    im2 = nib.load(gz_path).get_fdata()#np.swapaxes(nib.load(gz_path).get_fdata()[:,:,120],0,1)
    im2 = np.swapaxes(im2[7:187,16:216,120], 0, 1)
    im2 = im2/255.0
    im2 -= im2.mean()
    im = resize(im2, (100,90))

    orig_height , orig_width = im2.shape
    new_height, new_width = im.shape
    #print (size)
    
    layers = [2,15000,1]

    grilla_x = meshgrid_from_subdiv(im2.shape, (-1,1))
    grilla_y = im2[:,:,None]
    x_train = grilla_x[::orig_height // new_height, ::orig_width // new_width]
    y_train = im[:,:,None]

    v = np.pi 
    lis = [1e-3]#[1e-1, 1e-2, 1e-3]
    if tipo3 == "1":
        wlist = [10*v,20*v,30*v,40*v,50*v]
    else:
        wlist = [60*v,70*v,80*v,90*v,100*v]
    fig, axes = plt.subplots(5, 4, figsize=(12, 15))
    for lr in lis:
        for i, w in enumerate(wlist):#,:
        
            with open('super/'+tipo+tipo2+'-lr'+str(lr)+'w-'+str(round(w,3))+'.pickle', 'rb') as input_file:
                res = pickle.load(input_file)
            
            with open('super/'+tipo+'uniform-lr'+str(lr)+'w-'+str(round(w,3))+'.pickle', 'rb') as input_file:
                res2 = pickle.load(input_file)
            
            loss = res['train_loss']
            lossdev = res['dev_loss']
            params = res['best_param']
            lossdevuni = res2['dev_loss']

            
            axes[i, 0].plot([i for i in range(1000)], loss, label = 'train')
            axes[i, 0].plot([i for i in range(1000)], lossdev, label = 'dev')
            axes[i, 0].plot([i for i in range(1000)], lossdevuni, label = 'devuni')
            axes[i, 0].set_title('w = '+str(round(w,3))+', lr = '+str(lr))
            axes[i, 0].set_yscale("log")
            axes[i, 0].legend(fontsize=8)



            im1_plot = axes[i, 1].imshow(im, cmap="gray")
            axes[i, 1].set_title("low-resolution")
            fig.colorbar(im1_plot, ax=axes[i, 1])

            im2_plot = axes[i, 2].imshow(im2, cmap="gray")
            axes[i, 2].set_title("original")
            fig.colorbar(im2_plot, ax=axes[i, 2])

            im3_plot = axes[i,3].imshow(forward_passJJ(grilla_x, params), cmap="gray")
            axes[i, 3].set_title("prediction")
            fig.colorbar(im3_plot, ax=axes[i, 3])
            
    plt.tight_layout()
    plt.savefig('super/'+tipo+'-'+tipo2+'-super1-'+tipo3+'.pdf')
    plt.show()

def plotinpaint(tipo, tipo2, tipo3):

    gz_path = "../../../dataset/sub-r039s003_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz"

    im = nib.load(gz_path).get_fdata()#np.swapaxes(nib.load(gz_path).get_fdata()[:,:,120],0,1)
    im = onp.swapaxes(im[7:187,16:216,120], 0, 1)
    im = onp.array(im)
    im = im/255.0
    im -= im.mean()

    #print (size)
    valor = 0.1
    hole_x_min, hole_x_max = -valor , valor 
    hole_y_min, hole_y_max = -valor , valor 

    x = onp.linspace(-1, 1, im.shape[1])
    y = onp.linspace(-1, 1, im.shape[0]) 
    X, Y = onp.meshgrid(x, y)

    # Create a mask with ones everywhere, except in the hole region
    mask = onp.ones(im.shape, dtype=bool)
    mask[(X >= hole_x_min) & (X <= hole_x_max) & (Y >= hole_y_min) & (Y <= hole_y_max)] = False
    
    layers = [2,15000,1]
    grilla_x = onp.array(meshgrid_from_subdiv(im.shape, (-1,1)))
    grilla_y = im
    x_train = grilla_x[mask]
    y_train = im[mask][:,None]
    image_with_hole = onp.copy(im)
    image_with_hole[~mask] = np.nan

    v = np.pi 
                
    lis = [1e-1]#[1e-1, 1e-2, 1e-3]
    wlist = [90*v]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for lr in lis:
        for i, w in enumerate(wlist):#,:
        
            with open('inpainting/N'+tipo+tipo2+'-lr'+str(lr)+'w-'+str(round(w,3))+'.pickle', 'rb') as input_file:
                res = pickle.load(input_file)
            
            with open('inpainting/N'+tipo+'uniform-lr'+str(lr)+'w-'+str(round(w,3))+'.pickle', 'rb') as input_file:
                res2 = pickle.load(input_file)
            
            loss = res['train_loss']
            lossdev = res['dev_loss']
            params = res['best_param']
            lossdevuni = res2['dev_loss']

            
            axes[0].plot([i for i in range(1000)], loss, label = 'train')
            axes[0].plot([i for i in range(1000)], lossdev, label = 'dev')
            axes[0].plot([i for i in range(1000)], lossdevuni, label = 'devuni')
            axes[0].set_title('w = '+str(round(w,3))+', lr = '+str(lr))
            axes[0].set_yscale("log")
            axes[0].legend(fontsize=8)

            vmim = im.min()
            vmax = im.max()

            im1_plot = axes[1].imshow(image_with_hole, cmap="gray")
            axes[1].set_title("Image with hole")
            fig.colorbar(im1_plot, ax=axes[1])

            im2_plot = axes[ 2].imshow(im, cmap="gray")
            axes[2].set_title("original")
            fig.colorbar(im1_plot, ax=axes[2])

            im3_plot = axes[3].imshow(forward_passJJ(grilla_x, params), cmap="gray",  vmin=vmim, vmax=vmax)
            axes[3].set_title("prediction")
            fig.colorbar(im1_plot, ax=axes[3])
            
        plt.tight_layout()
        plt.savefig('inpainting/01'+tipo+'-'+tipo2+'-super-'+tipo3+'.pdf')
        plt.show()



def plotsuperm(tipo, tipo2):

    gz_path = "dataset/sub-r039s003_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz"

    im2 = nib.load(gz_path).get_fdata()
    im2 = np.swapaxes(im2[7:187,16:216,120], 0, 1)
    im2 = im2/255.0
    im2 -= im2.mean()
    im = resize(im2, (100,90))

    grilla_x = meshgrid_from_subdiv(im2.shape, (-1,1))
    grilla_y = im2[:,:,None]
    x_train = grilla_x
    y_train = im[:,:,None]
    v = np.pi 
    lis = [1e-3]#[1e-1, 1e-2, 1e-3]
    w = 45*v

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for lr in lis:
        for i, m in enumerate([30000,40000,50000]):#,:
        
            with open('super/2'+tipo+tipo2+'-lr'+str(lr)+'w-'+str(round(w,3))+'m'+str(m)+'.pickle', 'rb') as input_file:
                res = pickle.load(input_file)
            
            with open('super/2'+tipo+'uniform-lr'+str(lr)+'w-'+str(round(w,3))+'m'+str(m)+'.pickle', 'rb') as input_file:
                res2 = pickle.load(input_file)
            
            loss = res['train_loss']
            lossdev = res['dev_loss']
            params = res['best_param']
            lossdevuni = res2['dev_loss']

            
            axes[i, 0].plot([i for i in range(1000)], loss, label = 'train')
            axes[i, 0].plot([i for i in range(1000)], lossdev, label = 'dev')
            axes[i, 0].plot([i for i in range(1000)], lossdevuni, label = 'devuni')
            axes[i, 0].set_title('w = '+str(round(w,3))+', lr = '+str(lr))
            axes[i, 0].set_yscale("log")
            axes[i, 0].legend(fontsize=8)



            im1_plot = axes[i, 1].imshow(im, cmap="gray")
            axes[i, 1].set_title("low-resolution")
            fig.colorbar(im1_plot, ax=axes[i, 1])

            im2_plot = axes[i, 2].imshow(im2, cmap="gray")
            axes[i, 2].set_title("original")
            fig.colorbar(im2_plot, ax=axes[i, 2])

            im3_plot = axes[i,3].imshow(forward_passJJ(grilla_x, params), cmap="gray")
            axes[i, 3].set_title("prediction")
            fig.colorbar(im3_plot, ax=axes[i, 3])
            
    plt.tight_layout()
    plt.savefig('super/m'+tipo+'-'+tipo2+'-super2.pdf')
    plt.show()


if __name__ == "__main__":
  """
  val = "RhoMediod" 
  plotinpaint("brain", val, "1") #Medoid, Mean, RhoMean, RhoMediod
  plotinpaint("brain", val, "2")
  val = "Medoid" 
  plotinpaint("brain", val, "1") #Medoid, Mean, RhoMean, RhoMediod
  plotinpaint("brain", val, "2")
  """
  val = "RhoMean" 
  plotinpaint("brain", val, "2") #Medoid, Mean, RhoMean, RhoMediod
  #plotinpaint("brain", val, "2")
  """
  val = "Mean" 
  plotinpaint("brain", val, "1") #Medoid, Mean, RhoMean, RhoMediod
  plotinpaint("brain", val, "2")
  #plotsuper1("brain")
  #plotinpaint("brain")
  """

  
