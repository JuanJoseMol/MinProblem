import numpy as onp
from matplotlib import pyplot as plt
from jax import random, grad, jit, vmap
import jax.numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image
from coordgrid import meshgrid_from_subdiv, flatten_all_but_lastdim
import nibabel as nib

def plotmean(tipo2, n_test):

    n1 = [i for i in range(n_test )]

    
    def slope(u):
        return np.polyfit(np.linspace(0,1,u.shape[0]), np.log(np.ravel(np.abs(u/u[0]))), 1)[0]


    if tipo2 == "astro":

        from skimage import data
        from skimage.color import rgb2gray


        #ej = data.camera()
        ej2 = data.astronaut()
        ej2 = rgb2gray(ej2)#[20:220,120:320]
        N_s = 200
        size = (N_s, N_s)
        ima = Image.fromarray(ej2)
        resized = ima.resize(size, Image.LANCZOS)
        im = np.array(resized)
        im -= im.mean()
        #im = im[:,:,None]
        del ej2
        del ima
        del resized
        layers = [2,10000,1]

        ft = np.fft.fft2(im)
        print (ft.shape)
        aN = np.sqrt(2/(layers[-1] + layers[-2]))
        lr =1e-05
        it=10
        tipo = "norm"
        SW = [60,90,120]
        colors = plt.cm.plasma(np.linspace(0,.8,4))
        fig = plt.figure(figsize=(6,6))
        for j, w in enumerate(SW):           
                for k in range(n_test):
                    with open('results/'+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(w)+'a'+str(aN.round(3))+tipo+str(k)+'.txt', 'r') as f:
                        u = onp.loadtxt(f)
                    u = u.reshape((11,size[0],size[1]))
                    u = u[:,int(size[0]/2),:]
                    u = vmap(np.fft.fft)(u) - ft[int(size[0]/2),:]

                    n1[k] = vmap(slope)(u[:it,:].T)#np.abs(u[it,:])
                #print (np.array(n1[0]).shape)#print (np.array([-n1[i][15:225]  for i in range(n_test)]).shape)
                y = np.mean(np.array([-n1[l]  for l in range(n_test)]), axis=0) #np.array(-n1[0])#gaussian_filter1d(-n1[l],10)
                #with open('results/'+tipo2+'-'+tipo+'-w'+str(int(w))+'.txt', 'w') as g:
                #    onp.savetxt(g, y)
                plt.plot(np.linspace(0,(int(size[0]/4)),size[1]), y, label = '$\sigma_w$ = '+str(int(w)))
        #plt.ylim([-0.5, 2])
        plt.legend()
        #plt.tight_layout()
        plt.show()


    if tipo2 == "numbers":
        from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist
        (traind, _), (_, _) = mnist.load_data()
        traind = traind.astype('float32') / 255.
        im = traind[100,:,:]
        im -= im.mean()
        N_s = im.shape[0]
        size = (N_s, N_s)
        #im = im[:,:,None]
        del traind
        layers = [2,10000,1]

        ft = np.fft.fft2(im)
        print (ft.shape)


        aN = np.sqrt(2/(layers[-1] + layers[-2]))
        lr =1e-05
        it=10
        tipo = "norm"
        SW = [20, 40, 60]
        colors = plt.cm.plasma(np.linspace(0,.8,4))
        fig = plt.figure(figsize=(6,6))
        for j, w in enumerate(SW):           
                for k in range(n_test):
                    with open('results/'+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(w)+'a'+str(aN.round(3))+tipo+str(k)+'.txt', 'r') as f:
                        u = onp.loadtxt(f)
                    u = u.reshape((11,size[0],size[1]))
                    u = u[:,int(size[0]/2),:]
                    u = vmap(np.fft.fft)(u) - ft[int(size[0]/2),:]

                    n1[k] = vmap(slope)(u[:it,:].T)#np.abs(u[it,:])
                #print (np.array(n1[0]).shape)#print (np.array([-n1[i][15:225]  for i in range(n_test)]).shape)
                y = np.mean(np.array([-n1[l]  for l in range(n_test)]), axis=0) #np.array(-n1[0])#gaussian_filter1d(-n1[l],10)
                #with open('results/'+tipo2+'-'+tipo+'-w'+str(int(w))+'.txt', 'w') as g:
                #    onp.savetxt(g, y)
                plt.plot(np.linspace(0,(int(size[0]/4)),size[1]), y, label = '$\sigma_w$ = '+str(int(w)))
        #plt.ylim([-0.5, 2])
        plt.legend()
        #plt.tight_layout()
        plt.show()



    
    if tipo2 == "brain":
        import nibabel as nib
        #../../../
        A = nib.load("../../../dataset/sub-r001s001_ses-1_space-MNI152NLin2009aSym_T1w.nii").get_fdata()
        im = onp.swapaxes(A[:,:,120], 0, 1)
        size = im.shape
        print (size)
        im -= im.mean()
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

        colors = plt.cm.plasma(np.linspace(0,.8,4))
        fig = plt.figure(figsize=(6,6))
        for j, w in enumerate(SW):           
                for k in range(n_test):
                    with open('results/'+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(w)+'a'+str(aN.round(3))+tipo+str(k)+'.txt', 'r') as f:
                        u = onp.loadtxt(f)
                    u = u.reshape((11,size[0],size[1]))
                    u = u[:,int(size[0]/2),:]
                    u = vmap(np.fft.fft)(u) - ft[int(size[0]/2),:]

                    n1[k] = vmap(slope)(u[:it,:].T)#np.abs(u[it,:])
                #print (np.array(n1[0]).shape)#print (np.array([-n1[i][15:225]  for i in range(n_test)]).shape)
                y = np.mean(np.array([-n1[l]  for l in range(n_test)]), axis=0) #np.array(-n1[0])#gaussian_filter1d(-n1[l],10)
                #with open('results/'+tipo2+'-'+tipo+'-w'+str(int(w))+'.txt', 'w') as g:
                #    onp.savetxt(g, y)
                plt.plot(np.linspace(0,(int(size[0]/4)),size[1]), y, label = '$\sigma_w$ = '+str(int(w)))
        #plt.ylim([-0.5, 2])
        plt.legend()
        #plt.tight_layout()
        plt.show()


    if tipo2 == "human":
        from skimage.color import rgb2gray
        #../../../
        img = Image.open('../../../dataset/1 (107).jpg')
        # Convert the image to a Numpy array
        img_array = onp.array(img)
        s = (140,184)
        im1 = Image.fromarray(img_array[:,:,0])
        resized1 = im1.resize(s, Image.LANCZOS)
        im2 = Image.fromarray(img_array[:,:,1])
        resized2 = im2.resize(s, Image.LANCZOS)
        im3 = Image.fromarray(img_array[:,:,2])
        resized3 = im3.resize(s, Image.LANCZOS)
        im = onp.concatenate((np.array(resized1)[:,:,None], np.array(resized2)[:,:,None], np.array(resized3)[:,:,None]), axis=2)
        
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

        ft = np.fft.fft2(im[:,:,0])
        print (ft.shape)

        aN = np.sqrt(2/(layers[-1] + layers[-2]))
        lr =1e-05
        it=10
        tipo = "norm"
        SW = [7,14,20]

        colors = plt.cm.plasma(np.linspace(0,.8,4))
        fig = plt.figure(figsize=(6,6))
        for j, w in enumerate(SW):
            
                for k in range(n_test):
                    with open('results/'+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(w)+'a'+str(aN.round(3))+tipo+str(k)+'.txt', 'r') as f:
                        u = onp.loadtxt(f)
                        #print (u.shape)
                    u = u.reshape((11,size[0],size[1],3))
                    u = u[:,int(size[0]/2),:,0]
                    u = vmap(np.fft.fft)(u) - ft[int(size[0]/2),:]

                    n1[k] = vmap(slope)(u[:it,:].T)#np.abs(u[it,:])
                #print (np.array(n1[0]).shape)#print (np.array([-n1[i][15:225]  for i in range(n_test)]).shape)
                y = np.mean(np.array([-n1[l]  for l in range(n_test)]), axis=0) #np.array(-n1[0])#gaussian_filter1d(-n1[l],10)
                #with open('results/'+tipo2+'-'+tipo+'-w'+str(int(w))+'.txt', 'w') as g:
                #    onp.savetxt(g, y)
                plt.plot(np.linspace(0,(int(size[0]/4)),size[1]), y, label = '$\sigma_w$ = '+str(int(w)))
        #plt.ylim([-0.5, 2])
        plt.legend()
        #plt.tight_layout()
        plt.show()


    if tipo2 == "3d":
      from skimage.color import rgb2gray
      #../../../
      import nibabel as nib
      from skimage.transform import resize
      A = nib.load("dataset/sub-r001s027_ses-1_space-MNI152NLin2009aSym_T1w.nii").get_fdata()
      #im = np.swapaxes(A[5:189,16:216,100:106], 0, 1)
      #im = resize(im, (50,46,10))
      #im =  np.swapaxes(A[5:191,14:218,98:108], 0, 1)
      #im = resize(im, (68,62,10))
      im =  np.swapaxes(A[7:187,16:216,98:108], 0, 1)
      im = resize(im, (100,90,10))
      im -= im.mean()
      size = im.shape
      
      #im = im[:,:,None]
      del A
      layers = [3,10000,1]
      grilla = meshgrid_from_subdiv(im.shape, (-1,1))
      x_train = grilla
      y_train = im[:,:,:,None]
      print (size)
      """
      plt.imshow(im[:,:,5])
      plt.show()
      """
      colors = plt.cm.plasma(np.linspace(0,.8,4))
      fig = plt.figure(figsize=(6,6))
      for j, w in enumerate(SW):           
                for k in range(n_test):
                    with open('results/'+tipo2+'lr'+str(lr)+'Ns'+str(im.shape[0])+'-w'+str(w)+'a'+str(aN.round(3))+tipo+str(k)+'.txt', 'r') as f:
                        u = onp.loadtxt(f)
                    u = u.reshape((11,size[0],size[1]))
                    u = u[:,int(size[0]/2),:]
                    u = vmap(np.fft.fft)(u) - ft[int(size[0]/2),:]

                    n1[k] = vmap(slope)(u[:it,:].T)#np.abs(u[it,:])
                #print (np.array(n1[0]).shape)#print (np.array([-n1[i][15:225]  for i in range(n_test)]).shape)
                y = np.mean(np.array([-n1[l]  for l in range(n_test)]), axis=0) #np.array(-n1[0])#gaussian_filter1d(-n1[l],10)
                #with open('results/'+tipo2+'-'+tipo+'-w'+str(int(w))+'.txt', 'w') as g:
                #    onp.savetxt(g, y)
                plt.plot(np.linspace(0,(int(size[0]/4)),size[1]), y, label = '$\sigma_w$ = '+str(int(w)))
        #plt.ylim([-0.5, 2])
      plt.legend()
        #plt.tight_layout()
      plt.show()
    
    return None




if __name__ == "__main__":
    #plotmean("brain", 50)
    plotmean("human", 50)
    #plotmean("numbers", 50)
