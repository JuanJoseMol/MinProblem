import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy import integrate
import jax
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RectBivariateSpline
from PIL import Image
import nibabel as nib


def pdfBrain(z1, z2, T, s=0):
    if T== 1 :
        x_root = 0.39791340077051474
    if T== 2 :
        x_root = 0.04464474022380398
    if T== 3 :
        x_root = 0.012770452481267854
    if T== 4 :
        x_root = 0.005352295370462971
    if T== 5 :
        x_root = 0.0026353277418056566
    if T== 6 :
        x_root = 0.00140805559123339
    if T== 7 :
        x_root = 0.0007904120549175704
    if T== 8 :
        x_root = 0.00045726414565818005
    if T== 9 :
        x_root = 0.0002697348646654501
    if T== 10 :
        x_root = 0.00016102466826510868
    if T== 11 :
        x_root = 9.673476153354124e-05
    if T== 12 :
        x_root = 5.831429472552495e-05
    if T== 13 :
        x_root = 3.5230537597521e-05
    if T== 14 :
        x_root = 2.131917664544327e-05
    if T== 15 :
        x_root = 1.291132445319485e-05
    if T== 16 :
        x_root = 7.82428255436289e-06
    if T== 17 :
        x_root = 4.743191616423465e-06
    if T== 18 :
        x_root = 2.8758853068461077e-06
    if T== 19 :
        x_root = 1.7439907287803812e-06
    if T== 20 :
        x_root = 1.0576286504966855e-06
    if T== 21 :
        x_root = 6.414171407950021e-07
    if T== 22 :
        x_root = 3.890023850313749e-07
    if T== 23 :
        x_root = 2.359192832809505e-07
    if T== 24 :
        x_root = 1.4308243396180418e-07
    if T== 25 :
        x_root = 8.677747326794503e-08
    if T== 26 :
        x_root = 5.2629788588040184e-08
    if T== 27 :
        x_root = 3.192135362122153e-08
    if T== 28 :
        x_root = 1.9360567876369915e-08
    if T== 29 :
        x_root = 1.1742703203966129e-08
   

    file_path = "dataset/"
    with open('dataset/Mean-197-233-189-brain.txt', 'r') as f:
        A = np.array(np.loadtxt(f))
    A = A.reshape((197,233,189))
    im = np.swapaxes(A[7:187,16:216,120], 0, 1)
    size = im.shape
    #print (size)
    im = im/255.0
    im -= im.mean()
    Nx, Ny = im.shape
    #im = im[:,:,None]
    del A
    ft = np.fft.fftshift(np.fft.fft2(im))
    g  = np.abs(ft)**2 
    
    rhotemp = np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)
    #rhotemp[0] = np.max(rhotemp)
    #rhotemp[1] = np.max(rhotemp)
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    #print (x_values.shape, y_values.shape, rhotemp.shape)
    r = RectBivariateSpline(x_values, y_values, rhotemp)

    return r(z1, z2, grid=False)


def pdfHuman(z1, z2, T, s=0):
    """
    if T==5:
        x_root = 8.134362545179581e-08

    if T==1:
        x_root = 2.701402656296036e-06
    #if T ==.5:
    #    x_root = 9.893738272263976e-06
    """
    if T== 0.0001 :
        x_root = 6.797886463568256
    if T== 0.005 :
        x_root = 0.08647913181912989
    if T== 0.01 :
        x_root = 0.018819183419613368
    if T== 0.05 :
        x_root = 0.0005818279715787992
    if T== 0.1 :
        x_root = 0.00017342790226053574
    if T== 0.5 :
        x_root = 9.893738272263976e-06
    if T== 1 :
        x_root = 4.424901777493711e-06
    if T== 1.5 :
        x_root = 1.2390335416052948e-06
    if T== 2 :
        x_root = 7.001701432581594e-07
    if T== 3 :
        x_root = 2.9621618963772706e-07
    if T== 4 :
        x_root = 1.4897077932865928e-07
    if T== 5 :
        x_root = 1.718663536539893e-07
    if T== 6 :
        x_root = 4.6423988825081376e-08
    if T== 7 :
        x_root = 2.7163874488355605e-08
    if T== 8 :
        x_root = 1.718663536539893e-07
    if T== 9 :
        x_root = 4.6423988825081376e-08
    if T== 10 :
        x_root = 1.27618532461024e-08


    from skimage.transform import resize
    #../../../
    with open('dataset/Mean-300-150-human.txt', 'r') as f:
        im = np.array(np.loadtxt(f))
    im = im.reshape((300,150,3))
    im = im/255.0
    #im -= im.mean()
    size = im.shape
    Nx, Ny = im.shape[0:2]
    #im = im[:,:,None]
    #print (im.shape)

    ft = np.fft.fftshift(np.fft.fft2(im, axes=(0,1)))
    g  = np.abs(ft)**2
    
    
    rhotemp = np.where(g[:,:,0] >= x_root, np.log(g[:,:,0]/x_root)/(2*T), 0)
    #rhotemp[0] = np.max(rhotemp)
    #rhotemp[1] = np.max(rhotemp)
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    r = RectBivariateSpline(x_values, y_values, rhotemp)

    return r(z1, z2, grid=False)

def pdf3d(z1, z2, z3,T, s=0):

    if T== 1 :
        x_root = 38.93524409340064
    if T== 2 :
        x_root = 9.439745927915354
    if T== 3 :
        x_root = 3.732997002766484
    if T== 4 :
        x_root = 1.8294435094457326
    if T== 5 :
        x_root = 1.0159157782209194
    if T== 6 :
        x_root = 0.6142276206401664
    if T== 7 :
        x_root = 0.3949477204469739
    if T== 8 :
        x_root = 0.26549242738391254
    if T== 9 :
        x_root = 0.18467254539709388
    if T== 10 :
        x_root = 0.13183672399018342
    if T== 11 :
        x_root = 0.09598007727172964
    if T== 12 :
        x_root = 0.07093695875947423
    if T== 13 :
        x_root = 0.05304501518831825
    if T== 14 :
        x_root = 0.040007226739464274
    if T== 15 :
        x_root = 0.030386227986230495
    if T== 16 :
        x_root = 0.023205585416793587
    if T== 17 :
        x_root = 0.01779791181899265
    if T== 18 :
        x_root = 0.013697182107639285
    if T== 19 :
        x_root = 0.01056933887877232
    if T== 20 :
        x_root = 0.008172001794369298
    if T== 21 :
        x_root = 0.006328728882286288
    if T== 22 :
        x_root = 0.004907688885501318
    if T== 23 :
        x_root = 0.003809292134170264
    if T== 24 :
        x_root = 0.002958891850645116
    if T== 25 :
        x_root = 0.002299667353432619
    if T== 26 :
        x_root = 0.0017881675331555755
    if T== 27 :
        x_root = 0.0013909911968913776
    if T== 28 :
        x_root = 0.001082263220374588
    if T== 29 :
        x_root = 0.0008422248312091917
    if T== 30 :
        x_root = 0.000655562412354089
    if T== 31 :
        x_root = 0.0005103502145322833
    if T== 32 :
        x_root = 0.0003973454978370083
    if T== 33 :
        x_root = 0.00030938599149180274
    if T== 34 :
        x_root = 0.0002409067247901942
    if T== 35 :
        x_root = 0.00018759044006970465
    if T== 36 :
        x_root = 0.00014607769127566073
    if T== 37 :
        x_root = 0.00011375652331140852
    if T== 38 :
        x_root = 8.858767852531674e-05
    if T== 39 :
        x_root = 6.898806770211212e-05
    if T== 40 :
        x_root = 5.37270308234582e-05
    if T== 41 :
        x_root = 4.184107193966444e-05
    if T== 42 :
        x_root = 3.2584831720918535e-05
    if T== 43 :
        x_root = 2.5376375666633037e-05
    if T== 44 :
        x_root = 1.9762732782760928e-05
    if T== 45 :
        x_root = 1.5390763329629472e-05
    if T== 46 :
        x_root = 1.198619781123529e-05
    if T== 47 :
        x_root = 9.334452321378861e-06
    if T== 48 :
        x_root = 7.2696151312602184e-06
    if T== 49 :
        x_root = 5.661479284551331e-06
    if T== 50 :
        x_root = 4.409147712774281e-06
    if T== 51 :
        x_root = 3.4338389901016705e-06
    if T== 52 :
        x_root = 2.6742688055741996e-06
    if T== 53 :
        x_root = 2.0827419780539746e-06
    if T== 54 :
        x_root = 1.6220438317555663e-06
    if T== 55 :
        x_root = 1.2632197501967512e-06
    if T== 56 :
        x_root = 9.837988193591198e-07
    if T== 57 :
        x_root = 7.661951065326109e-07
    if T== 58 :
        x_root = 5.967084595933515e-07
    if T== 59 :
        x_root = 4.6471404419567985e-07
    if T== 60 :
        x_root = 3.6191849434564687e-07
    if T== 61 :
        x_root = 2.818603120399349e-07
    if T== 62 :
        x_root = 2.1951573809211546e-07
    if T== 63 :
        x_root = 1.7095458299501593e-07
    if T== 64 :
        x_root = 1.331403867762869e-07
    if T== 65 :
        x_root = 1.036917210481076e-07
    if T== 66 :
        x_root = 8.075451335935414e-08
    if T== 67 :
        x_root = 6.289102489313717e-08
    if T== 68 :
        x_root = 4.897993039458535e-08
    if T== 69 :
        x_root = 3.814554145302569e-08
    if T== 70 :
        x_root = 2.970784646587534e-08
    if T== 71 :
        x_root = 2.313602004948546e-08
    if T== 72 :
        x_root = 1.8018585795354396e-08
    if T== 73 :
        x_root = 1.4032885136950481e-08
    if T== 74 :
        x_root = 1.0928424004342695e-08


    from skimage.color import rgb2gray
    #../../../
    from skimage.transform import resize
    with open('dataset/Mean-197-233-189-brain.txt', 'r') as f:
        A = np.array(np.loadtxt(f))
    A = A.reshape((197,233,189))

    im =  np.swapaxes(A[7:187,16:216,5:185], 0, 1)
    im = resize(im, (50,45,45))
    im = im/255.0
    im -= im.mean()
    Nx, Ny, Nz = im.shape
    
    #im = im[:,:,None]
    del A
    ft = np.fft.fftshift(np.fft.fftn(im))
    g  = np.abs(ft)**2
    
    rhotemp = np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)
    #rhotemp[0] = np.max(rhotemp)
    #rhotemp[1] = np.max(rhotemp)
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    z_values = np.linspace(-1,1,Nz)
    r = RegularGridInterpolator((x_values, y_values, z_values), rhotemp)

    return r((z1, z2, z3))

########################################################################################################

def pdfBrainRhoMean(z1, z2, T, s=0):



    file_path = "dataset/"#"../../../dataset/bestTinitial/"
    with open(file_path +'T'+str(T)+'Mean-rhoBrain.txt', 'r') as f:
        rhotemp = np.array(np.loadtxt(f))

    Nx, Ny = rhotemp.shape
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    #print (x_values.shape, y_values.shape, rhotemp.shape)
    r = RectBivariateSpline(x_values, y_values, rhotemp)

    return r(z1, z2, grid=False)

def ppdfBrainRhoMean(z1, z2, T, s=0):



    file_path = "../../../dataset/bestTinitial/"
    with open(file_path +'T'+str(T)+'Mean-rhoBrain.txt', 'r') as f:
        rhotemp = np.array(np.loadtxt(f))

    Nx, Ny = rhotemp.shape
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    #print (x_values.shape, y_values.shape, rhotemp.shape)
    r = RectBivariateSpline(x_values, y_values, rhotemp)

    return r(z1, z2, grid=False)


def pdfLowBrainRhoMean(z1, z2, T, s=0):



    file_path = "dataset/" #"../../../dataset/bestTinitial/"
    with open(file_path +'TLow'+str(T)+'Mean-rhoBrain.txt', 'r') as f:
        rhotemp = np.array(np.loadtxt(f))

    Nx, Ny = rhotemp.shape
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    #print (x_values.shape, y_values.shape, rhotemp.shape)
    r = RectBivariateSpline(x_values, y_values, rhotemp)

    return r(z1, z2, grid=False)



def pdfHumanRhoMean(z1, z2, T, s=0):

    file_path = "dataset/"
    with open('dataset/T'+str(T)+'Mean-rhoHuman.txt', 'r') as f:
        rhotemp = np.array(np.loadtxt(f))
   
    Nx, Ny = rhotemp.shape
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    r = RectBivariateSpline(x_values, y_values, rhotemp)

    return r(z1, z2, grid=False)



def pdf3dRhoMean(z1, z2, z3,T, s=0):

    
    with open('dataset/T'+str(T)+'Mean-rho3d.txt', 'r') as f:
        rhotemp = np.array(np.loadtxt(f))
    rhotemp = rhotemp.reshape((50,45,45))

    Nx, Ny, Nz = rhotemp.shape
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    z_values = np.linspace(-1,1,Nz)
    r = RegularGridInterpolator((x_values, y_values, z_values), rhotemp)

    return r((z1, z2, z3))


##########################################################################################    

def tablepdfBrainRhoMean(z1, z2, N, s=0):
    file_path = "dataset/"
    with open('dataset/N-'+str(N)+'-T1.8Mean-rhoBrain.txt', 'r') as f:
        rhotemp = np.array(np.loadtxt(f))

    Nx, Ny = rhotemp.shape
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    #print (x_values.shape, y_values.shape, rhotemp.shape)
    r = RectBivariateSpline(x_values, y_values, rhotemp)

    return r(z1, z2, grid=False)

def tablepdfHumanRhoMean(z1, z2, N, s=0):
    file_path = "dataset/"
    with open('dataset/N-'+str(N)+'-T0.75Mean-rhoHuman.txt', 'r') as f:
        rhotemp = np.array(np.loadtxt(f))
   
    Nx, Ny = rhotemp.shape
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    r = RectBivariateSpline(x_values, y_values, rhotemp)

    return r(z1, z2, grid=False)

def tablepdf3dRhoMean(z1, z2, z3,N, s=0):
    with open('dataset/N-'+str(N)+'-T1.2Mean-rho3d.txt', 'r') as f:
        rhotemp = np.array(np.loadtxt(f))
    rhotemp = rhotemp.reshape((50,45,45))

    Nx, Ny, Nz = rhotemp.shape
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    z_values = np.linspace(-1,1,Nz)
    r = RegularGridInterpolator((x_values, y_values, z_values), rhotemp)

    return r((z1, z2, z3))


#########################################################################################

def pdfBrainMediod(z1, z2, T, s=0):
    if T==30:
        x_root = 7.682393299207371e-05 #0.00027248324396737
    if T==25:
        x_root = 5.13574959509428e-07
    if T==20:
        x_root = 6.2627760746882564e-06
    if T==10:
        x_root = 0.0010012272424466278
    if T==5:
        x_root = 0.022607137903027062


    file_path = "dataset/"
    A = nib.load("dataset/mediod.nii.gz").get_fdata()
    im = np.swapaxes(A[7:187,16:216,120], 0, 1)
    size = im.shape
    #print (size)
    im = im/255.0
    im -= im.mean()
    Nx, Ny = im.shape
    #im = im[:,:,None]
    del A
    ft = np.fft.fftshift(np.fft.fft2(im))
    g  = np.abs(ft)**2 
    
    rhotemp = np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)
    #rhotemp[0] = np.max(rhotemp)
    #rhotemp[1] = np.max(rhotemp)
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    #print (x_values.shape, y_values.shape, rhotemp.shape)
    r = RectBivariateSpline(x_values, y_values, rhotemp)

    return r(z1, z2, grid=False)

def pdfHumanMediod(z1, z2, T, s=0):
    if T==5:
        x_root = 0.21815941272060835
    if T==10:
        x_root = 0.015280430484485439

    if T==20:
        x_root = 0.00010157213647859554 #
    if T==30:
        x_root = 6.842384428547901e-07

    from skimage.transform import resize
    #../../../
    im = Image.open('dataset/mediodhuman.jpg')
    # Convert the image to a Numpy array
    im = np.array(im)
    s = (150,300)
    im1 = Image.fromarray(im[:,:,0])
    resized1 = im1.resize(s, Image.LANCZOS)
    im2 = Image.fromarray(im[:,:,1])
    resized2 = im2.resize(s, Image.LANCZOS)
    im3 = Image.fromarray(im[:,:,2])
    resized3 = im3.resize(s, Image.LANCZOS)
    im = np.asarray(np.concatenate((np.array(resized1)[:,:,None], np.array(resized2)[:,:,None], 
            np.array(resized3)[:,:,None]), axis=2),dtype = 'float64')

    im[:,:,0] = im[:,:,0]/255.0
    im[:,:,1] = im[:,:,1]/255.0
    im[:,:,2] = im[:,:,2]/255.0
    im[:,:,0] -= im[:,:,0].mean()
    im[:,:,1] -= im[:,:,1].mean()
    im[:,:,2] -= im[:,:,2].mean()

    #im -= im.mean()
    size = im.shape
    Nx, Ny = im.shape[0:2]
    #im = im[:,:,None]
    #print (im.shape)

    ft = np.fft.fftshift(np.fft.fft2(im, axes=(0,1)))
    g  = np.abs(ft)**2
    
    
    rhotemp = np.where(g[:,:,0] >= x_root, np.log(g[:,:,0]/x_root)/(2*T), 0)
    #rhotemp[0] = np.max(rhotemp)
    #rhotemp[1] = np.max(rhotemp)
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    r = RectBivariateSpline(x_values, y_values, rhotemp)

    return r(z1, z2, grid=False)

def pdf3dMediod(z1, z2, z3,T, s=0):
    if T==5:
        x_root = 6.267894069419006
    
    if T==20:
        x_root = 0.07506097153519298
    if T==10:
        x_root = 1.0950490932327703 #
    if T==30:
        x_root = 0.0060661577618844695


    from skimage.color import rgb2gray
    #../../../
    from skimage.transform import resize
    A = nib.load("dataset/mediod.nii.gz").get_fdata()

    im =  np.swapaxes(A[7:187,16:216,5:185], 0, 1)
    im = resize(im, (50,45,45))
    im = im/255.0
    im -= im.mean()
    Nx, Ny, Nz = im.shape
    
    #im = im[:,:,None]
    del A
    ft = np.fft.fftshift(np.fft.fftn(im))
    g  = np.abs(ft)**2
    
    rhotemp = np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)
    #rhotemp[0] = np.max(rhotemp)
    #rhotemp[1] = np.max(rhotemp)
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    z_values = np.linspace(-1,1,Nz)
    r = RegularGridInterpolator((x_values, y_values, z_values), rhotemp)

    return r((z1, z2, z3))

#######################################################################################

def pdfBrainRhoMediod(z1, z2, T, s=0):



    file_path = "dataset/"
    with open('dataset/T'+str(T)+'Mediod-rhoBrain.txt', 'r') as f:
        rhotemp = np.array(np.loadtxt(f))
    
    Nx, Ny = rhotemp.shape
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    #print (x_values.shape, y_values.shape, rhotemp.shape)
    r = RectBivariateSpline(x_values, y_values, rhotemp)

    return r(z1, z2, grid=False)



def pdfHumanRhoMediod(z1, z2, T, s=0):

    file_path = "dataset/"
    with open('dataset/T'+str(T)+'Mediod-rhoHuman.txt', 'r') as f:
        rhotemp = np.array(np.loadtxt(f))
    
    Nx, Ny = rhotemp.shape
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    r = RectBivariateSpline(x_values, y_values, rhotemp)

    return r(z1, z2, grid=False)



def pdf3dRhoMediod(z1, z2, z3,T, s=0):

    
    with open('dataset/T'+str(T)+'Mediod-rho3d.txt', 'r') as f:
        rhotemp = np.array(np.loadtxt(f))
    rhotemp = rhotemp.reshape((50,45,45))

    Nx, Ny, Nz = rhotemp.shape
    #rhotemp[0] = np.max(rhotemp)
    #rhotemp[1] = np.max(rhotemp)
    x_values = np.linspace(-1,1,Nx)
    y_values = np.linspace(-1,1,Ny)
    z_values = np.linspace(-1,1,Nz)
    r = RegularGridInterpolator((x_values, y_values, z_values), rhotemp)

    return r((z1, z2, z3))

def sample_from_pdf_rejection(n_samples, func, sigma_a, s=0, batch_size=1000):
    x_range = (-1, 1)
    
    # Precompute max_pdf by sampling in the range
    x_samples = np.linspace(x_range[0], x_range[1], 1000)
    max_pdf = max(func(x, sigma_a, s=0) for x in x_samples)
    
    samples = np.empty(n_samples)  # Pre-allocate array for samples
    count = 0
    
    while count < n_samples:
        # Generate batch of random samples
        x_batch = np.random.uniform(x_range[0], x_range[1], batch_size)
        y_batch = np.random.uniform(0, max_pdf, batch_size)
        
        # Evaluate func for the entire batch
        pdf_batch = np.array([func(x, sigma_a, s=0) for x in x_batch])
        
        # Accept/reject samples
        accepted = x_batch[y_batch <= pdf_batch]
        n_accepted = len(accepted)
        
        # Add the accepted samples to the result
        if count + n_accepted >= n_samples:
            samples[count:n_samples] = accepted[:n_samples - count]
            count = n_samples
        else:
            samples[count:count + n_accepted] = accepted
            count += n_accepted
    
    return samples

def sample2d(n_samples, func, sigma_a, s=0, batch_size=1000):
    sampled_x_values = []
    sampled_y_values = []

    while len(sampled_x_values) < n_samples and len(sampled_y_values)< n_samples:

        samples_x = jax.random.uniform(jax.random.PRNGKey(0), shape=(40000,), minval=-1, maxval=1)
        samples_y = jax.random.uniform(jax.random.PRNGKey(1), shape=(40000,), minval=-1, maxval=1)

        pdf_values_at_samples = func(samples_x, samples_y)
        random_values = jax.random.uniform(jax.random.PRNGKey(2), shape=(40000,)) * jnp.max(pdf_values_at_samples)
        accepted_samples = random_values < pdf_values_at_samples
        sampled_x_values.extend(samples_x[accepted_samples])
        sampled_y_values.extend(samples_y[accepted_samples])
    samples = np.concatenate((np.array(sampled_x_values)[None,:n_samples],np.array(sampled_y_values)[None,:n_samples]),axis=0)
    
    return samples

def sample3d(n_samples, func, sigma_a, s=0, batch_size=1000):
    sampled_x_values = []
    sampled_y_values = []
    sampled_z_values = []

    while len(sampled_x_values)< n_samples and len(sampled_y_values)< n_samples and len(sampled_z_values)< n_samples:

        samples_x = jax.random.uniform(jax.random.PRNGKey(0), shape=(40000,), minval=-1, maxval=1)
        samples_y = jax.random.uniform(jax.random.PRNGKey(1), shape=(40000,), minval=-1, maxval=1)
        samples_z = jax.random.uniform(jax.random.PRNGKey(2), shape=(40000,), minval=-1, maxval=1)

        pdf_values_at_samples = func(samples_x, samples_y, samples_z)
        random_values = jax.random.uniform(jax.random.PRNGKey(3), shape=(40000,)) * jnp.max(pdf_values_at_samples)
        accepted_samples = random_values < pdf_values_at_samples
        sampled_x_values.extend(samples_x[accepted_samples])
        sampled_y_values.extend(samples_y[accepted_samples])
        sampled_z_values.extend(samples_z[accepted_samples])
    samples = np.concatenate((np.array(sampled_x_values)[None,:n_samples],np.array(sampled_y_values)[None,:n_samples],
                            np.array(sampled_z_values)[None,:n_samples]),axis=0)
    
    return samples


if __name__ == "__main__":
    # Generate samples
    """
    n = 1000
    s = 10
    sigma_a = np.sqrt(2/2000)
    samples1 = sample_from_pdf_rejection(n, pdfCons,sigma_a, s=s)
    #samples2 = sample_from_pdf_rejection(n, pdf2,sigma_a, s=0)

    # Optionally, plot the results
    plt.hist(samples1, bins=50, density=True, alpha=0.6)
    #plt.hist(samples2, bins=50, density=True, alpha=0.6)

    # Overlay the original PDF
    x = np.linspace(-1, 1, n)
    plt.plot(x, [pdfCons(val,sigma_a, s=s) for val in x], 'k', linewidth=2)
    plt.show()
    #print (quad(lambda x: pdf2(x,sigma_a, s=s), -1, 1))
    
    from scipy.stats import gaussian_kde
    import numpy as np 

    def f(x):
        return np.exp(-x**2/2)/np.sqrt(2*np.pi)

    y = np.linspace(-1,1,1000)
    data = f(y)
    kde = kde = gaussian_kde(data, bw_method='scott')
    samples1 = kde.resample(100)
    samples2 = kde.resample(100)
    print (np.array_equal(np.sort(samples1), np.sort(samples2)))"""

    print (pdfPDE3d(0,0,0))
    print (pdfPDEhuman(0,0))
    print (pdfPDEnumbers(0,0))
    print (pdfPDEbrain(0,0))
    