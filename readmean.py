import os
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt

"""
# Define the top-level directories
base_dirs = ["../../../dataset/ATLAS_2/Training/R001",
             "../../../dataset/ATLAS_2/Training/R009",
             "../../../dataset/ATLAS_2/Training/R031",
             "../../../dataset/ATLAS_2/Training/R038",
             "../../../dataset/ATLAS_2/Training/R052"]  # Replace with actual paths

S = np.zeros((197, 233, 189))
# Loop over all base directories (R01, R38, etc.)
for base_dir in base_dirs:
    i = 0
    for root, dirs, files in os.walk(base_dir):
        
        for file in files:
            if file.endswith('T1w.nii.gz'):
                i += 1
                gz_path = os.path.join(root, file)
                
                # Load the NIfTI file directly (nibabel handles .nii.gz)
                nii_image = nib.load(gz_path)
                image = nii_image.get_fdata()
                S += np.array(image)/209
                print (i, nii_image.shape)
                
                # Process the NIfTI image (add your processing code here)
                #print(f"Processing {file}")

with open('../../../dataset/Mean-197-233-189-brain.txt', 'w') as f:
    np.savetxt(f, S.reshape(197,-1))


plt.imshow(np.array(S)[:,100,:])
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
plt.imshow(np.array(S)[100,:,:])
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
plt.imshow(np.array(S)[:,:,100])
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
"""
##################################################################################
"""
from PIL import Image
base_dir = "../../../dataset/Humans"
i = 0
w = []
h = []
r = []
for root, dirs, files in os.walk(base_dir):
        
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            i += 1
            path = os.path.join(root, file)
            s = np.array(Image.open(path))
            print (i, s.shape)
            w.append(s.shape[0])
            h.append(s.shape[1])
            r.append(s.shape[0]/s.shape[1])
            if s.shape[0] == 500:
                print (s.shape)
            if s.shape[1] == 333:
                print (s.shape)

print (np.min(np.array(w)), np.max(np.array(w)))
print (np.min(np.array(h)), np.max(np.array(h)))
print (np.median(np.array(w)), np.median(np.array(h)))
print (np.mean(np.array(w)), np.mean(np.array(h)))
print (np.min(np.array(r)), np.max(np.array(r)))
print (np.mean(np.array(r)), np.median(np.array(r)))

"""
################################### deleting ############################
"""
import os
from PIL import Image

# Folder where your images are stored
image_folder = "../../../dataset/Humans"

# Loop through the files in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Add more extensions if needed
        image_path = os.path.join(image_folder, filename)
        
        try:
            # Open the image file
            with Image.open(image_path) as img:
                # Check the shape (size and number of channels)
                img
                
                if len(img.getbands()) != 3:
                    # If the image does not have 3 channels (RGB), delete it
                    print(f"Deleting {filename} - Shape: {img.size}, Channels: {len(img.getbands())}")
                    os.remove(image_path)
                elif np.array(img).shape[0]<=np.array(img).shape[1]:
                    # If the image does not have 3 channels (RGB), delete it
                    print(f"Deleting {filename} - Shape: {img.size}, Channels: {len(img.getbands())}")
                    os.remove(image_path)
                elif np.array(img).shape[0]<500:
                    # If the image does not have 3 channels (RGB), delete it
                    print(f"Deleting {filename} - Shape: {img.size}, Channels: {len(img.getbands())}")
                    os.remove(image_path)
                elif (np.array(img).shape[0]/np.array(img).shape[1])<1.3 or (np.array(img).shape[0]/np.array(img).shape[1])>1.7:
                    # If the image does not have 3 channels (RGB), delete it
                    print(f"Deleting {filename} - Shape: {img.size}, Channels: {len(img.getbands())}")
                    os.remove(image_path)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
"""
#######################################
"""

import os
from PIL import Image

# Folder where your images are stored
image_folder = "../../../dataset/Humans"

target_size = (150,300)  # For example, resize all images to 256x256

# Initialize a list to store the image data
images_resized = []

# Loop through the files in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Add more extensions if needed
        image_path = os.path.join(image_folder, filename)
        
        try:
            # Open the image file
            with Image.open(image_path) as img:
                # Ensure the image has 3 channels (RGB)
                if len(img.getbands()) == 3:
                    # Resize the image to the target size
                    img_resized = img.resize(target_size, Image.BILINEAR)
                    
                    # Convert the image to a NumPy array
                    img_array = np.array(img_resized, dtype=np.float32) / 255.0  # Normalize to [0, 1]
                    
                    # Add the resized image to the list
                    images_resized.append(img_array)
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Stack all images into a 4D array (num_images, width, height, channels)
images_stack = np.stack(images_resized, axis=0)

# Compute the mean of the images
mean_image = np.mean(images_stack, axis=0)

from matplotlib import pyplot as plt
plt.imshow(mean_image)
plt.show()
print (mean_image.shape)

with open('../../../dataset/Mean-300-150-human.txt', 'w') as f:
    np.savetxt(f, mean_image.reshape(300,-1))

"""
################################################# find the medoid brain #########################
"""
import os
import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from skimage.transform import resize


base_dirs = ["../../../dataset/ATLAS_2/Training/R001",
             "../../../dataset/ATLAS_2/Training/R009",
             "../../../dataset/ATLAS_2/Training/R031",
             "../../../dataset/ATLAS_2/Training/R038",
             "../../../dataset/ATLAS_2/Training/R052"]

# List to hold the flattened arrays of each image
images = []
i = 0
# Loop over all base directories and load .nii files
for base_dir in base_dirs:
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('T1w.nii.gz'):
                print (i)
                i += 1
                gz_path = os.path.join(root, file)
                img = nib.load(gz_path).get_fdata()#np.swapaxes(nib.load(gz_path).get_fdata()[:,:,120],0,1)
                img = resize(img, (50,45,45))
                images.append(img.flatten())  # Flatten the 3D image into 1D array

# Convert the list of images to a NumPy array for easier distance calculation
images = np.array(images)
print (images.shape)

"""
"""
small = 100000
for i in range(len(images)):
    s = 0
    for j in range(len(images)):
        s += np.sum((images[i,:]-images[j,:])**2)
    print (i, np.sqrt(s)) 
    if np.sqrt(s) < small:
        small = np.sqrt(s)
        small_idx = i

print (small, small_idx)

# Compute the pairwise distances between all images
distances = cdist(images, images, metric='euclidean')
print (distances.shape)

# Find the medoid (the index of the image that minimizes the sum of distances to all others)
medoid_idx = np.argmin(distances.sum(axis=1))
print (medoid_idx)

medoid_image_path = None
i = 0
for base_dir in base_dirs:
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('T1w.nii.gz'):
                if i == medoid_idx:
                    medoid_image_path = os.path.join(root, file)
                    break
                i += 1

# Load the medoid image
if medoid_image_path:
    medoid_image = nib.load(medoid_image_path).get_fdata()#np.swapaxes(nib.load(medoid_image_path).get_fdata()[:,:,120],0,1)
    print(f"Medoid image found at: {medoid_image_path}")
else:
    print("Medoid image not found.")
"""






################################### Find medoid faces #########################################
"""
import os
import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from skimage.transform import resize
from PIL import Image


base_dirs = ["../../../dataset/Humans"]

# List to hold the flattened arrays of each image
images = []
i = 0
# Loop over all base directories and load .nii files
for base_dir in base_dirs:
    print (base_dir)
    
    for root, dirs, files in os.walk(base_dir):
        print (root, dirs, files)
        for file in files:
            print (file)
            if file.endswith('.jpg') or file.endswith('.png'):
                print (i)
                i += 1
              
                gz_path = os.path.join(root, file)
                img = Image.open(gz_path)
                img_array = np.array(img)
                size = img_array.shape
                print (size)
                s = (150,300)
                im1 = Image.fromarray(img_array[:,:,0])
                resized1 = im1.resize(s, Image.LANCZOS)
                im2 = Image.fromarray(img_array[:,:,1])
                resized2 = im2.resize(s, Image.LANCZOS)
                im3 = Image.fromarray(img_array[:,:,2])
                resized3 = im3.resize(s, Image.LANCZOS)
                im = np.asarray(np.concatenate((np.array(resized1)[:,:,None], np.array(resized2)[:,:,None]
                                , np.array(resized3)[:,:,None]), axis=2), dtype = 'float64')   
                images.append(im.flatten())
               
# Compute the pairwise distances between all images
distances = cdist(images, images, metric='euclidean')
print (distances.shape)

# Find the medoid (the index of the image that minimizes the sum of distances to all others)
medoid_idx = np.argmin(distances.sum(axis=1))
print (medoid_idx)

medoid_image_path = None
i = 0
for base_dir in base_dirs:
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                if i == medoid_idx:
                    medoid_image_path = os.path.join(root, file)
                    break
                i += 1

# Load the medoid image
if medoid_image_path:
    medoid_image = Image.open(medoid_image_path)#np.swapaxes(nib.load(medoid_image_path).get_fdata()[:,:,120],0,1)
    print(f"Medoid image found at: {medoid_image_path}")
else:
    print("Medoid image not found.")

"""
########################################## mean rho from 3d ########################


def target_function(x, g_array, T):

    area = 8
    
    # Compute the integrand values for the filtered g_array
    integrand_values = (np.maximum((np.log(g) - np.log(x)) / (2 * T), 0))
    
    # Approximate the integral as a Riemann sum
    integral_value = np.sum(integrand_values)*area/np.size(g)
    
    return integral_value

# Bisection method
def bisection_method(g_array, T, tol=1e-6, max_iter=100):
    a, b = 1e-8, 100000  # Define the search interval

    for i in range(max_iter):
        c = (a + b) / 2  # Midpoint

        # Evaluate the function at the midpoint
        f_c = target_function(c, g_array, T)

        # Check if the function value is close to 0.5
        if np.abs(f_c - 1) < tol:
            return c
        
        # Decide which half of the interval to search in
        f_a = target_function(a, g_array, T)
        if np.sign(f_c - 1) == np.sign(f_a - 1):
            a = c  # Narrow the interval to [c, b]
        else:
            b = c  # Narrow the interval to [a, c]
    
    raise ValueError("Bisection method did not converge")


from skimage.transform import resize

# Define the top-level directories
base_dirs = ["../../../dataset/ATLAS_2/"]  # Replace with actual paths

for T in [1.2, 1.3]:
    print ("----------------------------------T = ", T, "----------------------------------") 
    S = np.zeros((50, 45, 45))
    # Loop over all base directories (R01, R38, etc.)
    i = 0
    for base_dir in base_dirs:
        
        for root, dirs, files in os.walk(base_dir):
            
            for file in files:
                if file.endswith('T1w.nii.gz'):
                    print (i)
                    i += 1
                    gz_path = os.path.join(root, file)
                    
                    # Load the NIfTI file directly (nibabel handles .nii.gz)
                    nii_image = nib.load(gz_path)
                    image = nii_image.get_fdata()
                    im =  np.swapaxes(image[7:187,16:216,5:185], 0, 1)
                    im = resize(im, (50,45,45))
                    im = im/255.0
                    im -= im.mean()
                    g = np.abs(np.fft.fftshift(np.fft.fftn(im)))**2 
                    x_root = bisection_method(g, T)

                    S += np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)


    print (i)
    with open('../../../dataset/T'+str(T)+'Mean-rho3d.txt', 'w') as f:
        np.savetxt(f, S.reshape(50,-1)/i)



"""
plt.imshow(np.array(S)[:,20,:])
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
plt.imshow(np.array(S)[20,:,:])
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
plt.imshow(np.array(S)[:,:,20])
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()


########################################## mean rho from brain ########################


def target_function(x, g_array, T):

    area = 4
    
    # Compute the integrand values for the filtered g_array
    integrand_values = (np.maximum((np.log(g) - np.log(x)) / (2 * T), 0))
    
    # Approximate the integral as a Riemann sum
    integral_value = np.sum(integrand_values)*area/np.size(g)
    
    return integral_value

# Bisection method
def bisection_method(g_array, T, tol=1e-6, max_iter=100):
    a, b = 1e-8, 100000  # Define the search interval

    for i in range(max_iter):
        c = (a + b) / 2  # Midpoint

        # Evaluate the function at the midpoint
        f_c = target_function(c, g_array, T)

        # Check if the function value is close to 0.5
        if np.abs(f_c - 1) < tol:
            return c
        
        # Decide which half of the interval to search in
        f_a = target_function(a, g_array, T)
        if np.sign(f_c - 1) == np.sign(f_a - 1):
            a = c  # Narrow the interval to [c, b]
        else:
            b = c  # Narrow the interval to [a, c]
    
    raise ValueError("Bisection method did not converge")


from skimage.transform import resize

# Define the top-level directories
base_dirs = ["../../../dataset/ATLAS_2/"]  # Replace with actual paths

for T in [1.9]:
    print ("---------------------------------- T = ", T, "-----------------------------------")
    S = np.zeros((200, 180))
    # Loop over all base directories (R01, R38, etc.)
    i = 0
    for base_dir in base_dirs:
        
        for root, dirs, files in os.walk(base_dir):
            
            for file in files:
                if file.endswith('T1w.nii.gz'):
                    print (i)
                    i += 1
                    gz_path = os.path.join(root, file)
                    
                    # Load the NIfTI file directly (nibabel handles .nii.gz)
                    nii_image = nib.load(gz_path)
                    image = nii_image.get_fdata()
                    im =  np.swapaxes(image[7:187,16:216,120], 0, 1)
                    #im = resize(im, (100,90))
                    im = im/255.0
                    im -= im.mean()
                    g = np.abs(np.fft.fftshift(np.fft.fft2(im)))**2
                    x_root = bisection_method(g, T)

                    S += np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)
                    #plt.imshow(np.where(g >= x_root, np.log(g/x_root)/(2*T), 0))
                    #plt.show()


    print (i)
    with open('../../../dataset/T'+str(T)+'Mean-rhoBrain.txt', 'w') as f:
        np.savetxt(f, S/i)


#plt.imshow(np.array(S)[:,:]/i)
#plt.colorbar(fraction=0.046, pad=0.04)
#plt.show()



########################################## mean rho from human ########################


def target_function(x, g_array, T):

    area = 4
    
    # Compute the integrand values for the filtered g_array
    integrand_values = (np.maximum((np.log(g) - np.log(x)) / (2 * T), 0))
    
    # Approximate the integral as a Riemann sum
    integral_value = np.sum(integrand_values)*area/np.size(g)
    
    return integral_value

# Bisection method
def bisection_method(g_array, T, tol=1e-6, max_iter=100):
    a, b = 1e-8, 100000  # Define the search interval

    for i in range(max_iter):
        c = (a + b) / 2  # Midpoint

        # Evaluate the function at the midpoint
        f_c = target_function(c, g_array, T)

        # Check if the function value is close to 0.5
        if np.abs(f_c - 1) < tol:
            return c
        
        # Decide which half of the interval to search in
        f_a = target_function(a, g_array, T)
        if np.sign(f_c - 1) == np.sign(f_a - 1):
            a = c  # Narrow the interval to [c, b]
        else:
            b = c  # Narrow the interval to [a, c]
    
    raise ValueError("Bisection method did not converge")


from PIL import Image


base_dirs = ["../../../dataset/Humans"]

for T in [0.8]:
    print ("---------------------------------- T = ", T, "-----------------------------------")
    S = np.zeros((300, 150))

    # List to hold the flattened arrays of each image
    images = []
    i = 0
    # Loop over all base directories and load .nii files
    for base_dir in base_dirs:
        #print (base_dir)
        
        for root, dirs, files in os.walk(base_dir):
            #print (root, dirs, files)
            for file in files:
                #print (file)
                if file.endswith('.jpg') or file.endswith('.png'):
                    print (i)
                    i += 1
                
                    gz_path = os.path.join(root, file)
                    img = Image.open(gz_path)
                    img_array = np.array(img)
                    size = img_array.shape
                    #print (size)
                    s = (150,300)
                    im1 = Image.fromarray(img_array[:,:,0])
                    resized1 = im1.resize(s, Image.LANCZOS)
                    im2 = Image.fromarray(img_array[:,:,1])
                    resized2 = im2.resize(s, Image.LANCZOS)
                    im3 = Image.fromarray(img_array[:,:,2])
                    resized3 = im3.resize(s, Image.LANCZOS)
                    im = np.asarray(np.concatenate((np.array(resized1)[:,:,None], np.array(resized2)[:,:,None]
                                    , np.array(resized3)[:,:,None]), axis=2), dtype = 'float64')   
                    im[:,:,0] = im[:,:,0]/255.0
                    im[:,:,1] = im[:,:,1]/255.0
                    im[:,:,2] = im[:,:,2]/255.0
                    im[:,:,0] -= im[:,:,0].mean()
                    im[:,:,1] -= im[:,:,1].mean()
                    im[:,:,2] -= im[:,:,2].mean()
                    g = np.abs(np.fft.fftshift(np.fft.fft2(im[:,:,0])))**2
                    x_root = bisection_method(g, T)

                    S += np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)


    print (i)
    with open('../../../dataset/T'+str(T)+'Mean-rhoHuman.txt', 'w') as f:
        np.savetxt(f, S/i)


#plt.imshow(np.array(S)[:,:]/i)
#plt.colorbar(fraction=0.046, pad=0.04)
#plt.show()



for T in [1.2, 1.3]:
    ########################################## mediod rho from 3d ########################

    from scipy.spatial.distance import cdist

    def target_function(x, g_array, T):

        area = 8
        
        # Compute the integrand values for the filtered g_array
        integrand_values = (np.maximum((np.log(g) - np.log(x)) / (2 * T), 0))
        
        # Approximate the integral as a Riemann sum
        integral_value = np.sum(integrand_values)*area/np.size(g)
        
        return integral_value

    # Bisection method
    def bisection_method(g_array, T, tol=1e-6, max_iter=100):
        a, b = 1e-8, 100000  # Define the search interval

        for i in range(max_iter):
            c = (a + b) / 2  # Midpoint

            # Evaluate the function at the midpoint
            f_c = target_function(c, g_array, T)

            # Check if the function value is close to 0.5
            if np.abs(f_c - 1) < tol:
                return c
            
            # Decide which half of the interval to search in
            f_a = target_function(a, g_array, T)
            if np.sign(f_c - 1) == np.sign(f_a - 1):
                a = c  # Narrow the interval to [c, b]
            else:
                b = c  # Narrow the interval to [a, c]
        
        raise ValueError("Bisection method did not converge")


    from skimage.transform import resize

    # Define the top-level directories
    base_dirs = ["../../../dataset/ATLAS_2/"]  # Replace with actual paths

    S = []
    # Loop over all base directories (R01, R38, etc.)
    i = 0
    for base_dir in base_dirs:
        
        for root, dirs, files in os.walk(base_dir):
            
            for file in files:
                if file.endswith('T1w.nii.gz'):
                    print (i)
                    i += 1
                    gz_path = os.path.join(root, file)
                    
                    # Load the NIfTI file directly (nibabel handles .nii.gz)
                    nii_image = nib.load(gz_path)
                    image = nii_image.get_fdata()
                    im =  np.swapaxes(image[7:187,16:216,5:185], 0, 1)
                    im = resize(im, (50,45,45))
                    im = im/255.0
                    im -= im.mean()
                    g = np.abs(np.fft.fftshift(np.fft.fftn(im)))**2 
                    #T=5
                    x_root = bisection_method(g, T)

                    S.append(np.ravel(np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)))

    # Compute the pairwise distances between all images
    distances = cdist(S, S, metric='euclidean')
    print (distances.shape)

    # Find the medoid (the index of the image that minimizes the sum of distances to all others)
    medoid_idx = np.argmin(distances.sum(axis=1))
    print (medoid_idx)

    medoid_image_path = None
    i = 0
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('T1w.nii.gz'):
                    if i == medoid_idx:
                        medoid_image_path = os.path.join(root, file)
                        break
                    i += 1

    # Load the medoid image
    if medoid_image_path:
        medoid_image = nib.load(medoid_image_path).get_fdata()#np.swapaxes(nib.load(medoid_image_path).get_fdata()[:,:,120],0,1)
        print(f"Medoid image found at: {medoid_image_path}")
    else:
        print("Medoid image not found.")

    medoid_image =  np.swapaxes(medoid_image[7:187,16:216,5:185], 0, 1)
    medoid_image = resize(medoid_image, (50,45,45))
    medoid_image = medoid_image/255.0
    medoid_image -= medoid_image.mean()
    g = np.abs(np.fft.fftshift(np.fft.fftn(medoid_image)))**2
    #T=5
    x_root = bisection_method(g, T)

    res = np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)


    print (i)
    with open('../../../dataset/T'+str(T)+'Mediod-rho3d.txt', 'w') as f:
        np.savetxt(f, res.reshape(50,-1))

    
    plt.imshow(np.array(res)[:,20,:])
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()
    plt.imshow(np.array(res)[20,:,:])
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()
    plt.imshow(np.array(res)[:,:,20])
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()
    

    ########################################## mediod rho from brain ########################
    

    from scipy.spatial.distance import cdist

    def target_function(x, g_array, T):

        area = 4
        
        # Compute the integrand values for the filtered g_array
        integrand_values = (np.maximum((np.log(g) - np.log(x)) / (2 * T), 0))
        
        # Approximate the integral as a Riemann sum
        integral_value = np.sum(integrand_values)*area/np.size(g)
        
        return integral_value

    # Bisection method
    def bisection_method(g_array, T, tol=1e-6, max_iter=100):
        a, b = 1e-8, 100000  # Define the search interval

        for i in range(max_iter):
            c = (a + b) / 2  # Midpoint

            # Evaluate the function at the midpoint
            f_c = target_function(c, g_array, T)

            # Check if the function value is close to 0.5
            if np.abs(f_c - 1) < tol:
                return c
            
            # Decide which half of the interval to search in
            f_a = target_function(a, g_array, T)
            if np.sign(f_c - 1) == np.sign(f_a - 1):
                a = c  # Narrow the interval to [c, b]
            else:
                b = c  # Narrow the interval to [a, c]
        
        raise ValueError("Bisection method did not converge")


    from skimage.transform import resize

    # Define the top-level directories
    base_dirs = ["../../../dataset/ATLAS_2/Training/R001",
                "../../../dataset/ATLAS_2/Training/R009",
                "../../../dataset/ATLAS_2/Training/R031",
                "../../../dataset/ATLAS_2/Training/R038",
                "../../../dataset/ATLAS_2/Training/R052"]  # Replace with actual paths

    S = []
    # Loop over all base directories (R01, R38, etc.)
    i = 0
    for base_dir in base_dirs:
        
        for root, dirs, files in os.walk(base_dir):
            
            for file in files:
                if file.endswith('T1w.nii.gz'):
                    print (i)
                    i += 1
                    gz_path = os.path.join(root, file)
                    
                    # Load the NIfTI file directly (nibabel handles .nii.gz)
                    nii_image = nib.load(gz_path)
                    image = nii_image.get_fdata()
                    im =  np.swapaxes(image[7:187,16:216,120], 0, 1)
                    #im = resize(im, (50,45,45))
                    im = im/255.0
                    im -= im.mean()
                    g = np.abs(np.fft.fftshift(np.fft.fft2(im)))**2
                    #T=5
                    x_root = bisection_method(g, T)

                    S.append(np.ravel(np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)))

    # Compute the pairwise distances between all images
    distances = cdist(S, S, metric='euclidean')
    print (distances.shape)

    # Find the medoid (the index of the image that minimizes the sum of distances to all others)
    medoid_idx = np.argmin(distances.sum(axis=1))
    print (medoid_idx)

    medoid_image_path = None
    i = 0
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('T1w.nii.gz'):
                    if i == medoid_idx:
                        medoid_image_path = os.path.join(root, file)
                        break
                    i += 1

    # Load the medoid image
    if medoid_image_path:
        medoid_image = nib.load(medoid_image_path).get_fdata()#np.swapaxes(nib.load(medoid_image_path).get_fdata()[:,:,120],0,1)
        print(f"Medoid image found at: {medoid_image_path}")
    else:
        print("Medoid image not found.")

    medoid_image =  np.swapaxes(medoid_image[7:187,16:216,120], 0, 1)
    medoid_image = medoid_image/255.0
    medoid_image -= medoid_image.mean()
    g = np.abs(np.fft.fftshift(np.fft.fftn(medoid_image)))**2 
    #T=5
    x_root = bisection_method(g, T)

    res = np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)


    print (i)
    with open('../../../dataset/bestTinitial/T'+str(T)+'Mediod-rhoBrain.txt', 'w') as f:
        np.savetxt(f, res)

    
    #plt.imshow(np.array(res))
    #plt.colorbar(fraction=0.046, pad=0.04)
    #plt.show()
    



    
    ########################################## mediod rho from human ########################
    

    from scipy.spatial.distance import cdist

    def target_function(x, g_array, T):

        area = 4
        
        # Compute the integrand values for the filtered g_array
        integrand_values = (np.maximum((np.log(g) - np.log(x)) / (2 * T), 0))
        
        # Approximate the integral as a Riemann sum
        integral_value = np.sum(integrand_values)*area/np.size(g)
        
        return integral_value

    # Bisection method
    def bisection_method(g_array, T, tol=1e-6, max_iter=100):
        a, b = 1e-8, 100000  # Define the search interval

        for i in range(max_iter):
            c = (a + b) / 2  # Midpoint

            # Evaluate the function at the midpoint
            f_c = target_function(c, g_array, T)

            # Check if the function value is close to 0.5
            if np.abs(f_c - 1) < tol:
                return c
            
            # Decide which half of the interval to search in
            f_a = target_function(a, g_array, T)
            if np.sign(f_c - 1) == np.sign(f_a - 1):
                a = c  # Narrow the interval to [c, b]
            else:
                b = c  # Narrow the interval to [a, c]
        
        raise ValueError("Bisection method did not converge")


    from PIL import Image


    base_dirs = ["../../../dataset/Humans"]
    S = []

    # List to hold the flattened arrays of each image
    images = []
    i = 0
    # Loop over all base directories and load .nii files
    for base_dir in base_dirs:
        print (base_dir)
        
        for root, dirs, files in os.walk(base_dir):
            print (root, dirs, files)
            for file in files:
                print (file)
                if file.endswith('.jpg') or file.endswith('.png'):
                    print (i)
                    i += 1
                
                    gz_path = os.path.join(root, file)
                    img = Image.open(gz_path)
                    img_array = np.array(img)
                    size = img_array.shape
                    print (size)
                    s = (150,300)
                    im1 = Image.fromarray(img_array[:,:,0])
                    resized1 = im1.resize(s, Image.LANCZOS)
                    im2 = Image.fromarray(img_array[:,:,1])
                    resized2 = im2.resize(s, Image.LANCZOS)
                    im3 = Image.fromarray(img_array[:,:,2])
                    resized3 = im3.resize(s, Image.LANCZOS)
                    im = np.asarray(np.concatenate((np.array(resized1)[:,:,None], np.array(resized2)[:,:,None]
                                    , np.array(resized3)[:,:,None]), axis=2), dtype = 'float64')   
                    im[:,:,0] = im[:,:,0]/255.0
                    im[:,:,1] = im[:,:,1]/255.0
                    im[:,:,2] = im[:,:,2]/255.0
                    im[:,:,0] -= im[:,:,0].mean()
                    im[:,:,1] -= im[:,:,1].mean()
                    im[:,:,2] -= im[:,:,2].mean()
                    g = np.abs(np.fft.fftshift(np.fft.fft2(im[:,:,0])))**2 
                    #T=5
                    x_root = bisection_method(g, T)

                    S.append(np.ravel(np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)))

    # Compute the pairwise distances between all images
    distances = cdist(S, S, metric='euclidean')
    print (distances.shape)

    # Find the medoid (the index of the image that minimizes the sum of distances to all others)
    medoid_idx = np.argmin(distances.sum(axis=1))
    print (medoid_idx)

    medoid_image_path = None
    i = 0
    for base_dir in base_dirs:
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    if i == medoid_idx:
                        medoid_image_path = os.path.join(root, file)
                        break
                    i += 1
                    print (i)

    # Load the medoid image
    if medoid_image_path:
        medoid_image = Image.open(medoid_image_path)#np.swapaxes(nib.load(medoid_image_path).get_fdata()[:,:,120],0,1)
        print(f"Medoid image found at: {medoid_image_path}")
    else:
        print("Medoid image not found.")


    img_array = np.array(medoid_image)
    size = img_array.shape
    print (size)
    s = (150,300)
    im1 = Image.fromarray(img_array[:,:,0])
    resized1 = im1.resize(s, Image.LANCZOS)
    im2 = Image.fromarray(img_array[:,:,1])
    resized2 = im2.resize(s, Image.LANCZOS)
    im3 = Image.fromarray(img_array[:,:,2])
    resized3 = im3.resize(s, Image.LANCZOS)
    im = np.asarray(np.concatenate((np.array(resized1)[:,:,None], np.array(resized2)[:,:,None]
                    , np.array(resized3)[:,:,None]), axis=2), dtype = 'float64')  

    im[:,:,0] = im[:,:,0]/255.0
    im[:,:,1] = im[:,:,1]/255.0
    im[:,:,2] = im[:,:,2]/255.0
    im[:,:,0] -= im[:,:,0].mean()
    im[:,:,1] -= im[:,:,1].mean()
    im[:,:,2] -= im[:,:,2].mean()
    g = np.abs(np.fft.fftshift(np.fft.fft2(im[:,:,0])))**2 
    x_root = bisection_method(g, T)

    res = np.where(g >= x_root, np.log(g/x_root)/(2*T), 0)


    print (i)
    with open('../../../dataset/bestTinitial/T'+str(T)+'Mediod-rhoHuman.txt', 'w') as f:
        np.savetxt(f, res)

    
    #plt.imshow(np.array(res))
    #plt.colorbar(fraction=0.046, pad=0.04)
    #plt.show()
    """

