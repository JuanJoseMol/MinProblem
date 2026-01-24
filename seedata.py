import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

################################################

A = nib.load("../../../dataset/sub-r001s027_ses-1_space-MNI152NLin2009aSym_T1w.nii").get_fdata()
print (A.shape)
A =np.swapaxes(A[7:187,16:216,:-9], 0, 1)
print (A.shape)
plt.imshow(A[100,:,:])
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
plt.imshow(A[:,100,:])
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
plt.imshow(A[:,:,100])
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()

#####################################################
"""
from PIL import Image
import numpy as np
# Open the image file
img = Image.open('../../../dataset/Humans/1 (107).jpg')
#0.2125*ejt[0:16,16:80,32:96,0] + 0.7154*ejt[0:16,16:80,32:96,1] + 0.0721*ejt[0:16,16:80,32:96,2]
# Convert the image to a Numpy array
img_array = np.array(img)
print (img_array.shape)
#N_s = 200
size = (140,184)
im1 = Image.fromarray(img_array[:,:,0])
resized1 = im1.resize(size, Image.LANCZOS)
im2 = Image.fromarray(img_array[:,:,1])
resized2 = im2.resize(size, Image.LANCZOS)
im3 = Image.fromarray(img_array[:,:,2])
resized3 = im3.resize(size, Image.LANCZOS)
im = np.concatenate((np.array(resized1)[:,:,None], np.array(resized2)[:,:,None], np.array(resized3)[:,:,None]), axis=2)
# Display the array
print (im.shape)
plt.imshow(im)
plt.show()
"""