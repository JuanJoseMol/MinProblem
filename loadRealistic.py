import numpy as np
from PIL import Image
import scipy.ndimage as ndi
import jax.numpy as jnp
from matplotlib import pyplot as plt

def load_image(ej):
    if ej == "noise":
        img = Image.open("../data/noiseRaro0649.png").convert("RGB") 
        img = jnp.array(img)/255.0
        img = img ** 2.2
        img = downsample_antialiased(img, factor=2)
        img = add_poisson_noise(img,max_photons=30,readout_std=2.0)
        img = img ** (1/2.2)
        
        imgTrue = Image.open("../data/noiseRaro0649.png").convert("RGB") 
        imgTrue = jnp.array(imgTrue)/255.0
        imgTrue  = downsample_antialiased(imgTrue , factor=2)
    if ej == "super":
        img = Image.open("../data/mariposa0829.png").convert("RGB") 
        img = downsample_antialiased(img, factor=2)
        img = jnp.array(img)/255.0
        imgTrue = Image.open("../data/mariposa0829.png").convert("RGB") 
        imgTrue = jnp.array(imgTrue)/255.0
    if ej == "fitting":
        img = Image.open("../data/playa0823.png").convert("RGB") 
        img = downsample_antialiased(img, factor=2)
        img = jnp.array(img)/255.0
        imgTrue = img
    if ej == "fitting2":
        img = Image.open("../data/loros23.png").convert("RGB") 
        img = jnp.array(img)/255.0
        imgTrue = img
    if ej == "fitting3":
        img = Image.open("../data/castle19.png").convert("RGB") 
        img = jnp.array(img)/255.0
        imgTrue = img
    if ej == "super2":
        img = Image.open("../data/pinguino0344.png").convert("RGB") 
        img = jnp.array(img)[70:-70,48:-48]
        img = downsample_antialiased(img, factor=2)
        img = img/255.0
        imgTrue = Image.open("../data/pinguino0344.png").convert("RGB") 
        imgTrue = jnp.array(imgTrue)[70:-70,48:-48]/255.0
    if ej == "super3":
        img = Image.open("../data/Coral0026.png").convert("RGB") 
        img = jnp.array(img)[48:-48,70:-70]
        img = downsample_antialiased(img, factor=2)
        img = img/255.0
        imgTrue = Image.open("../data/Coral0026.png").convert("RGB") 
        imgTrue = jnp.array(imgTrue)[48:-48,70:-70]/255.0
    if ej == "noise2":
        img = Image.open("../data/limones0802.png").convert("RGB") 
        img = np.array(img)
        img = img/255.0
        img = img ** 2.2
        img = downsample_antialiased(img, factor=2)
        img = add_poisson_noise(img,max_photons=30,readout_std=2.0)
        img = img ** (1/2.2)

        imgTrue = Image.open("../data/limones0802.png").convert("RGB") 
        imgTrue = np.array(imgTrue)
        imgTrue = imgTrue/255.0
        imgTrue  = downsample_antialiased(imgTrue , factor=2)
    if ej == "noise3":
        img = Image.open("../data/hongo0858.png").convert("RGB") 
        img = np.array(img)
        img = img/255.0
        img = img ** 2.2
        img = downsample_antialiased(img, factor=2)
        img = add_poisson_noise(img,max_photons=30,readout_std=2.0)
        img = img ** (1/2.2)
        #
        imgTrue = Image.open("../data/hongo0858.png").convert("RGB") 
        imgTrue = np.array(imgTrue)
        imgTrue = imgTrue/255.0
        imgTrue  = downsample_antialiased(imgTrue , factor=2)
    return img, imgTrue


def add_poisson_noise(img, max_photons=30, readout_std=2.0):

    img = np.clip(img, 0.0, 1.0)
    lam = max_photons * img
    poisson_noise = np.random.poisson(lam)

    readout_noise = np.random.normal(loc=0.0, scale=readout_std, size=img.shape)
    noisy_counts = poisson_noise + readout_noise
    noisy_img = noisy_counts / max_photons

    return np.clip(noisy_img, 0.0, 1.0)



def downsample_antialiased(img, factor, sigma=None):
    if sigma is None:
        sigma = factor / 2
    img_blur = ndi.gaussian_filter(img, sigma=(sigma, sigma, 0))
    return img_blur[::factor, ::factor, :]

def make_image_dataset(img):
    H, W, C = img.shape
    xs = jnp.linspace(-1, 1, W)
    ys = jnp.linspace(-1, 1, H)
    X, Y = jnp.meshgrid(xs, ys, indexing="xy")

    coords = jnp.stack([X, Y], axis=-1)   # (H, W, 2)
    coords = coords.reshape(-1, 2)

    values = img.reshape(-1, C)
    return coords, values

if __name__ == "__main__":
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    """
    img = np.load('../data/data_div2k.npz')
    img = img['test_data']
    
    plt.imshow(img[0])
    plt.title(f"{0}-{img[0].shape}")
    plt.show()
    plt.imshow(img[8])
    plt.title(f"{8}-{img[8].shape}")
    plt.show()"""

    """
    name = "hongo0858"

    img = Image.open(f"../data/{name}.png").convert("RGB") 
    img = np.array(img)[48:-48,70:-70]
    #img, _ = load_image("super3")
    print (img.shape)
    plt.imshow(img)
    plt.title(f"{name}-{img.shape}")
    plt.show()"""

    img, imgTrue = load_image("noise")   
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("Input image")
    plt.subplot(1,2,2)
    plt.imshow(imgTrue)
    plt.title("Ground truth image")
    plt.show()
    psnr_val = psnr(imgTrue, img, data_range=1.0)
    ssim_val = ssim(imgTrue, img, channel_axis=-1, data_range=1.0)
    print (f"PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")

