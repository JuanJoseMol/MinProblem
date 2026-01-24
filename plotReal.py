import numpy as np
import glob
import re
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage as ndi
from loadRealistic import load_image, make_image_dataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import pickle
import jax.numpy as jnp
import jax
from jax.nn import relu
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def ssim(gt, pred):
    H, W = gt.shape[:2]
    min_side = min(H, W)

    if min_side < 3:
        return np.nan  # too small to compute SSIM safely

    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1

    return structural_similarity(
        gt,
        pred,
        data_range=1.0,
        channel_axis=-1,
        win_size=win_size,
    )

def _patch_numpy_for_pickle():
    import numpy
    import numpy.core as _core

    # Backward compatibility for old pickles
    if not hasattr(numpy, "_core"):
        numpy._core = _core

class NumpyCompatUnpickler(pickle.Unpickler):
    """
    Unpickler that remaps legacy NumPy module paths.
    """

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")

        return super().find_class(module, name)


def load_pickle_pred(path, verbose=False):
    if verbose:
        import numpy as np
        print(f"[INFO] Loading {path}")
        print(f"[INFO] NumPy version: {np.__version__}")

    try:
        with open(path, "rb") as f:
            return pickle.load(f)

    except ModuleNotFoundError as e:
        if "numpy._core" in str(e):
            if verbose:
                print("[WARN] Legacy NumPy pickle detected")
                print("       Applying compatibility unpickler...")

            with open(path, "rb") as f:
                return NumpyCompatUnpickler(f).load()
        raise

def make_grid(H, W):
    y = jnp.linspace(-1.0, 1.0, W)
    x = jnp.linspace(-1.0, 1.0, H)
    yy, xx = jnp.meshgrid(x, y, indexing="ij")
    grid = jnp.stack([xx, yy], axis=-1)
    return grid.reshape(-1, 2)

def FF_forward(X, params):
    Ws, bs = params
    
    H = jnp.dot(X, Ws[0])
    H = jnp.concatenate([jnp.sin(H),
                          jnp.cos(H)], axis=-1)
    for i in range(1, len(Ws)-1):
        H = jnp.dot(H, Ws[i]) + bs[i-1]
        H = relu(H)  
    return jnp.dot(H, Ws[-1])


def Gaussian_forward(X, params, alpha=30.0):
    Ws, bs = params

    H = X
    for i in range(len(Ws)-1):
        Z = jnp.matmul(H, Ws[i]) + bs[i]
        H = jnp.exp(-jnp.abs(alpha* Z)**2)

    return jnp.matmul(H, Ws[-1])+ bs[-1]

def SIREN_forward(X, params, w0=40.0):
    Ws, bs = params

    H = jnp.sin(w0 * (jnp.matmul(X, Ws[0]) + bs[0]))
    for i in range(1, len(Ws)-1):
        H = jnp.sin(w0 *jnp.matmul(H, Ws[i]) + bs[i])
    return jnp.matmul(H, Ws[-1])

def complex_gabor(z, omega0, sigma0):
    return jnp.sin(omega0 * z)* jnp.exp( - (jnp.abs(sigma0 * z)) ** 2)

def WIRE_forward(X, params, omega0=20.0, sigma0=10.0):

    Ws, bs = params
    z = jnp.matmul(X, Ws[0])  + bs[0] 
    H = complex_gabor(z, omega0, sigma0)
    
    for i in range(1, len(Ws) - 1):
        z = jnp.matmul(H, Ws[i]) + bs[i]
        H = complex_gabor(z, omega0, sigma0)
    
    output = jnp.matmul(H, Ws[-1]) + bs[-1]
    return output 

def eval_ff_image(params, H, W, chunk=8192):
    X = make_grid(H, W)
    Ys = []

    for i in range(0, X.shape[0], chunk):
        Ys.append(FF_forward(X[i:i+chunk], params))

    Y = jnp.concatenate(Ys, axis=0)
    return np.array(Y).reshape(H, W, 3)

def eval_siren_image(params, H, W, w0=40.0, chunk=8192):
    X = make_grid(H, W)
    Ys = []

    for i in range(0, X.shape[0], chunk):
        Ys.append(SIREN_forward(X[i:i+chunk], params, w0))

    Y = jnp.concatenate(Ys, axis=0)
    return np.array(Y).reshape(H, W, 3)

def eval_gauss_image(params, H, W, alpha=30.0, chunk=8192):
    X = make_grid(H, W)
    Ys = []

    for i in range(0, X.shape[0], chunk):
        Ys.append(Gaussian_forward(X[i:i+chunk], params, alpha))

    Y = jnp.concatenate(Ys, axis=0)
    return np.array(Y).reshape(H, W, 3)

def eval_wire_image(params, H, W, omega0=20.0, sigma0=10.0, chunk=8192):
    X = make_grid(H, W)
    Ys = []

    for i in range(0, X.shape[0], chunk):
        Ys.append(WIRE_forward(X[i:i+chunk], params, omega0, sigma0))

    Y = jnp.concatenate(Ys, axis=0)
    return np.array(Y).reshape(H, W, 3)


def compute_metrics(gt, pred):
    data_range = gt.max() - gt.min()
    psnr_v = peak_signal_noise_ratio(gt, pred, data_range=data_range)
    ssim_v = ssim(gt, pred)
    return psnr_v, ssim_v



def plot_image_with_zoom(
    ax,
    img,
    roi_x, roi_y, roi_w, roi_h,
    gt=None,
    show_metrics=False,
):
    """
    Plot image with red ROI and zoom inset (no title, flush bottom-right).
    Optionally compute and display PSNR / SSIM.
    """
    ax.imshow(img)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---- Red ROI rectangle ----
    rect = patches.Rectangle(
        (roi_x, roi_y),
        roi_w,
        roi_h,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)

    # ---- Zoom inset (flush bottom-right) ----
    axins = inset_axes(
        ax,
        width="38%",
        height="38%",
        loc="lower right",
        borderpad=0,
    )

    zoom = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    axins.imshow(zoom)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.axis("off")

    # ---- Metrics (only for predictions) ----
    if show_metrics and gt is not None:
        psnr, ssim = compute_metrics(gt, img)
        ax.text(
            0.02, 0.95,
            f"PSNR: {psnr:.2f}\nSSIM: {ssim:.4f}",
            transform=ax.transAxes,
            va="top", ha="left",
            color="white", fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6, pad=3),
        )


"""
# Sigma values of interest
#sigmaW_list = [6, 20, 30, 35, 40, 50]
sigmaW_list = [70,90,98,110,130]

# Dictionary: sigmaW -> list of loss curves
loss_dict = {s: [] for s in sigmaW_list}

pattern = re.compile(r"w0(\d+)-k(\d+)\.npy")

for fname in glob.glob("realistic/lossSIREN-fitting-lr0.01-w0*-k*.npy"):#lossFF-fitting-lr0.01-sigmaW*-k*.npy"):
    match = pattern.search(fname)
    if match:
        sigmaW = int(match.group(1))
        if sigmaW in loss_dict:
            loss = np.load(fname)
            loss_dict[sigmaW].append(loss)

# Sanity check
for s in sigmaW_list:
    print(f"sigmaW={s}: {len(loss_dict[s])} runs")

stats = {}

for s, runs in loss_dict.items():
    runs = np.array(runs)  # shape: (n_runs, n_iters)
    
    if runs.size == 0:
        continue

    median = np.median(runs, axis=0)
    best   = np.min(runs, axis=0)
    worst  = np.max(runs, axis=0)

    stats[s] = {
        "median": median,
        "best": best,
        "worst": worst
    }
plt.figure(figsize=(8, 6))

for s in sigmaW_list:
    if s not in stats:
        continue

    iters = np.arange(len(stats[s]["median"]))

    plt.plot(
        iters,
        stats[s]["median"],
        label=f"sigmaW={s}"
    )

    plt.fill_between(
        iters,
        stats[s]["best"],
        stats[s]["worst"],
        alpha=0.2
    )

plt.yscale("log")  # typical for loss curves
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.title("Training loss vs iteration\nMedian with best/worst envelope")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig('realistic/fitting-losses.pdf', bbox_inches='tight')
plt.show()
"""
################################## GT images plot ##################################

"""
fig, axes = plt.subplots(1, 3, figsize=(12, 4))


img3, imgTrue = load_image("noise")   
img2, imgTrue = load_image("super")   
img1, imgTrue = load_image("fitting") 

images = [img1, img2, img3]
titles = ["Image 1", "Image 2", "Image 3"]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img)
    ax.set_aspect("auto")   # 🔑 force same subplot shape
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig('realistic/GTimages.pdf', bbox_inches='tight')
plt.show()
"""


################################## Predictions plot ##################################
"""

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

#img3 = np.load("realistic/predFF-fitting-lr0.01-sigmaW50-k0.npy")
#img2 = np.load("realistic/predFF-fitting-lr0.01-sigmaW35-k0.npy")
#img1 = np.load("realistic/predFF-fitting-lr0.01-sigmaW6-k0.npy")
img3 = np.load("realistic/predSIREN-fitting-lr0.01-w070-k4.npy")
img2 = np.load("realistic/predSIREN-fitting-lr0.01-w090-k0.npy")
img1 = np.load("realistic/predSIREN-fitting-lr0.01-w070-k0.npy")
print (img1.shape, img2.shape, img3.shape)
img1 = img1.reshape((512, 512, 3))
img2 = img2.reshape((512, 512, 3))
img3 = img3.reshape((512, 512, 3))

images = [img1, img2, img3]
titles = ["Image 1", "Image 2", "Image 3"]

for ax, img, title in zip(axes, images, titles):
    if img.ndim == 2:          # grayscale
        ax.imshow(img, cmap="gray")
    else:                     # color (RGB)
        ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
#plt.savefig('realistic/fitting-preds.pdf', bbox_inches='tight')
plt.show()
"""


################################ losses ###############################
"""
folder = "realistic"
prefix = "lossFFsgd"

plt.figure(figsize=(6, 4))

for fname in sorted(os.listdir(folder)):
    if fname.startswith(prefix) and fname.endswith(".npy"):
        #if "lr0.1" in fname:
        #    continue
        path = os.path.join(folder, fname)
        loss = np.load(path)          # 1D array
        plt.plot(loss, alpha=0.4)     # alpha helps when many curves

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("FF + SGD loss curves (realistic)")
plt.grid(True)
plt.tight_layout()
plt.show()
"""

"""


img1, img1True = load_image("noise")
img2, img2True = load_image("noise2")
img3, img3True = load_image("noise3")

fig, axes = plt.subplots(1, 3, figsize=(9, 4))
plt.subplot(1,3,1)
plt.imshow(img1)
psnr, ssim = compute_metrics(img1True, img1)
plt.axis('off')
plt.title(f"PSNR: {psnr:.2f} SSIM: {ssim:.3f}", fontsize=13)
plt.subplot(1,3,2)
plt.imshow(img2)
psnr, ssim = compute_metrics(img2True, img2)
plt.axis('off')
plt.title(f"PSNR: {psnr:.2f} SSIM: {ssim:.3f}", fontsize=13)
plt.subplot(1,3,3)
plt.imshow(img3)
psnr, ssim = compute_metrics(img3True, img3)
plt.axis('off')
plt.title(f"PSNR: {psnr:.2f} SSIM: {ssim:.3f}", fontsize=13) 
plt.tight_layout()
plt.show()

"""


#########################################################################################################
################################### New #################################################################
#########################################################################################################

def build_filename(model, mode, long, lr, param, value, version="", k=1):
    """
    model   ∈ {"WIRE", "Gauss", "SIREN"}
    mode    ∈ {"fitting", "super", "noise"}
    param   ∈ {"s", "a", "w"}   (sigma, alpha, omega)
    version ∈ {"", "2", "3"}
    """
    return (
        f"realistic/{model}_adam{mode}_{long}--lr{lr}-{param}{value}-k{k}.pickle"
    )

def load_model_case(model, mode, long=50000, lr=0.001, version="", k=1):
    if model == "WIRE":
        param, value = "s", "010.0"
    elif model == "Gauss":
        param, value = "a", "30.0"
    elif model == "SIREN":
        lr=0.0001
        param, value = "w", "040"
    else:
        raise ValueError(f"Unknown model: {model}")

    fname = build_filename(
        model=model,
        mode=mode,
        long=long,
        lr=lr,
        param=param,
        value=value,
        version=version,
        k=k,
    )
    return load_pickle_pred(fname)


v = ""
long = 50000
long2 = 500

img_noise, img_noise_gt = load_image(f"noise{v}")
img_super, img_super_gt = load_image(f"super{v}")
img_fit,   img_fit_gt   = load_image(f"fitting{v}")
#new2ourFFadamsuper2_50000--lr0.001-s114-k1.pickle

if v == "":
    dic_fit   = load_pickle_pred(f"realistic/new2ourFFadamfitting{v}_{long}--lr0.001-s112-k1.pickle")
    dic_super = load_pickle_pred(f"realistic/new2ourFFadamsuper{v}_{long}--lr0.001-s110-k1.pickle")
    dic_noise = load_pickle_pred(f"realistic/new2adamFFadamnoise{v}_{long2}--lr0.001-s137-k1.pickle")#73
if v == "2":
    dic_fit   = load_pickle_pred(f"realistic/new2ourFFadamfitting{v}_{long}--lr0.001-s97-k1.pickle")
    dic_super = load_pickle_pred(f"realistic/new2ourFFadamsuper{v}_{long}--lr0.001-s114-k1.pickle")
    dic_noise = load_pickle_pred(f"realistic/new2adamFFadamnoise{v}_{long2}--lr0.001-s89-k1.pickle")#44
if v == "3":
    dic_fit   = load_pickle_pred(f"realistic/new2ourFFadamfitting{v}_{long}--lr0.001-s119-k1.pickle")
    dic_super = load_pickle_pred(f"realistic/new2ourFFadamsuper{v}_{long}--lr0.001-s155-k1.pickle")
    dic_noise = load_pickle_pred(f"realistic/new2adamFFadamnoise{v}_{long2}--lr0.001-s91-k1.pickle")##46
dic_fit_SIREN   = load_model_case("SIREN", f"fitting{v}")
dic_super_SIREN = load_model_case("SIREN", f"super{v}")
dic_noise_SIREN = load_model_case("SIREN", f"noise{v}", long=long2)
dic_fit_GAUSS   = load_model_case("Gauss", f"fitting{v}")
dic_super_GAUSS = load_model_case("Gauss", f"super{v}")
dic_noise_GAUSS = load_model_case("Gauss", f"noise{v}", long=long2)
dic_fit_WIRE   = load_model_case("WIRE",  f"fitting{v}")
dic_super_WIRE = load_model_case("WIRE",  f"super{v}")
dic_noise_WIRE = load_model_case("WIRE",  f"noise{v}", long=long2)


pred1 = dic_fit["pred"].reshape(img_fit_gt.shape)
pred3 = dic_noise["pred"].reshape(img_noise_gt.shape)
pred1siren = dic_fit_SIREN["pred"].reshape(img_fit_gt.shape)
pred3siren = dic_noise_SIREN["pred"].reshape(img_noise_gt.shape)
pred1gaus = dic_fit_GAUSS["pred"].reshape(img_fit_gt.shape)
pred3gaus = dic_noise_GAUSS["pred"].reshape(img_noise_gt.shape)
pred1wire = dic_fit_WIRE["pred"].reshape(img_fit_gt.shape)
pred3wire = dic_noise_WIRE["pred"].reshape(img_noise_gt.shape)

H2, W2, _ = img_super_gt.shape
params_super = dic_super["param"]
pred2 = eval_ff_image(params_super, H2, W2)
params_super_siren = dic_super_SIREN["param"]
pred2siren = eval_siren_image(params_super_siren, H2, W2)
params_super_gauss = dic_super_GAUSS["param"]
pred2gaus = eval_gauss_image(params_super_gauss, H2, W2)
params_super_wire = dic_super_WIRE["param"]
pred2wire = eval_wire_image(params_super_wire, H2, W2)

pred1  = np.clip(pred1,  0.0, 1.0)
pred2  = np.clip(pred2,  0.0, 1.0)
pred3  = np.clip(pred3,  0.0, 1.0)
pred1siren = np.clip(pred1siren, 0.0, 1.0)
pred2siren = np.clip(pred2siren, 0.0, 1.0)
pred3siren = np.clip(pred3siren, 0.0, 1.0)
pred1gaus = np.clip(pred1gaus, 0.0, 1.0)
pred2gaus = np.clip(pred2gaus, 0.0, 1.0)
pred3gaus = np.clip(pred3gaus, 0.0, 1.0)
pred1wire = np.clip(pred1wire, 0.0, 1.0)
pred2wire = np.clip(pred2wire, 0.0, 1.0)
pred3wire = np.clip(pred3wire, 0.0, 1.0)

images_gt    = [img_fit_gt, img_super_gt, img_noise_gt]
images_FF     = [pred1,  pred2,  pred3]
images_SIREN  = [pred1siren, pred2siren, pred3siren]
images_GAUSS = [pred1gaus, pred2gaus, pred3gaus]
images_WIRE  = [pred1wire, pred2wire, pred3wire]


methods = ["GT", "FF", "SIREN", "GAUSS", "WIRE"]

images_methods = [
    images_gt,
    images_FF,
    images_SIREN,
    images_GAUSS,
    images_WIRE,
]

row_titles = ["Fitting", "Super-resolution", "Denoising"]

fig, axes = plt.subplots(3, 5, figsize=(15, 7))

for i in range(3):  # rows = tasks
    gt = images_gt[i]

    for j in range(5):  # columns = methods
        ax = axes[i, j]
        im = images_methods[j][i]

        H, W, _ = gt.shape
        roi_x, roi_y = 2 * H // 5, 2 * W // 5
        roi_w = roi_h = H // 10

        if methods[j] == "GT":
            plot_image_with_zoom(
                ax,
                im,
                roi_x, roi_y, roi_w, roi_h,
            )
        else:
            plot_image_with_zoom(
                ax,
                im,
                roi_x, roi_y, roi_w, roi_h,
                gt=gt,
                show_metrics=True,
            )

        if i == 0:
            ax.set_title(methods[j], fontsize=13)

        if j == 0:
            ax.set_ylabel(row_titles[i], fontsize=13)

y_positions = [0.81,0.50, 0.17]

for label, y in zip(row_titles, y_positions):
    fig.text(
        0.02, y,
        label,
        va="center",
        ha="left",
        fontsize=13,
        rotation=90,
    )
# =========================
# FINAL LAYOUT
# =========================
plt.tight_layout(rect=[0.03, 0.0, 1.0, 1.0])
plt.savefig(f"realistic/figs/2fig_1-{v}-{long2}.pdf", bbox_inches="tight")
plt.show()
