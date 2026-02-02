# ============================================================
# Full script: density_experiment_L2.py
# ============================================================
# Computes finite-sample L2 error of optimal density estimates
# for Brain (2D), 3D Brain volume, and Human datasets
# and exports ONE LaTeX table with all three cases.
# ============================================================

import os
import numpy as np
import nibabel as nib
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import resize

# ============================================================
# 1. Core numerical routines
# ============================================================

def target_function(x, g_array, T, area):
    integrand = np.maximum((np.log(g_array) - np.log(x)) / (2 * T), 0)
    return np.sum(integrand) * area / g_array.size


def bisection_method(g_array, T, area, tol=1e-6, max_iter=100):
    a, b = 1e-8, 1e8
    for _ in range(max_iter):
        c = 0.5 * (a + b)
        f_c = target_function(c, g_array, T, area)

        if abs(f_c - 1) < tol:
            return c

        f_a = target_function(a, g_array, T, area)
        if np.sign(f_c - 1) == np.sign(f_a - 1):
            a = c
        else:
            b = c

    raise RuntimeError("Bisection did not converge")


def compute_rho_from_g(g, T, area):
    x_root = bisection_method(g, T, area)
    return np.where(g >= x_root, np.log(g / x_root) / (2 * T), 0)


def l2_error(rho_hat, rho_ref):
    return np.sqrt(np.mean((rho_hat - rho_ref) ** 2))/np.sqrt(np.mean(rho_ref ** 2))

def l1_error(rho_hat, rho_ref):
    return np.mean(np.abs(rho_hat - rho_ref) )


# ============================================================
# 3. Training-set density extraction
# ============================================================

def collect_rhos_brain(base_dirs, max_samples=450):
    rho_list = []
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith("T1w.nii.gz"):
                    path = os.path.join(root, file)

                    nii = nib.load(path).get_fdata()
                    im = np.swapaxes(nii[7:187, 16:216, 120], 0, 1)
                    im = im / 255.0
                    im -= im.mean()

                    g = np.abs(np.fft.fftshift(np.fft.fft2(im))) ** 2
                    rho_list.append(compute_rho_from_g(g, T=1.8, area=4))

                    if len(rho_list) >= max_samples:
                        return rho_list
    return rho_list


def collect_rhos_3d(base_dirs, max_samples=450):
    rho_list = []
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith("T1w.nii.gz"):
                    path = os.path.join(root, file)

                    nii = nib.load(path).get_fdata()
                    im = np.swapaxes(nii[7:187, 16:216, 5:185], 0, 1)
                    im = resize(im, (50,45,45))
                    im = im / 255.0
                    im -= im.mean()

                    g = np.abs(np.fft.fftshift(np.fft.fftn(im))) ** 2
                    rho_list.append(compute_rho_from_g(g, T=1.2, area=8))

                    if len(rho_list) >= max_samples:
                        return rho_list
    return rho_list


def collect_rhos_human(base_dirs, max_samples=450):
    rho_list = []
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    path = os.path.join(root, file)

                    img = Image.open(path)
                    img_array = np.array(img)

                    s = (150, 300)
                    im1 = Image.fromarray(img_array[:, :, 0]).resize(s, Image.LANCZOS)
                    im2 = Image.fromarray(img_array[:, :, 1]).resize(s, Image.LANCZOS)
                    im3 = Image.fromarray(img_array[:, :, 2]).resize(s, Image.LANCZOS)

                    im = np.concatenate(
                        (
                            np.array(im1)[:, :, None],
                            np.array(im2)[:, :, None],
                            np.array(im3)[:, :, None],
                        ),
                        axis=2,
                    ).astype("float64")

                    im[:, :, 0] /= 255.0
                    im[:, :, 1] /= 255.0
                    im[:, :, 2] /= 255.0

                    im[:, :, 0] -= im[:, :, 0].mean()
                    im[:, :, 1] -= im[:, :, 1].mean()
                    im[:, :, 2] -= im[:, :, 2].mean()

                    g = np.abs(np.fft.fftshift(np.fft.fft2(im[:, :, 0]))) ** 2
                    rho_list.append(compute_rho_from_g(g, 0.75, 4))

                    if len(rho_list) >= max_samples:
                        return rho_list

    return rho_list



# ============================================================
# 2. Reference densities
# ============================================================
brain_rhos = np.array(collect_rhos_brain(["dataset/ATLAS_2/"]))#testbrain/"]))#
rhos_3d = np.array(collect_rhos_3d(["dataset/ATLAS_2/"])) #testbrain/"]))#
human_rhos = np.array(collect_rhos_human(["dataset/Humans/"])) #testhuman/"]))#
print (brain_rhos.shape)
print (rhos_3d.shape)
print (human_rhos.shape)

rho_ref_3d = np.mean(rhos_3d, axis=0)
rho_ref_brain = np.mean(brain_rhos, axis=0)
rho_ref_human = np.mean(human_rhos, axis=0)

print (l1_error(rho_ref_3d,0))
print (l1_error(rho_ref_brain,0))
print (l1_error(rho_ref_human,0))

"""
# ---------- Brain / 3D reference ----------
A = np.loadtxt("dataset/Mean-197-233-189-brain.txt").reshape((197, 233, 189))

# 3D
im_3d = np.swapaxes(A[7:187, 16:216, 5:185], 0, 1)
im_3d = resize(im_3d, (50,45,45))
im_3d = im_3d/255.0
im_3d -= im_3d.mean()
g_ref_3d = np.abs(np.fft.fftshift(np.fft.fftn(im_3d))) ** 2
rho_ref_3d = compute_rho_from_g(g_ref_3d, T=1.2, area=8)

# Brain 2D
im_brain = np.swapaxes(A[7:187, 16:216, 120], 0, 1)
im_brain = im_brain/255.0
im_brain -= im_brain.mean()
g_ref_brain = np.abs(np.fft.fftshift(np.fft.fft2(im_brain))) ** 2
rho_ref_brain = compute_rho_from_g(g_ref_brain, T=1.8, area=4)

# ---------- Human reference ----------
H = np.loadtxt("dataset/Mean-300-150-human.txt").reshape((300, 150, 3))
g_ref_human = np.abs(np.fft.fftshift(np.fft.fft2(H[:, :, 0]/255.0))) ** 2
rho_ref_human = compute_rho_from_g(g_ref_human, T=0.75, area=4)
"""

# ============================================================
# 4. Finite-sample experiment
# ============================================================

def finite_sample_errors(rho_list, rho_ref, N_values, reps=10, seed=0):
    rng = np.random.default_rng(seed)
    rho_array = np.stack(rho_list)
    errors = {}

    for N in N_values:
        vals = []
        for _ in range(reps):
            idx = rng.choice(len(rho_array), N, replace=False)
            rho_hat = rho_array[idx].mean(axis=0)
            vals.append(l2_error(rho_hat, rho_ref))
        errors[N] = (np.mean(vals), np.std(vals))
        if rho_list[-1].shape == rho_ref_brain.shape:
            print ("Saving brain rho_hat")
            with open('dataset/N-'+str(N)+'-T1.8Mean-rhoBrain.txt', 'w') as f: 
                np.savetxt(f, rho_hat)
        if rho_list[-1].shape == rho_ref_3d.shape:
            print ("Saving 3d rho_hat")
            with open('dataset/N-'+str(N)+'-T1.2Mean-rho3d.txt', 'w') as f: 
                np.savetxt(f, rho_hat.reshape(50,-1))
        if rho_list[-1].shape == rho_ref_human.shape:
            print ("Saving human rho_hat")
            with open('dataset/N-'+str(N)+'-T0.75Mean-rhoHuman.txt', 'w') as f: 
                np.savetxt(f, rho_hat)
    return errors


# ============================================================
# 5. Run experiment
# ============================================================

N_values = [1,2, 4, 8, 16, 32, 64]

brain_rhos = collect_rhos_brain(["dataset/ATLAS_2/"])
rhos_3d = collect_rhos_3d(["dataset/ATLAS_2/"])
human_rhos = collect_rhos_human(["dataset/Humans/"])

errors_brain = finite_sample_errors(brain_rhos, rho_ref_brain, N_values)
errors_3d = finite_sample_errors(rhos_3d, rho_ref_3d, N_values)
errors_human = finite_sample_errors(human_rhos, rho_ref_human, N_values)


# ============================================================
# 6. Export LaTeX table
# ============================================================

with open("results2/density_relativeL2_error.tex", "w") as f:
    f.write("\\begin{table}[t]\n")
    f.write("\\centering\n")
    f.write("\\caption{Finite-sample $L^2$ error of optimal density estimation.}\n")
    f.write("\\label{tab:density-l2}\n")
    f.write("\\begin{tabular}{c c c c}\n")
    f.write("\\toprule\n")
    f.write("$N$ & Brain & 3D & Human \\\\\n")
    f.write("\\midrule\n")

    for N in N_values:
        b = errors_brain[N]
        d = errors_3d[N]
        h = errors_human[N]
        f.write(
            f"{N} & "
            f"${b[0]:.3f} \\pm {b[1]:.4f}$ & "
            f"${d[0]:.3f} \\pm {d[1]:.4f}$ & "
            f"${h[0]:.3f} \\pm {h[1]:.4f}$ \\\\\n"
        )

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print("LaTeX table saved to density_L2_error.tex")
