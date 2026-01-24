import numpy
import jax.numpy as np
import jax
from scipy.optimize import minimize
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray
from coordgrid import meshgrid_from_subdiv, flatten_all_but_lastdim
from PIL import Image

# Parameters
T = 1  # fixed value for T
a = 1  # Here, 'a' represents the boundary of the domain in both x and y

# Objective function to minimize





#ej = data.camera()
ej2 = data.astronaut()
im = rgb2gray(ej2)
N_s = im.shape[0]
size = (N_s, N_s)
#ima = Image.fromarray(im)
#resized = ima.resize(size, Image.LANCZOS)
#im = np.array(resized)
#del ima
#del resized
#im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)#np.array(im, dtype=np.float32)
ft = np.fft.fftshift(np.fft.fft2(im))
# Define the grid
x = np.linspace(-a, a, N_s)
y = np.linspace(-a, a, N_s)
"""
X, Y = np.meshgrid(x,x)
R = np.sqrt(X**2 + Y**2)
ej1 = onp.array(f(R))
ej1[R > 0.99] = 0"""




# Assuming g is a 2D array evaluated on the grid
g = np.abs(ft)  # Ensuring g is positive and non-zero for numerical stability



"""
plt.imshow(im)
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()

plt.imshow(np.log(g))
plt.colorbar(fraction=0.046, pad=0.04)
plt.show()
"""


"""
# Objective function to minimize
@jax.jit
def objective(f_values):
    # Reshape f_values back into a 2D function f(x, y)
    f = f_values.reshape((N_s, N_s))
    integrand = (np.abs(g)**2+ 1e-5 ) * np.exp(-2 * f * T)
    result = np.trapz(np.trapz(integrand, x), y)  # Double integration using trapezoidal rule
    return result

# Constraint: double integral(f) = 1
def constraint_integral(f_values):
    f = f_values.reshape((N_s, N_s))
    integral_value = np.trapz(np.trapz(f, x), y)  # Double integration using trapezoidal rule
    return np.maximum(0,np.abs(integral_value - 1) - 0.00001)

# Positivity constraint: f(x, y) >= 0
def constraint_positivity(f_values):
    return f_values -0.1 # Ensure all values in f_values are non-negative

# Initial guess for f(x, y) as a flat array
initial_guess = numpy.random.rand(N_s * N_s)

# Set up the constraints
constraints = [
    {'type': 'eq', 'fun': constraint_integral},
    {'type': 'ineq', 'fun': constraint_positivity}
]
der = jax.jit(jax.grad(lambda x: objective(x)))
# Perform the optimization
result = minimize(objective, initial_guess, jac= der, constraints=constraints, options={'maxiter': 100})

# The optimal f(x, y)
f_optimal = result.x.reshape((N_s, N_s))

# Plotting the result
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.imshow(f_optimal)
plt.colorbar()
plt.title('Optimized Function f(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""
min_value = 0.1
x = np.linspace(-1, 1, N_s)
y = np.linspace(-1, 1, N_s)
X, Y = np.meshgrid(x, y)
n = 7  # You can adjust this degree

# Number of coefficients for a polynomial of degree n in 2D
num_coeffs = ((n // 2) + 1) * ((n // 2) + 1)

# Basis function for the polynomial (monomials)
def polynomial_basis(x, y, n):
    terms = []
    for i in range(0, n+1, 2):
        for j in range(0, n+1, 2):
            terms.append((x**i) * (y**j))
    return np.array(terms)

# Objective function to minimize
def objective(coeffs):
    f_values = np.zeros_like(X)
    for i, coeff in enumerate(coeffs):
        f_values += coeff * polynomial_basis(X, Y, n)[i]
    integrand = (np.abs(g)**2+0.00001) * np.exp(-2 * f_values * T)
    result = np.trapz(np.trapz(integrand, x), y)  # Double integration using trapezoidal rule
    return result

# Constraint: double integral(f) = 1
def constraint_integral(coeffs):
    f_values = np.zeros_like(X)
    for i, coeff in enumerate(coeffs):
        f_values += coeff * polynomial_basis(X, Y, n)[i]
    integral_value = np.trapz(np.trapz(f_values, x), y)
    return integral_value - 1

# Positivity constraint: f(x, y) >= min_value
def constraint_positivity(coeffs):
    f_values = np.zeros_like(X)
    for i, coeff in enumerate(coeffs):
        f_values += coeff * polynomial_basis(X, Y, n)[i]
    return f_values.flatten() - min_value

# Initial guess for the coefficients
initial_guess = numpy.random.rand(num_coeffs)

# Set up the constraints
constraints = [
    {'type': 'eq', 'fun': constraint_integral},
    {'type': 'ineq', 'fun': constraint_positivity}
]

# Perform the optimization
result = minimize(objective, initial_guess, constraints=constraints, method='SLSQP')

# Extract the optimal coefficients
f_optimal_coeffs = result.x

# Reconstruct the optimal function f(x, y) for plotting
f_optimal = np.zeros_like(X)
for i, coeff in enumerate(f_optimal_coeffs):
    f_optimal += coeff * polynomial_basis(X, Y, n)[i]

print (np.trapz(np.trapz(f_optimal, x), y))
# Plotting the result
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.imshow(f_optimal)
#plt.contourf(x, y, f_optimal, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Optimized Function f(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()