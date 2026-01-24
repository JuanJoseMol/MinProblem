import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from matplotlib import pyplot as plt

# Parameters
n = 240
T = 1  # fixed value for T
a = 22 # parameter a
low = 0.001

x = np.linspace(0, a, int(n/2))

def func(x):
  freq = 2.1
  sin = np.sin(2*np.pi*freq*x)
  return np.sign(sin)*(np.abs(sin) > 0.5)


y = np.linspace(-1,1,n)
g  = np.fft.fft(func(y))[0:int(n/2)]#np.fft.fftshift(np.fft.fft(func(y)))[0:int(n/2)]

vmin, vmax =np.min(np.abs(g)),np.max(np.abs(g))
print (np.count_nonzero(np.abs(g)**2<0.1) )
plt.plot(np.abs(g))
plt.ylim(vmin,vmax)
#plt.yscale('log')
plt.show()

print (np.min(np.abs(g)),np.max(np.abs(g)))

# Objective function to minimize
def objective(f_coeffs):
    f = np.poly1d(f_coeffs)  # Assuming f is represented as a polynomial
    integrand = lambda x: np.interp(x, np.linspace(0, a, int(n/2)), np.abs(g)**2+0.00000001)* np.exp(-2 * f(x) * T)
    result, _ = quad(integrand, 0, a, limit=100)
    return result

# Constraint: integral(f) = 1
def constraint_integral(f_coeffs):
    f = np.poly1d(f_coeffs)
    integral_value, _ = quad(f, 0, a)
    return np.maximum(0,np.abs(integral_value - .5) - 0.00001)

def constraint_positivity(f_coeffs):
    f = np.poly1d(f_coeffs)
    # Sample points in the domain [0, a] to ensure f(x) is positive
    x_sample = np.linspace(0, a, int(n/2))
    return f(x_sample)- low

# Initial guess for the coefficients of f (e.g., as a polynomial)
initial_guess = np.random.rand(10)  # For a quadratic polynomial, for example

# Set up the constraint as a dictionary
constraints = [
    {'type': 'eq', 'fun': constraint_integral},
    {'type': 'ineq', 'fun': constraint_positivity}
]

# Perform the optimization
result = minimize(objective, initial_guess, constraints=constraints, options={'ftol': 1e-9, 'disp': True})

# The optimal coefficients for f
f_optimal = np.poly1d(result.x)
comma_separated = ', '.join(map(str, result.x))
print(comma_separated)
print (f_optimal)

z = np.linspace(0, a, int(n/2))
plt.plot(z, f_optimal(z))
plt.show()

print (quad(f_optimal, 0, a))