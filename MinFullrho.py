import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from matplotlib import pyplot as plt

# Parameters
T = 0.05  # fixed value for T
a = 1 # parameter a
low = 0.001
#x = np.linspace(0, a, 120)
sigma_a = np.sqrt(2/2000)
n = 120

def func(x):
  freq = 2.1
  sin = np.sin(2*np.pi*freq*x)
  return np.sign(sin)*(np.abs(sin) > 0.5)

y = np.linspace(-1,1,int(2*n))
g  = np.fft.fft(func(y))[0:n]
"""
plt.plot(x, np.abs(g))
plt.yscale('log')
plt.show()

print (np.min(np.abs(g)),np.max(np.abs(g)))
coeffs = np.random.rand(5)
f_coeffs = []
for i in range(2*len(coeffs)-1):
    #if i == 0:
    #    f_coeffs.append(1)
    #    f_coeffs.append(0)
    if i % 2 == 0:
        f_coeffs.append(coeffs[int(i/2)])
    else:
        f_coeffs.append(0)
print (f_coeffs)
print (len(f_coeffs))

f = np.poly1d(f_coeffs)
z = np.linspace(-a, a, 120)
plt.plot(z, f(z))
plt.show()

# Objective function to minimize

def objective(f_coeffs):
    f = np.poly1d(f_coeffs)  # Assuming f is represented as a polynomial
    integrand = (2*np.pi*quad (lambda x, y: quad( lambda x, y: (np.sinc(2*(x-y)))**2 *f(y), 0, a, 120)[0]*np.exp(-2 * f(x) * T), 0, a, 120)[0]
            + quad(np.interp(x, np.linspace(0, a, 120), np.abs(g)**2+0.00000001)* np.exp(-2 * f(x) * T), 0, a)[0])
    #result, _ = quad(integrand, 0, a, limit=100)
    return integrand#result"""
"""
def objective(coeffs):
    f_coeffs = []
    for i in range(2*len(coeffs)-1):
        #if i == 0:
        #    f_coeffs.append(1)
        #    f_coeffs.append(0)
        if i % 2 == 0:
            f_coeffs.append(coeffs[int(i/2)])
        else:
            f_coeffs.append(0)
    f = np.poly1d(f_coeffs)  # Assuming f is represented as a polynomial
    
    # Define the convolution integrand
    def convolution_integrand(y, x):
        return (np.sinc(2 * (x - y))**2+np.sinc(2 * (x + y))**2) * f(y)
    
    # Perform the double integral
    def double_integral(x):
        return quad(lambda y: convolution_integrand(y, x), -a, a)[0]
    
    conv_term = 8 * quad(lambda x: double_integral(x) * np.exp(-2 * f(x) * T), -a, a)[0]
    
    # Original integrand term
    integrand_term = quad(lambda x: np.interp(x, np.linspace(-a, a, n), np.abs(g)**2 + 0.00000001) * np.exp(-2 * f(x) * T), -a, a)[0]
    
    # Sum both terms
    return sigma_a**2*conv_term + integrand_term


# Constraint: integral(f) = 1
def constraint_integral(coeffs):
    f_coeffs = []
    for i in range(2*len(coeffs)-1):
        #if i == 0:
        #    f_coeffs.append(1)
        #    f_coeffs.append(0)
        if i % 2 == 0:
            f_coeffs.append(coeffs[int(i/2)])
        else:
            f_coeffs.append(0)
    f = np.poly1d(f_coeffs)
    integral_value, _ = quad(f, -a, a)
    return np.maximum(0,np.abs(integral_value - 1) - 0.00001)

def constraint_positivity(coeffs):
    f_coeffs = []
    for i in range(2*len(coeffs)-1):
        #if i == 0:
        #    f_coeffs.append(1)
        #    f_coeffs.append(0)
        if i % 2 == 0:
            f_coeffs.append(coeffs[int(i/2)])
        else:
            f_coeffs.append(0)
    f = np.poly1d(f_coeffs)
    # Sample points in the domain [0, a] to ensure f(x) is positive
    x_sample = np.linspace(-a, a, n)
    return f(x_sample)- low

# Initial guess for the coefficients of f (e.g., as a polynomial)
initial_guess = np.random.rand(5)  # For a quadratic polynomial, for example

# Set up the constraint as a dictionary
constraints = [
    {'type': 'eq', 'fun': constraint_integral},
    {'type': 'ineq', 'fun': constraint_positivity}
]

# Perform the optimization
result = minimize(objective, initial_guess, constraints=constraints, options={'ftol': 1e-9, 'disp': True})

# The optimal coefficients for f
f_coeffs = []
for i in range(2*len(result.x)-1):
    #if i == 0:
    #    f_coeffs.append(1)
    #    f_coeffs.append(0)
    if i % 2 == 0:
        f_coeffs.append(result.x[int(i/2)])
    else:
        f_coeffs.append(0)
f_optimal = np.poly1d(f_coeffs)
comma_separated = ', '.join(map(str, f_coeffs))
print(comma_separated)
print (f_optimal)

z = np.linspace(-a, a, n)
plt.plot(z, f_optimal(z))
plt.show()

print (quad(f_optimal, -a, a))
"""

def objective(f_coeffs):
    f = np.poly1d(f_coeffs)  # Assuming f is represented as a polynomial
    
    # Define the convolution integrand
    def convolution_integrand(y, x):
        return (np.sinc(2 * (x - y))**2+np.sinc(2 * (x + y))**2) * f(y)
    
    # Perform the double integral
    def double_integral(x):
        return quad(lambda y: convolution_integrand(y, x), 0, a)[0]
    
    conv_term = 8 * quad(lambda x: double_integral(x) * np.exp(-2 * f(x) * T), 0, a)[0]
    
    # Original integrand term
    integrand_term = quad(lambda x: np.interp(x, np.linspace(0, a, n), np.abs(g)**2 + 0.00000001) * np.exp(-2 * f(x) * T), 0, a)[0]
    
    # Sum both terms
    return sigma_a**2*conv_term + integrand_term


# Constraint: integral(f) = 1
def constraint_integral(f_coeffs):
    f = np.poly1d(f_coeffs)
    integral_value, _ = quad(f, 0, a)
    return np.maximum(0,np.abs(integral_value - .5) - 0.00001)

def constraint_positivity(f_coeffs):
    f = np.poly1d(f_coeffs)
    # Sample points in the domain [0, a] to ensure f(x) is positive
    x_sample = np.linspace(0, a, n)
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
f_coeffs = result.x
f_optimal = np.poly1d(f_coeffs)
comma_separated = ', '.join(map(str, f_coeffs))
print(comma_separated)
print (f_optimal)

z = np.linspace(0, a, n)
plt.plot(z, f_optimal(z))
plt.show()

print (quad(f_optimal, 0, a))