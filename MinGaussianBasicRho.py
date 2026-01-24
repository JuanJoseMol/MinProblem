import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from matplotlib import pyplot as plt
from scipy.integrate import dblquad

# Parameters
T = 10  # fixed value for T
a = 60 # parameter a
#x = np.linspace(0, a, 120)
sigma_a = 1*np.sqrt(2/2000)
low = 0

def func(x):
  freq = 2.1
  sin = np.sin(2*np.pi*freq*x)
  return np.sign(sin)*(np.abs(sin) > 0.5)

n = 120
y = np.linspace(-1,1,int(2*n))
g  = np.fft.fft(func(y))[0:n]
"""
plt.plot(x, np.abs(g))
plt.yscale('log')
plt.show()
"""
print (np.min(np.abs(g)),np.max(np.abs(g)))



# Objective function to minimize
def objective(params):
    sigma = params  # Extract a and sigma from params
    term2 = lambda x, y: (np.interp(x, np.linspace(0, a, n), np.abs(g)**2 + 0.00000001)*np.exp(-2 * np.exp(-x**2 / (2 * sigma)) * T))
   
    term1 = lambda x, y: (((sigma_a**2)*2*(np.sinc(2*(x - y))**2+np.sinc(2*(x + y))**2) 
            *np.exp(-y**2 / (2 * sigma)))* np.exp(-2 * np.exp(-x**2 / (2 * sigma)) * T))
    #integral = 
    return dblquad(term1, 0, a, lambda x: 0, lambda x: a)[0]


# Constraint: integral(f) = 1
def constraint_integral(params):
    integral_value, _ = quad(lambda x: np.exp(-x**2 / (2 * params)), 0, a)
    return np.maximum(0,np.abs(integral_value - .5) - 0.00001)

def constraint_positivity(params):
    # Sample points in the domain [0, a] to ensure f(x) is positive
    x_sample = np.linspace(0, a, n)
    return np.exp(-x_sample**2 / (2 * params))- low

# Initial guess for the coefficients of f (e.g., as a polynomial)
initial_guess = np.array([10])  # For a quadratic polynomial, for example

# Set up the constraint as a dictionary
constraints = [
    {'type': 'eq', 'fun': constraint_integral},
    {'type': 'ineq', 'fun': constraint_positivity}
]

# Perform the optimization
result = minimize(objective, initial_guess, constraints=constraints, options={'ftol': 1e-9, 'disp': True})

# The optimal coefficients for f
f_optimal = lambda x: np.exp(-x**2 / (2 * result.x))
comma_separated = ', '.join(map(str, result.x))
print(comma_separated)


z = np.linspace(0, a, 120)
plt.plot(z, f_optimal(z))
plt.show()

print (quad(f_optimal, 0, a))