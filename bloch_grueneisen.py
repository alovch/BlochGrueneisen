import numpy as np
import mpmath as mp
from scipy.optimize import differential_evolution
from scipy.integrate import quad


def integrand(x,  n):
    return mp.power(x, n)/((mp.exp(x) - 1)*(1 - mp.exp(-x)))


def bg_func(p, temp):
    temp_deb, a_const, rho0, n = p
    coeff = a_const * (np.power(temp, n)/np.power(temp_deb, n))
    integrated = quad(integrand, 0, temp_deb/temp, args=(n,), limit=100)[0]
    return rho0 + coeff * integrated


def residual(p, temp, rho):
    temp_deb, a_const, rho0, n = p
    coeff = a_const * (np.power(temp, n)/np.power(temp_deb, n))
    integrated = np.asarray([quad(integrand, 0, temp_deb/_temp, args=(n,), limit=100)[0] for _temp in temp])
    error = rho - rho0 - coeff * integrated
    return np.sum(error**2)

# the resistivity data file must contain temperature and resistivity as the 1st and 2nd columns, respectively


data = np.loadtxt(r'C:\Users\User1\Documents\resist.dat')  # path needs to be adjusted  

temp_exp = data[:, 0]
rho_exp = data[:, 1]

# local minimization often fails, that's why differential evolution is used (may be time-consuming)

bounds = [(100.0, 300.0), (1.0E-06, 1.0E-05), (1.0E-6, 1.0E-5), (2.00, 5.00)]  # bounds need to be adjusted
p_opt = differential_evolution(residual, bounds, args=(temp_exp, rho_exp), maxiter=30000, popsize=500)

print(p_opt)

# the fit is written to a new file for use in plotting software

name = r'C:\Users\User1\Documents\resist_fit.dat'

# temperature range for writing the fit function may be adjusted

temp_fit = np.linspace(2, 300, 300)

with open(name, 'w') as f:
    for _temp_fit in temp_fit:
        i = str(_temp_fit) + ' ' + str(bg_func(p_opt.x, _temp_fit)) + '\n'
        f.write(i)
