'''
Created on Oct 27, 2021

@author: simon
'''
from numpy.random import default_rng
import numpy as np
from scipy.optimize import minimize

from collections import namedtuple

from greg import (
    correlation, force_doubly_nonnegative, decay_model, EMI, covariance_matrix, valid_G,
    hadreg)

# function that accumulates error over multiple sets of coh parameters
# function that optimizes w.r.t. parameters

SimCG0 = namedtuple('SimCG0', ['C_obs', 'G0'])

def logit(p):
    return np.log(p / (1 - p))

def logistic(x):
    return (1 + np.exp(-x)) ** (-1)

# function that gets phase linking results
def accuracy_scenario(hadreglparam, data):
    alpha = logistic(hadreglparam[0])
    nu = logistic(hadreglparam[1])
    acc = []
    for simCG0 in data:
        G = hadreg(simCG0.G0, alpha=alpha, nu=nu)
        cphases = EMI(simCG0.C_obs, G=G, corr=False)
        acc.append(circular_accuracy(cphases))
    return np.mean(acc)

def circular_accuracy(cphases):
    # assumes the true phases are zero
    cphases /= np.abs(cphases)
    return 1 - np.mean(cphases[:, 1:].real)

def prepare_data(paramlist, rng=None):
    data = []
    for params in paramlist:
        C_obs = correlation(covariance_matrix(decay_model(rng=rng, **params)))
        G0 = valid_G(C_obs, corr=True)
        data.append(SimCG0(C_obs=C_obs, G0=G0))
    return data

def default_paramlist(L=100, R=5000):
    params0 = {
        'R': R, 'L': L, 'P': 40,'incoh_bad': None}
    coh_decay_list = [0.5, 0.9]
    coh_infty_list = [0.0, 0.1, 0.3, 0.5, 0.7]
    
    paramlist = []
    for coh_decay in coh_decay_list:
        for coh_infty in coh_infty_list:
            params = params0.copy()
            params.update({'coh_decay': coh_decay, 'coh_infty': coh_infty})
            paramlist.append(params)
    return paramlist

def optimize_hadreg(data, hadreglparam0=None, maxiter=5):
    if hadreglparam0 is None:
        hadreglparam0 = np.zeros(2)
    def fun(hadreglparam):
        return accuracy_scenario(hadreglparam, data)
    res = minimize(fun, hadreglparam0, method='BFGS', options={'maxiter': maxiter})
    hadreglparam = res.x
#     return np.array([logistic(hadreglparam[0]), logistic(hadreglparam[1])])
    return hadreglparam
        
if __name__ == '__main__':
#     print(circular_accuracy(results_scenario(500)))
    # write wrapper, save results, parallelize
    looks = [10, 20, 40, 80, 160, 320]
    seed = 1
    rng = default_rng(seed)
    paramlist = default_paramlist(L=100)
    data = prepare_data(paramlist, rng=rng)
    optimize_hadreg(data, maxiter=20)
