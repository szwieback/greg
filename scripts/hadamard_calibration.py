'''
Created on Oct 27, 2021

@author: simon
'''
from numpy.random import default_rng
import numpy as np
from scipy.optimize import minimize
import pickle
import zlib
from collections import namedtuple
import os

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

def enforce_directory(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            pass

def save_object(obj, filename):
    enforce_directory(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        f.write(zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)))

def load_object(filename):
    if os.path.splitext(filename)[1].strip() == '.npy':
        return np.load(filename)
    with open(filename, 'rb') as f:
        obj = pickle.loads(zlib.decompress(f.read()))
    return obj

# add opt out option
# function that gets phase linking results
def accuracy_scenario(hadreglparam, data):
    if hadreglparam is not None:
        alpha, nu = logistic(hadreglparam[0]), logistic(hadreglparam[1])
    acc = []
    for simCG0 in data:
        if hadreglparam is not None:
            G = hadreg(simCG0.G0, alpha=alpha, nu=nu)
        else:
            G = simCG0.G0
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
        'R': R, 'L': L, 'P': 40, 'incoh_bad': None}
    coh_decay_list = [0.5, 0.9]
    coh_infty_list = [0.0, 0.1, 0.3, 0.5, 0.7]

    paramlist = []
    for coh_decay in coh_decay_list:
        for coh_infty in coh_infty_list:
            params = params0.copy()
            params.update({'coh_decay': coh_decay, 'coh_infty': coh_infty})
            paramlist.append(params)
    return paramlist

# return dictionary that includes accuracy for optimal and no reg
def optimize_hadreg(data, hadreglparam0=None, maxiter=20):
    if hadreglparam0 is None:
        hadreglparam0 = np.zeros(2)
    f_noreg = accuracy_scenario(None, data)
    def fun(hadreglparam):
        return accuracy_scenario(hadreglparam, data)
    res = minimize(fun, hadreglparam0, method='BFGS', options={'maxiter': maxiter})
    hadregres = {'hadreglparam': res.x, 'f': res.fun, 'f_noreg': f_noreg}
    return hadregres

def calibrate_hadreg(
        pathout, looks, seed=1, R=10000, overwrite=False, njobs=-2, maxiter=20):
    res = {}
    def _calibrate_hadreg(L):
        fnout = os.path.join(pathout, f'{L}.p')
        if overwrite or not os.path.exists(fnout):
            rng = default_rng(seed)
            paramlist = default_paramlist(L=L, R=R)
            data = prepare_data(paramlist, rng=rng)
            hadregres = optimize_hadreg(data, maxiter=maxiter)
            save_object(hadregres, fnout)
        else:
            hadregres = load_object(fnout)
        return hadregres
    
        res[L] = hadregres
        
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=njobs)(delayed(_calibrate_hadreg)(L) for L in looks)
    for L in res:
        print(L, logistic(res[L][0]), logistic(res[L][1]))

if __name__ == '__main__':
    pathout = '/home2/Work/greg/hadamard'
    looks = np.arange(3, 26, 1) ** 2
    calibrate_hadreg(pathout, looks)
