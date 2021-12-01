'''
Created on Oct 27, 2021

@author: simon
'''
from numpy.random import default_rng
import numpy as np
from scipy.optimize import minimize
from collections import namedtuple
import os

from greg import (
    correlation, force_doubly_nonnegative, decay_model, EMI, covariance_matrix, 
    valid_G, hadreg, enforce_directory, load_object, save_object, 
    circular_accuracy)

SimCG0 = namedtuple('SimCG0', ['C_obs', 'G0'])

def logit(p):
    return np.log(p / (1 - p))

def logistic(x):
    return (1 + np.exp(-x)) ** (-1)

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


def prepare_data(paramlist, rng=None):
    data = []
    for params in paramlist:
        C_obs = correlation(covariance_matrix(decay_model(rng=rng, **params)))
        G0 = valid_G(C_obs, corr=True)
        data.append(SimCG0(C_obs=C_obs, G0=G0))
    return data

def default_paramlist(L=100, R=5000, P=40, coh_decay_list=None, coh_infty_list=None):
    params0 = {
        'R': R, 'L': L, 'P': P, 'incoh_bad': None}
    if coh_decay_list is None: coh_decay_list = [0.5, 0.9]
    if coh_infty_list is None: coh_infty_list = [0.0, 0.1, 0.3, 0.5, 0.7]

    paramlist = []
    for coh_decay in coh_decay_list:
        for coh_infty in coh_infty_list:
            params = params0.copy()
            params.update({'coh_decay': coh_decay, 'coh_infty': coh_infty})
            paramlist.append(params)
    return paramlist

def optimize_hadreg(data, hadreglparam0=None, maxiter=20, gtol=1e-8):
    if hadreglparam0 is None:
        hadreglparam0 = np.zeros(2)
    f_noreg = accuracy_scenario(None, data)
    def fun(hadreglparam):
        return accuracy_scenario(hadreglparam, data)
    options = {'maxiter': maxiter, 'gtol': gtol}
    res = minimize(fun, hadreglparam0, method='BFGS', options=options)
    hadregres = {'hadreglparam': res.x, 'f': res.fun, 'f_noreg': f_noreg}
    return hadregres

def calibrate_hadreg(
        pathout, looks, seed=1, R=10000, P=40, coh_decay_list=None, coh_infty_list=None,
        overwrite=False, njobs=-2, maxiter=20):
    res = {}
    def _calibrate_hadreg(L):
        fnout = os.path.join(pathout, f'{L}.p')
        if overwrite or not os.path.exists(fnout):
            rng = default_rng(seed)
            paramlist = default_paramlist(
                L=L, R=R, P=P, coh_decay_list=coh_decay_list, coh_infty_list=coh_infty_list)
            data = prepare_data(paramlist, rng=rng)
            hadregres = optimize_hadreg(data, maxiter=maxiter)
            save_object(hadregres, fnout)
        else:
            hadregres = load_object(fnout)
        return hadregres

        res[L] = hadregres

    from joblib import Parallel, delayed
    res = Parallel(n_jobs=njobs)(delayed(_calibrate_hadreg)(L) for L in looks)
    for jL, L in enumerate(looks):
        print(L, logistic(res[jL]['hadreglparam']), res[jL]['f'], res[jL]['f_noreg'])

if __name__ == '__main__':
    path0 = '/home2/Work/greg/hadamard'
    looks = np.arange(3, 26, 1) ** 2
    P = 40
    R = 10000
    scenarios = {'broad': (None, None), 'low': ([0.5], [0.0]), 'high': ([0.9], [0.5])}
    for scenario in scenarios:
        pathout = os.path.join(path0, scenario)
        coh_decay_list, coh_infty_list = scenarios[scenario]
        calibrate_hadreg(
            pathout, looks, P=P, R=R, coh_decay_list=coh_decay_list, 
            coh_infty_list=coh_infty_list)

