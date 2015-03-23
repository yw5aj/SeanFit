# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:34:00 2015

@author: yw5aj
"""

from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DT = 1E-3


def displ2force(params, displ):
    force = params[0] * (np.exp(displ * params[1]) - 1)
    return force


def relax(params, time):
    g1, tau1, g2, tau2 = params
    ginf = 1 - g1 - g2
    gr = g1 * np.exp(-time / tau1) +\
        g2 * np.exp(-time / tau2) +\
        ginf
    return gr


def creep(params, time, ginf):
    c1, tau1, c2, tau2 = params
    cinf = 1 - c1 - c2
    cc = 1 + (1 -
              c1 * np.exp(-time / tau1) -
              c2 * np.exp(-time / tau2) -
              cinf
              ) * (1 / ginf - 1)
    return cc


def predict_force(params_d2f, params_rel, time, displ):
    force_elastic = displ2force(params_d2f, displ)
    gr = relax(params_rel, time)
    force = np.convolve(gr, np.gradient(force_elastic) / DT)[:time.size] * DT
    return force


def get_r2(params, time, displ, force):
    params_d2f = params[:2]
    params_rel = params[2:]
    force_predicted = predict_force(params_d2f, params_rel, time, displ)
    maxidx = force.argmax()
    force_ramp = force[:maxidx]
    force_hold = force[maxidx:]
    force_predicted_ramp = force_predicted[:maxidx]
    force_predicted_hold = force_predicted[maxidx:]
    sst_ramp = ((force_ramp - force_ramp.mean()) ** 2).sum()
    sse_ramp = ((force_ramp - force_predicted_ramp) ** 2).sum()
    sst_hold = ((force_hold - force_hold.mean()) ** 2).sum()
    sse_hold = ((force_hold - force_predicted_hold) ** 2).sum()
    r2_ramp = 1 - sse_ramp / sst_ramp
    r2_hold = 1 - sse_hold / sst_hold
    r2 = np.mean((r2_ramp, r2_hold))
    return r2


if __name__ == '__main__':
    # Load all data
    stimuli_list = [10, 50, 200]
    time_list, force_list, displ_list = [], [], []
    for i, stimulus in enumerate(stimuli_list):
        time = pd.read_csv('./data/%dmN_time.txt' % stimulus).values.T[0]
        displ = pd.read_csv('./data/%dmN_disp.txt' % stimulus).values.T[0]
        force = pd.read_csv('./data/%dmN_force.txt' % stimulus).values.T[0]
        contact_idx = (displ > 0).nonzero()[0][0]
        time_list.append(time[contact_idx:contact_idx + 5000]
                         - time[contact_idx])
        displ_list.append(displ[contact_idx:contact_idx + 5000])
        force_list.append(force[contact_idx:contact_idx + 5000])
    # %% Start fitting the force trances

    def get_avg_r2(params, sign=1.):
        r2_list = []
        for i, displ in enumerate(displ_list):
            force = force_list[i]
            time = time_list[i]
            r2_list.append(get_r2(params, time, displ, force))
        avg_r2 = np.mean(r2_list)
        return sign * avg_r2
    bounds = ((0, None), (0, None),
              (0, 1), (0, None),
              (0, 1), (0, None))
    constraints = ({'type': 'ineq', 'fun': lambda x: 1 - x[2] - x[4]})
    x0 = (1e-2, 5, .3, .1, .3, 1)
    res = minimize(get_avg_r2, x0, args=(-1), method='SLSQP',
                   bounds=bounds, constraints=constraints)
    params = res.x
    avg_r2 = -res.fun
    # %% Plot the fitting
    fig, axs = plt.subplots(3, 2, figsize=(6.83, 7.5))
    for i, displ in enumerate(displ_list):
        force = force_list[i]
        time = time_list[i]
        force_predicted = predict_force(params[:2], params[2:], time, displ)
        axs[i, 0].plot(time, force, '.', color='.5')
        axs[i, 0].plot(time, force_predicted, '-k')

