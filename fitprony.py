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


def force2displ(params, force):
    displ = np.log(force / params[0] + 1) / params[1]
    return displ


def relax(params, time):
    g1, tau1, g2, tau2, g3, tau3 = params
    ginf = 1 - g1 - g2 - g3
    gr = g1 * np.exp(-time / tau1) +\
        g2 * np.exp(-time / tau2) +\
        g3 * np.exp(-time / tau3) +\
        ginf
    return gr


def creep(params, time, cinf):
    c1, tau1, c2, tau2, c3, tau3 = params
    cc = 1 + (1 -
              c1 * np.exp(-time / tau1) -
              c2 * np.exp(-time / tau2) -
              c3 * np.exp(-time / tau3)
              ) * (cinf - 1)
    return cc


def predict_force(params_d2f, params_rel, time, displ):
    force_elastic = displ2force(params_d2f, displ)
    gr = relax(params_rel, time)
    force = np.convolve(gr, np.gradient(force_elastic) / DT)[:time.size] * DT
    return force


def predict_displ(params_d2f, params_crp, time, force, cinf):
    displ_elastic = force2displ(params_d2f, force)
    cc = creep(params_crp, time, cinf)
    displ = np.convolve(cc, np.gradient(displ_elastic) / DT)[:time.size] * DT
    return displ


def get_crp_sse(params_crp, time, params_rel, cinf):
    cc = creep(params_crp, time, cinf)
    gr = relax(params_rel, time)
    sse = ((np.convolve(cc, gr)[:time.size] * DT - time) ** 2).sum()
    return sse


def get_force_r2(params, time, displ, force):
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

    def get_avg_force_r2(params, sign=1.):
        r2_list = []
        for i, displ in enumerate(displ_list):
            force = force_list[i]
            time = time_list[i]
            r2_list.append(get_force_r2(params, time, displ, force))
        avg_r2 = np.mean(r2_list)
        return sign * avg_r2
    bounds = ((0, None), (0, None),
              (0, 1), (1e-1, None),
              (0, 1), (1e-1, None),
              (0, 1), (1e-1, None))
    constraints = ({'type': 'ineq', 'fun': lambda x: 1 - np.sum(x[2::2])})
    x0 = (1e-2, 5, .3, .1, .3, 1, .3, 10)
    res = minimize(get_avg_force_r2, x0, args=(-1), method='SLSQP',
                   bounds=bounds, constraints=constraints)
    params = res.x
    avg_r2 = -res.fun
    params_rel = params[2:]
    params_d2f = params[:2]
    ginf = 1 - np.sum(params_rel[::2])
    # %% Predict displacement it takes to generate the desired force
    ramp_time = 1
    maxidx = int(ramp_time / DT)
    for i, stimulus in enumerate(stimuli_list):
        # Calculate cinf at this stimuli
        d0 = force2displ(params_d2f, stimulus * 1e-3)
        dinf = force2displ(params_d2f, stimulus * 1e-3 / ginf)
        cinf = dinf / d0
        # Get creep curve under this stimuli
        bounds = ((0, 1), (1e-1, None),
                  (0, 1), (1e-1, None),
                  (0, 1), (1e-1, None))
        constraints = ({'type': 'eq', 'fun': lambda x: 1 - np.sum(x[::2])})
        x0 = (.3, .1, .3, .1, .3, 1.)
        res = minimize(get_crp_sse, x0, args=(time, params_rel, cinf),
                       method='SLSQP', bounds=bounds, constraints=constraints)
        params_crp = res.x
        # Calculate force
        time = np.arange(5000) * DT
        force_cmd = np.empty_like(time)
        force_cmd[:maxidx] = np.linspace(0, stimulus * 1e-3, maxidx)
        force_cmd[maxidx:] = stimulus * 1e-3
        displ = predict_displ(params_d2f, params_crp, time, force_cmd, cinf)
        force_check = predict_force(params_d2f, params_rel, time, displ)
    # %% Plot the fitting
    fig, axs = plt.subplots(3, 3, figsize=(7, 7.5))
    for i, displ in enumerate(displ_list):
        force = force_list[i]
        time = time_list[i]
        force_predicted = predict_force(params[:2], params[2:], time, displ)
        axs[i, 0].plot(time, force, '.', color='.5')
        axs[i, 0].plot(time, force_predicted, '-k')

