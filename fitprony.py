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
ND2F = 2
FT = .002  # contact threshold


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


def predict_force(params_d2f, params_rel, time, displ):
    force_elastic = displ2force(params_d2f, displ)
    gr = relax(params_rel, time)
    force = np.convolve(gr, np.gradient(force_elastic) / DT)[:time.size] * DT
    return force


def get_force_sse(fitx, time, displ, force, x0):
    params = fitx * x0
    params_d2f = params[:ND2F]
    params_rel = params[ND2F:]
    force_predicted = predict_force(params_d2f, params_rel, time, displ)
    residual = force - force_predicted
    sse = ((residual / force.mean()) ** 2).sum()
    return sse


if __name__ == '__main__':
    # Load all data
    stimuli_list = [10, 50, 200]
    time_list, force_list, displ_list = [], [], []
    for i, stimulus in enumerate(stimuli_list):
        time = pd.read_csv('./data/%dmN_time.txt' % stimulus).values.T[0]
        displ = pd.read_csv('./data/%dmN_disp.txt' % stimulus).values.T[0]
        force = pd.read_csv('./data/%dmN_force.txt' % stimulus).values.T[0]
        contact_idx = (force > FT).nonzero()[0][0] - 100
        time_list.append(time[contact_idx:contact_idx + 5000]
                         - time[contact_idx])
        force_list.append(force[contact_idx:contact_idx + 5000])
        # Make displacement curve
        displ_static = displ[contact_idx + 2500]
        maxidx = force_list[-1].argmax()
        displ_new = np.empty_like(force_list[-1])
        displ_new[:maxidx] = np.linspace(0, displ_static, maxidx)
        displ_new[maxidx:] = displ_static
        displ_list.append(displ_new)
    # %% Start fitting the force trances
    params_list, params_d2f_list, params_rel_list = [], [], []
    for i, stimulus in enumerate(stimuli_list):
        force = force_list[i]
        displ = displ_list[i]
        time = time_list[i]
        bounds = ((0, None), (0, None),
                  (0, 1), (DT, None),
                  (0, 1), (DT, None),
                  (0, 1), (DT, None))
        constraints = ({'type': 'ineq',
                        'fun': lambda x: 1 - np.sum(x[ND2F::2])})
        x0 = np.array([.01, 7, .3, .001, .3, .5, .3, 10])
        res = minimize(get_force_sse, np.ones(8),
                       args=(time, displ, force, x0),
                       method='SLSQP', bounds=bounds, constraints=constraints)
        params = res.x * x0
        params_list.append(params)
        params_rel_list.append(params[ND2F:])
        params_d2f_list.append(params[:ND2F])
    # %% Find point by point
    resolution = 1e-2
    ramp_time = .5
    maxidx = int(ramp_time / resolution)
    displ_cmd_list, force_cmd_list, force_model_list = [], [], []
    time = np.arange(500) * resolution
    fine_time = np.arange(5000) * DT
    for i, stimulus in enumerate(stimuli_list):
        params_d2f = params_d2f_list[i]
        params_rel = params_rel_list[i]
        force_cmd = np.empty_like(time)
        force_cmd[:maxidx] = np.linspace(0, stimulus * 1e-3, maxidx)
        force_cmd[maxidx:] = stimulus * 1e-3
        displ = np.empty_like(time)
        for idx in range(time.size - 1):

            def get_current_sse(d):
                displ[idx + 1] = d
                current_displ = displ[:idx + 2]
                current_force = force_cmd[:idx + 2]
                current_time = time[:idx + 2]
                predicted_force = predict_force(params_d2f, params_rel,
                                                current_time, current_displ)
                sse = ((current_force - predicted_force) ** 2).sum()
                return sse
            d0 = force2displ(params_d2f, stimulus * 1e-3)
            d = minimize(get_current_sse, d0, method='L-BFGS-B',
                         bounds=((0, None), )).x
            displ[idx + 1] = d
        displ_cmd_list.append(np.interp(fine_time, time, displ))
        force_cmd_list.append(np.interp(fine_time, time, force_cmd))
        force_model_list.append(predict_force(params_d2f, params_rel,
                                              fine_time, displ_cmd_list[-1]))
    # %% Plot the fitting
    fig, axs = plt.subplots(3, 3, figsize=(7, 7))
    for i, displ in enumerate(displ_list):
        # First column, fitting
        force = force_list[i]
        time = time_list[i]
        params_d2f = params_d2f_list[i]
        params_rel = params_rel_list[i]
        force_predicted = predict_force(params_d2f, params_rel, time, displ)
        axs[i, 0].plot(time, force, '.', color='.5', label='Data')
        axs[i, 0].plot(time, force_predicted, '-k', label='Model')
        # Second column, commanded displ
        displ_cmd = displ_cmd_list[i]
        axs[i, 1].plot(time, displ_cmd, '-k', label='Commanded')
        # Thrid column, commanded force vs. modeled
        force_cmd = force_cmd_list[i]
        force_model = force_model_list[i]
        axs[i, 2].plot(time, force_cmd, '-b', label='Ideal/commanded')
        axs[i, 2].plot(time, force_model, '-r', label='Achieved in model')
    # Formatting
    for axes in axs.ravel():
        axes.set_xlim(-.5, 5.5)
    for axes in axs[-1]:
        axes.set_xlabel('Time (s)')
    for axes in axs.T[0]:
        axes.set_ylabel('Force (N)')
    for axes in axs.T[1]:
        axes.set_ylabel('Displacement (mm)')
    for axes in axs.T[2]:
        axes.set_ylabel('Force (N)')
    axs[0, 0].legend(loc=4)
    axs[0, 1].legend(loc=4)
    axs[0, 2].legend(loc=4)
    axs[0, 2].set_ylim(top=.012)
    axs[2, 2].set_ylim(top=.22)
    fig.tight_layout()
    fig.savefig('./plots/model.png')
