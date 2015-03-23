# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:34:00 2015

@author: yw5aj
"""

from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def triple_exp(params, x):
    y = params[0] * np.exp(x * params[1])\
        + params[2] * np.exp(x * params[3])\
        + params[4] * np.exp(x * params[5])
    return y


def rise_exp(params, x):
    y = params[0] * np.exp(x * params[1]) + params[2]
    return y


def get_r2(params, func, x, y, sign=1.):
    ynew = func(params, x)
    sstot = ((y - y.mean()) ** 2).sum()
    ssres = ((y - ynew) ** 2).sum()
    r2 = 1 - ssres / sstot
    return r2 * sign


if __name__ == '__main__':
    # Read the data
    stimuli_list = [10, 50, 200]
    time_list, force_list, displ_list = [], [], []
    for i, stimulus in enumerate(stimuli_list):
        time = pd.read_csv('./data/%dmN_time.txt' % stimulus).values.T[0]
        displ = pd.read_csv('./data/%dmN_disp.txt' % stimulus).values.T[0]
        force = pd.read_csv('./data/%dmN_force.txt' % stimulus).values.T[0]
        maxidx = force.argmax()
        time_list.append(time[maxidx:maxidx + 5000] - time[maxidx])
        displ_list.append(displ[maxidx:maxidx + 5000])
        force_list.append(force[maxidx:maxidx + 5000])
    # %% Start fitting the triple exponential decay
    triple_exp_params = []
    triple_exp_r2 = []
    fig, axs = plt.subplots(3, 1, figsize=(3.27, 7.5))
    for i in range(len(stimuli_list)):
        time = time_list[i]
        force = force_list[i]
        # Fit the force trace
        x0 = np.array([1, -1, 1, -1, 1, -1]) * force.max()
        bounds = ((0, None), (None, 0),
                  (0, None), (None, 0),
                  (0, None), (None, 0))
        res = minimize(
            get_r2, x0, args=(triple_exp, time, force, -1),
            method='SLSQP', bounds=bounds)
        triple_exp_params.append(res.x)
        triple_exp_r2.append(-res.fun)
        axs[i].plot(time, force, '.', color='.5')
        axs[i].plot(time, triple_exp(res.x, time), '-k')
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Force (N)')
    fig.tight_layout()
    fig.savefig('./plots/force_decay.png')
    plt.close(fig)
    print('The parameters for the three fits are: %s' % str(triple_exp_params))
    print('Goodness of fit for the three fits are: %s' % str(triple_exp_r2))
    # %% Start fitting the hyperelastic curve
    hyper_force, hyper_displ = [], []
    fig, axs = plt.subplots()
    for i in range(len(stimuli_list)):
        displ = displ_list[i]
        force = force_list[i]
        hyper_force.append(force[0])
        hyper_displ.append(displ[-1])
    hyper_displ = np.array(hyper_displ)
    hyper_force = np.array(hyper_force)
    axs.plot(hyper_displ, hyper_force, 'xk', ms=6, label='Experiment')
    x0 = np.array([hyper_force.mean(), 1., -hyper_force.mean()])
    bounds = ((0, None), (0, None), (None, None))
    res = minimize(
        get_r2, x0, args=(rise_exp, hyper_displ, hyper_force, -1),
        method='SLSQP', bounds=bounds)
    axs.plot(hyper_displ, rise_exp(res.x, hyper_displ), '-k', label='Fit')
    axs.legend(loc=2)
    axs.set_xlabel('Displacement (mm)')
    axs.set_ylabel('Force (N)')
    fig.tight_layout()
    fig.savefig('./plots/displ_force.png')
    hyper_param = res.x
    hyper_r2 = -res.fun
    print('The parameters for the hyper fit are: %s' % str(hyper_param))
    print('Goodness of fit for the hyper fit is: %.3f' % hyper_r2)
