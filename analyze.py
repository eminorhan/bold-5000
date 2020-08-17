"""Analyze results
"""
import os
import sys
import time
import argparse
import numpy as np
from collections import OrderedDict
import matplotlib.pylab as plt
import matplotlib as mp


def plot_corrs(corrs, model_name, save_filename):
    '''To visualize prediction accuracy'''
    plt.clf()

    regions = ['LHEarlyVis', 'LHRSC', 'LHPPA', 'LHOPA', 'LHLOC', 'RHEarlyVis', 'RHRSC', 'RHPPA', 'RHOPA', 'RHLOC']

    if model_name.startswith('alexnet'):
        layers = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    else:
        layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    colors = ['r', 'b', 'k']
    shade_colors = ['pink', 'lightskyblue']

    ax = [None] * 10

    c_ind = 0
    for c in corrs:
        r_ind = 0
        for r in regions:
            ax[r_ind] = plt.subplot(2, 5, r_ind+1)
            x = np.zeros(len(layers))
            s = np.zeros(len(layers))
            for l in layers:
                x[l] = np.mean(c[r, l])
                s[l] = np.std(c[r, l]) / np.sqrt(len(c[r, l]))
            plt.plot(layers, x, '-', color=colors[c_ind])
            plt.plot(layers, x, '.', color=colors[c_ind])
            plt.fill_between(layers, x - s, x + s, color=shade_colors[c_ind])
            plt.ylim([0, .25])
            plt.title(r, fontsize=10)

            if r_ind == 5:
                if model_name.startswith('alexnet'):
                    plt.yticks([0, .05, .1, .15, .2, .25], ['0', '0.05', '0.1', '0.15', '0.2', '0.25'])
                    plt.xticks(layers, ['0', '1', '2', '3', '4', '5', '6', '7', '8'])
                    plt.xlabel('Layers')
                    plt.ylabel('Correlation')
                    plt.text(0, 0.23, 'AlexNet-ImageNet', fontsize=8, color='r')
                    plt.text(0, 0.21, 'AlexNet-Random', fontsize=8, color='b')
                else:
                    plt.yticks([0, .05, .1, .15, .2, .25], ['0', '0.05', '0.1', '0.15', '0.2', '0.25'])
                    plt.xticks(layers, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
                    plt.xlabel('Layers')
                    plt.ylabel('Correlation')
            else:
                if model_name.startswith('alexnet'):
                    plt.yticks([0, .05, .1, .15, .2, .25], ['', '', '', '', '', ''])
                    plt.xticks(layers, ['', '', '', '', '', '', '', '', ''])
                else:
                    plt.yticks([0, .05, .1, .15, .2, .25], ['', '', '', '', '', ''])
                    plt.xticks(layers, ['', '', '', '', '', '', '', '', '', ''])

            ax[r_ind].spines["right"].set_visible(False)
            ax[r_ind].spines["top"].set_visible(False)
            ax[r_ind].yaxis.set_ticks_position('left')
            ax[r_ind].xaxis.set_ticks_position('bottom')

            r_ind += 1
        c_ind += 1

    mp.rcParams['axes.linewidth'] = 0.75
    mp.rcParams['patch.linewidth'] = 0.75
    mp.rcParams['patch.linewidth'] = 1.15
    mp.rcParams['font.sans-serif'] = ['FreeSans']
    mp.rcParams['mathtext.fontset'] = 'cm'

    plt.savefig(save_filename, bbox_inches='tight')


def extract_model_results(directory, model_name):
    '''Results for one model'''
    regions = ['LHEarlyVis', 'LHRSC', 'LHPPA', 'LHOPA', 'LHLOC', 'RHEarlyVis', 'RHRSC', 'RHPPA', 'RHOPA', 'RHLOC']

    if model_name.startswith('alexnet'):
        layers = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    else:
        layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    regs = [0.0, 5e-5, 5e-4]
    RR, LL, PP = np.meshgrid(regions, layers, regs)
    RR = RR.flatten()
    LL = LL.flatten()
    PP = PP.flatten()

    test_corrs = {}
    val_corrs = {}

    for job_idx in range(len(LL)):
        region = RR[job_idx]
        layer = LL[job_idx]
        reg = PP[job_idx]

        filename = 'CSI1' + '_' + region + '_' + str(layer) + '_' + str(reg) + '_' + model_name + '.tar.npz'

        test_output = np.load(os.path.join(directory, filename))['best_test_output']
        test_target = np.load(os.path.join(directory, filename))['best_test_target']

        val_output = np.load(os.path.join(directory, filename))['best_val_output']
        val_target = np.load(os.path.join(directory, filename))['best_val_target']

        d = test_output.shape[1]

        test_rho = np.corrcoef(test_output + 0.0001 * np.random.randn(test_output.shape[0], test_output.shape[1]),
                               test_target, rowvar=False)[d:, :d]  # add small noise to prevent degeneracy
        test_rho_diag = np.diag(test_rho)
        test_corrs[region, layer, reg] = test_rho_diag

        val_rho = np.corrcoef(val_output + 0.0001 * np.random.randn(val_output.shape[0], val_output.shape[1]),
                              val_target, rowvar=False)[d:, :d]
        val_rho_diag = np.diag(val_rho)
        val_corrs[region, layer, reg] = val_rho_diag

    best_test_corrs = {}

    for r in regions:
        for l in layers:

            print(r, l)
            mean_val_corr_0 = np.mean(val_corrs[r, l, 0.0])
            mean_val_corr_1 = np.mean(val_corrs[r, l, 5e-5])
            mean_val_corr_2 = np.mean(val_corrs[r, l, 5e-4])

            if np.max([mean_val_corr_0, mean_val_corr_1, mean_val_corr_2]) == mean_val_corr_0:
                best_test_corrs[r, l] = test_corrs[r, l, 0.0]
            elif np.max([mean_val_corr_0, mean_val_corr_1, mean_val_corr_2]) == mean_val_corr_1:
                best_test_corrs[r, l] = test_corrs[r, l, 5e-5]
            elif np.max([mean_val_corr_0, mean_val_corr_1, mean_val_corr_2]) == mean_val_corr_2:
                best_test_corrs[r, l] = test_corrs[r, l, 5e-4]

    return best_test_corrs


if __name__ == "__main__":

    freeze_trunk_aleximgnet = extract_model_results('../fit_results/freeze_trunk/CSI1/CSI1_aleximgnet/', 'alexnet_imgnet')
    freeze_trunk_alexrand = extract_model_results('../fit_results/freeze_trunk/CSI1/CSI1_alexrand/', 'alexnet_rand')

    train_trunk_aleximgnet = extract_model_results('../fit_results/train_trunk/CSI1/CSI1_aleximgnet/', 'alexnet_imgnet')
    train_trunk_alexrand = extract_model_results('../fit_results/train_trunk/CSI1/CSI1_alexrand/', 'alexnet_rand')

    frozen_save_filename = 'frozen_alexnet.pdf'
    train_save_filename = 'train_alexnet.pdf'

    plot_corrs([freeze_trunk_aleximgnet, freeze_trunk_alexrand], 'alexnet', save_filename=frozen_save_filename)
    plot_corrs([train_trunk_aleximgnet, train_trunk_alexrand], 'alexnet', save_filename=train_save_filename)

