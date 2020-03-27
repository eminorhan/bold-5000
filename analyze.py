"""Analyze results
"""
import os
import sys
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.utils.data
import torchvision
import torchvision.models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pylab as plt
import matplotlib as mp


def plot_corrs(corrs):
    '''To visualize prediction accuracy'''
    plt.clf()

    regions = ['LHPPA', 'RHEarlyVis', 'LHRSC', 'RHRSC', 'LHLOC', 'RHOPA', 'LHEarlyVis', 'LHOPA', 'RHPPA', 'RHLOC']
    layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colors = ['r', 'b', 'k']

    c_ind = 0
    for c in corrs:
        r_ind = 0
        for r in regions:
            plt.subplot(2, 5, r_ind+1)
            x = np.zeros(10)
            for l in layers:
                x[l] = np.nanmean(c[r, l])
            plt.plot(layers, x, color=colors[c_ind])
            plt.ylim([0, .2])
            r_ind += 1
        c_ind += 1

    plt.savefig('correlations.pdf', bbox_inches='tight')


def extract_model_results(directory, model_name):
    '''Results for one model'''
    regions = ['LHPPA', 'RHEarlyVis', 'LHRSC', 'RHRSC', 'LHLOC', 'RHOPA', 'LHEarlyVis', 'LHOPA', 'RHPPA', 'RHLOC']
    layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    RR, LL = np.meshgrid(regions, layers)
    RR = RR.flatten()
    LL = LL.flatten()

    corrs = {}
    losses = {}

    for job_idx in range(100):
        region = RR[job_idx]
        layer = LL[job_idx]
        filename = 'CSI1' + '_' + region + '_' + str(layer) + '_' + model_name + '.tar.npz'
        output = np.load(os.path.join(directory, filename))['best_output']
        target = np.load(os.path.join(directory, filename))['best_target']
        test_losses = np.load(os.path.join(directory, filename))['test_losses']

        d = output.shape[1]
        rho = np.corrcoef(output, target, rowvar=False)[d:, :d]
        rho_diag = np.diag(rho)

        corrs[region, layer] = rho_diag
        losses[region, layer] = np.nanmin(test_losses)

    return corrs, losses


if __name__ == "__main__":

    freeze_trunk_rand, _ = extract_model_results('../fit_results/freeze_trunk/CSI1_rand/', 'resnext101_32x8d_rand')
    freeze_trunk_imgnet, _ = extract_model_results('../fit_results/freeze_trunk/CSI1_imgnet/', 'resnext101_32x8d_imgnet')
    freeze_trunk_wsl, _ = extract_model_results('../fit_results/freeze_trunk/CSI1_wsl/', 'resnext101_32x8d_wsl')

    train_trunk_rand, _ = extract_model_results('../fit_results/train_trunk/CSI1_rand/', 'resnext101_32x8d_rand')
    train_trunk_imgnet, _ = extract_model_results('../fit_results/train_trunk/CSI1_imgnet/', 'resnext101_32x8d_imgnet')
    train_trunk_wsl, _ = extract_model_results('../fit_results/train_trunk/CSI1_wsl/', 'resnext101_32x8d_wsl')

    plot_corrs([freeze_trunk_rand, freeze_trunk_imgnet, freeze_trunk_wsl])
