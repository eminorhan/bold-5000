import os
import json
import numpy as np
from scipy.io import loadmat
from PIL import Image


def extract_subj_data(subj_name, test_list):
    data = loadmat('../' + subj_name + '/mat/' + subj_name + '_ROIs_TR34.mat')
    keys = ['LHPPA', 'RHLOC', 'LHLOC', 'RHEarlyVis', 'RHRSC', 'RHOPA', 'RHPPA', 'LHEarlyVis', 'LHRSC', 'LHOPA']

    subj_stim_list = np.loadtxt('../stim_lists/' + subj_name + '_stim_lists.txt', dtype='str')

    test_y = {}
    losses = {}

    for key in keys:

        test_y[key] = {}
        losses[key] = []

        for i in range(len(test_list)):
            test_y[key][test_list[i]] = []

    # u, indices = np.unique(subj_stim_list, return_inverse=True)

    for i in range(len(subj_stim_list)):
        if subj_stim_list[i] in test_list:
            for key in keys:
                test_y[key][subj_stim_list[i]].append(np.expand_dims(data[key][i], axis=0))

    for key in keys:
        for i in range(len(test_y[key])):
            test_y[key][test_list[i]] = np.concatenate(test_y[key][test_list[i]], axis=0)
            losses[key].append(repeat_baseline(test_y[key][test_list[i]]))

    for key in keys:
        losses[key] = np.concatenate(losses[key])
        print(key, len(losses[key]))

    np.save('../' + subj_name + '_repet_losses.npy', losses)

    return


def repeat_baseline(data):
    '''Compute the baseline from repetitions'''
    n_reps = data.shape[0]
    indices = np.arange(n_reps)
    loss = np.zeros(n_reps)

    for i in range(n_reps):
        ind = data[i]
        dep = np.mean(data[np.setdiff1d(indices, i)], axis=0)
        loss[i] = np.mean(abs(ind - dep))

    return loss


if __name__ == "__main__":

    repeat_imgs_list = list(np.loadtxt('../repeated_stimuli_113_list.txt', dtype='str'))

    subj_list = ['CSI1', 'CSI2', 'CSI3']

    for subj in subj_list:
        extract_subj_data(subj, repeat_imgs_list)