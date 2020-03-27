import os
import numpy as np
from scipy.io import loadmat
from PIL import Image


def preprocess_img(img_filename):
    img = Image.open('all_imgs/' + img_filename)
    img = img.convert("RGB")
    img = img.resize((256, 256), resample=Image.BILINEAR)

    left = (256 - 224) // 2
    top = (256 - 224) // 2
    right = (256 + 224) // 2
    bottom = (256 + 224) // 2

    img = img.crop((left, top, right, bottom))

    return np.asarray(img)

def extract_subj_data(subj_name, test_list):
    data = loadmat(subj_name + '/mat/' + subj_name + '_ROIs_TR34.mat')
    keys = ['LHPPA', 'RHLOC', 'LHLOC', 'RHEarlyVis', 'RHRSC', 'RHOPA', 'RHPPA', 'LHEarlyVis', 'LHRSC', 'LHOPA']

    subj_stim_list = np.loadtxt('stim_lists/' + subj_name + '_stim_lists.txt', dtype='str')

    train_x = []
    test_x = []
    train_y = {}
    test_y = {}

    for key in keys:
        train_y[key] = []
        test_y[key] = []

    for i in range(len(subj_stim_list)):

        img = np.expand_dims(preprocess_img(subj_stim_list[i]), axis=0)

        if subj_stim_list[i] in test_list:
            test_x.append(img)
            for key in keys:
                test_y[key].append(np.expand_dims(data[key][i], axis=0))
        else:
            train_x.append(img)
            for key in keys:
                train_y[key].append(np.expand_dims(data[key][i], axis=0))

    train_x = np.concatenate(train_x, axis=0)
    test_x = np.concatenate(test_x, axis=0)

    for key in keys:
        train_y[key] = np.concatenate(train_y[key], axis=0)
        test_y[key] = np.concatenate(test_y[key], axis=0)

    np.savez(subj_name + '_data.npz', train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y)

    return train_x, test_x, train_y, test_y


if __name__ == "__main__":

    dir_name = '/home/emin/Projects/bold5000/ROIs/all_imgs'
    all_imgs_list = os.listdir(dir_name)

    repeat_imgs_list = list(np.loadtxt('repeated_stimuli_113_list.txt', dtype='str'))
    nonrepeat_imgs_list = list(set(all_imgs_list) - set(repeat_imgs_list))

    n_test = 256
    n_repeat_test = 113
    nonrepeat_test_list = list(np.random.choice(nonrepeat_imgs_list, size=n_test - n_repeat_test, replace=False))

    test_list = nonrepeat_test_list + repeat_imgs_list

    subj_list = ['CSI1', 'CSI2', 'CSI3', 'CSI4']

    for subj in subj_list:
        a, b, c, d = extract_subj_data(subj, test_list)
        print('Subject: ', subj, 'train_x:', a.shape, 'test_x', b.shape, 'train_y', c['RHLOC'].shape, 'test_y',
              d['RHOPA'].shape)