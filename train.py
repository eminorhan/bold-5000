"""Fit ResNexts
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


parser = argparse.ArgumentParser(description='Train models on fMRI data')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default:4)')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--batch-size', default=164, type=int, help='mini-batch size, this is the total batch size of all '
                                                                'GPUs on the current node when using Data '
                                                                'Parallel or Distributed Data Parallel')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=0.0001, type=float, help='weight decay (default: 0)')
parser.add_argument('--subject', default='CSI1', type=str, help='subject', choices=['CSI1', 'CSI2', 'CSI3'])
parser.add_argument('--region', default='RHOPA', type=str, help='subject',
                    choices=['LHPPA', 'RHEarlyVis', 'LHRSC', 'RHRSC', 'LHLOC', 'RHOPA', 'LHEarlyVis', 'LHOPA', 'RHPPA',
                             'RHLOC'])
parser.add_argument('--model-name', type=str, default='resnext101_32x8d_rand',
                    choices=['resnext101_32x8d_rand', 'resnext101_32x8d_imgnet', 'resnext101_32x8d_wsl'],
                    help='evaluated model')
parser.add_argument('--freeze-trunk', default=False, action='store_true', help='freeze trunk?')
parser.add_argument('--layer', default=8, type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help='which layer')


def set_parameter_requires_grad(model, feature_extracting=True):
    '''Helper function for setting trunk to non-trainable'''
    if feature_extracting:
        for param in list(model.parameters())[:-2]:
            param.requires_grad = False


class Flatten(torch.nn.Module):
    r"""
    Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.
    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, \prod *dims)` (for the default case).
    """
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)


class LastLayerModel(torch.nn.Module):
    """define a new class specifically for last layer
    """
    def __init__(self, backbone):
        super(LastLayerModel, self).__init__()
        self.backbone = backbone
    def forward(self, x):
        x = self.backbone(x)
        # x = torch.squeeze(x)
        x = torch.nn.functional.softmax(x)
        return x


class Squeeze(torch.nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, input):
        output = torch.squeeze(torch.squeeze(input, dim=-1), dim=-1)
        return output


def load_model(args):
    "Loads one of the pretrained models."
    if args.model_name == 'resnext101_32x8d_wsl':
        model = torch.hub.load('facebookresearch/WSL-Images', args.model_name)
    elif args.model_name == 'resnext101_32x8d_imgnet':
        model = torchvision.models.resnext101_32x8d(pretrained=True)
    elif args.model_name == 'resnext101_32x8d_rand':
        model = torchvision.models.resnext101_32x8d(pretrained=False)
    else:
        raise ValueError('Model not available.')

    child_list = list(model.children())

    # Ugly case by case (yikes Pytorch!)
    if args.layer == 0:
        layer_list = child_list[:5]
        layer_list.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        in_dim = 256
    elif args.layer == 1:
        layer_list = child_list[:6]
        layer_list.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        layer_list.append(Flatten())
        layer_list.append(torch.nn.Linear(in_features=512, out_features=256, bias=False))
        in_dim = 256
    elif args.layer == 2:
        top = child_list[6][:5]
        layer_list = child_list[:6] + list(top.children())
        layer_list.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        layer_list.append(Flatten())
        layer_list.append(torch.nn.Linear(in_features=1024, out_features=256, bias=False))
        in_dim = 256
    elif args.layer == 3:
        top = child_list[6][:11]
        layer_list = child_list[:6] + list(top.children())
        layer_list.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        layer_list.append(Flatten())
        layer_list.append(torch.nn.Linear(in_features=1024, out_features=256, bias=False))
        in_dim = 256
    elif args.layer == 4:
        top = child_list[6][:17]
        layer_list = child_list[:6] + list(top.children())
        layer_list.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        layer_list.append(Flatten())
        layer_list.append(torch.nn.Linear(in_features=1024, out_features=256, bias=False))
        in_dim = 256
    elif args.layer == 5:
        top = child_list[6][:23]
        layer_list = child_list[:6] + list(top.children())
        layer_list.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        layer_list.append(Flatten())
        layer_list.append(torch.nn.Linear(in_features=1024, out_features=256, bias=False))
        in_dim = 256
    elif args.layer == 6:
        top = child_list[7][:2]
        layer_list = child_list[:7] + list(top.children())
        layer_list.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        layer_list.append(Flatten())
        layer_list.append(torch.nn.Linear(in_features=2048, out_features=256, bias=False))
        in_dim = 256
    elif args.layer == 7:
        top = child_list[7][:3]
        layer_list = child_list[:7] + list(top.children())
        layer_list.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        layer_list.append(Flatten())
        layer_list.append(torch.nn.Linear(in_features=2048, out_features=256, bias=False))
        in_dim = 256
    elif args.layer == 8:
        layer_list = child_list[:9]
        layer_list.append(Flatten())
        layer_list.append(torch.nn.Linear(in_features=2048, out_features=256, bias=False))
        in_dim = 256
    elif args.layer == 9:
        model = LastLayerModel(model)
        layer_list = list(model.children())
        layer_list.append(Flatten())
        layer_list.append(torch.nn.Linear(in_features=1000, out_features=256, bias=False))
        in_dim = 256
    else:
        raise ValueError('Not supported.')

    layer_list.append(Flatten())

    if args.subject == 'CSI1':
        out_dim_dict = {'LHPPA': 131, 'RHEarlyVis': 285, 'LHRSC': 86, 'RHRSC': 143, 'LHLOC': 152, 'RHOPA': 187,
                        'LHEarlyVis': 210, 'LHOPA': 101, 'RHPPA': 200, 'RHLOC': 190}
    elif args.subject == 'CSI2':
        out_dim_dict = {'LHPPA': 172, 'RHEarlyVis': 241, 'LHRSC': 59, 'RHRSC': 278, 'LHLOC': 327, 'RHOPA': 95,
                        'LHEarlyVis': 254, 'LHOPA': 85, 'RHPPA': 198, 'RHLOC': 561}
    elif args.subject == 'CSI3':
        out_dim_dict = {'LHPPA': 112, 'RHEarlyVis': 696, 'LHRSC': 78, 'RHRSC': 116, 'LHLOC': 430, 'RHOPA': 205,
                        'LHEarlyVis': 522, 'LHOPA': 187, 'RHPPA': 161, 'RHLOC': 597}

    layer_list.append(torch.nn.Linear(in_dim, out_dim_dict[args.region]))
    model = torch.nn.Sequential(*layer_list)

    # linear decoding
    if args.freeze_trunk:
        set_parameter_requires_grad(model)

    model = torch.nn.DataParallel(model).cuda()

    print(model)
    print('Freeze trunk:', args.freeze_trunk)

    return model


def dataset_generator(data_file, region):
    '''Generate train and test datasets'''
    data = np.load(data_file, allow_pickle=True)

    # generate train
    train_x = data['train_x']
    train_y = data['train_y'].item()[region]

    train_x_tensor = []
    for i in range(train_x.shape[0]):
        train_x_tensor.append(
            transforms.functional.normalize(transforms.functional.to_tensor(train_x[i]),
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]).view(-1, 3, 224, 224))

    train_x_tensor = torch.cat(train_x_tensor, dim=0)
    train_y_tensor = torch.from_numpy(train_y).float()
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor)

    # generate test
    test_x = data['test_x']
    test_y = data['test_y'].item()[region]

    test_x_tensor = []
    for i in range(test_x.shape[0]):
        test_x_tensor.append(
            transforms.functional.normalize(transforms.functional.to_tensor(test_x[i]),
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]).view(-1, 3, 224, 224))

    test_x_tensor = torch.cat(test_x_tensor, dim=0)
    test_y_tensor = torch.from_numpy(test_y).float()
    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_y_tensor)

    # generate val
    val_x = data['val_x']
    val_y = data['val_y'].item()[region]

    val_x_tensor = []
    for i in range(val_x.shape[0]):
        val_x_tensor.append(
            transforms.functional.normalize(transforms.functional.to_tensor(val_x[i]),
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]).view(-1, 3, 224, 224))

    val_x_tensor = torch.cat(val_x_tensor, dim=0)
    val_y_tensor = torch.from_numpy(val_y).float()
    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_y_tensor)

    return train_dataset, test_dataset, val_dataset


def train(model, criterion, optimizer, train_loader):
    '''Train model for one epoch'''

    # turn on train mode
    model.train()

    losses = []
    end = time.time()

    for i, (images, target) in enumerate(train_loader):

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # record loss
        losses.append(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # measure elapsed time
    print('Epoch elapsed time:', time.time() - end)

    tr_loss = np.mean(losses)

    print('Epoch traininig loss:', tr_loss)

    # return average training loss in this epoch
    return tr_loss


def validate(model, criterion, val_loader):
    '''Evaluate model on test data'''

    # turn on eval mode
    model.eval()

    losses = []
    outs = []
    trgts = []

    with torch.no_grad():

        for i, (images, target) in enumerate(val_loader):

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # record loss
            losses.append(loss.item())
            outs.append(output.cpu().numpy())
            trgts.append(target.cpu().numpy())

        val_loss = np.mean(losses)

        print('Epoch test loss:', val_loss)

    # return average test loss in this epoch
    return val_loss, np.concatenate(outs, axis=0), np.concatenate(trgts, axis=0)


def plot_preds(output, target):
    '''To visualize prediction accuracy'''
    plt.clf()

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.plot(target[i, :], 'r-')
        plt.plot(output[i, :], 'b-')
        plt.xticks([], [])
        plt.yticks([], [])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('test_predictions.pdf', bbox_inches='tight')


if __name__ == "__main__":

    args = parser.parse_args()

    # Set-up grid (TO DO: find a better way to do this)
    job_idx = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    regions = ['LHPPA', 'RHEarlyVis', 'LHRSC', 'RHRSC', 'LHLOC', 'RHOPA', 'LHEarlyVis', 'LHOPA', 'RHPPA', 'RHLOC']
    layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    rhos = [0., 5e-5, 5e-4]
    RR, LL, PP = np.meshgrid(regions, layers, rhos)
    RR = RR.flatten()
    LL = LL.flatten()
    PP = PP.flatten()
    args.region = RR[job_idx]
    args.layer = LL[job_idx]
    args.weight_decay = PP[job_idx]
    # #######################

    torch_hub_dir = '/misc/vlgscratch4/LakeGroup/emin/robust_vision/pretrained_models'
    torch.hub.set_dir(torch_hub_dir)

    model = load_model(args)

    data_file = '../data/' + args.subject + '_data.npz'
    train_dataset, test_dataset, val_dataset = dataset_generator(data_file, args.region)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, sampler=None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True, sampler=None)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, sampler=None)

    tr_losses = np.zeros(args.epochs)
    test_losses = np.zeros(args.epochs)
    val_losses = np.zeros(args.epochs)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.L1Loss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    best_val_loss = 1000000

    for i in range(args.epochs):
        tr_loss = train(model, criterion, optimizer, train_loader)
        test_loss, test_output, test_target = validate(model, criterion, test_loader)
        val_loss, val_output, val_target = validate(model, criterion, val_loader)

        tr_losses[i] = tr_loss
        test_losses[i] = test_loss
        val_losses[i] = val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_val_output = val_output
            best_val_target = val_target

            best_test_output = test_output
            best_test_target = test_target

        print('End of epoch:', i + 1)

    save_filename = args.subject + '_' + args.region + '_' +  str(args.layer) + '_' + str(args.weight_decay) + '_' + \
                    args.model_name + '.tar'

    # plot_preds(best_output, best_target)

    np.savez(save_filename, tr_losses=tr_losses, test_losses=test_losses, val_losses=val_losses,
             best_val_output=best_val_output, best_val_target=best_val_target,
             best_test_output=best_test_output, best_test_target=best_test_target)