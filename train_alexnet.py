"""Fit AlexNets
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
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default:4)')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int, help='mini-batch size (default: 732), this is the total batch'
                                                                ' size of all GPUs on the current node when using Data '
                                                                'Parallel or Distributed Data Parallel')
parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=0.0001, type=float, help='weight decay (default: 0)')
parser.add_argument('--subject', default='CSI1', type=str, help='subject', choices=['CSI1', 'CSI2', 'CSI3'])
parser.add_argument('--region', default='RHOPA', type=str, help='subject',
                    choices=['LHPPA', 'RHEarlyVis', 'LHRSC', 'RHRSC', 'LHLOC', 'RHOPA', 'LHEarlyVis', 'LHOPA', 'RHPPA',
                             'RHLOC'])
parser.add_argument('--model-name', type=str, default='resnext101_32x8d_rand',
                    choices=['alexnet_rand', 'alexnet_imgnet'],
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
    "Loads model architecture"
    if args.model_name == 'alexnet_imgnet':
        model = torchvision.models.alexnet(pretrained=True)
    elif args.model_name == 'alexnet_rand':
        model = torchvision.models.alexnet(pretrained=False)
    else:
        raise ValueError('Model not available.')

    # Ugly case by case (yikes Pytorch!)
    if args.layer == 0:
        layer_list = list(model.features[:2])
        layer_list.append(torch.nn.AdaptiveAvgPool2d(output_size=(16, 16)))
        in_dim = 16384
    elif args.layer == 1:
        layer_list = list(model.features[:5])
        layer_list.append(torch.nn.AdaptiveAvgPool2d(output_size=(8, 8)))
        in_dim = 12288
    elif args.layer == 2:
        layer_list = list(model.features[:8])
        layer_list.append(torch.nn.AdaptiveAvgPool2d(output_size=(6, 6)))
        in_dim = 13824
    elif args.layer == 3:
        layer_list = list(model.features[:10])
        layer_list.append(torch.nn.AdaptiveAvgPool2d(output_size=(6, 6)))
        in_dim = 9216
    elif args.layer == 4:
        layer_list = list(model.features[:12])
        layer_list.append(torch.nn.AdaptiveAvgPool2d(output_size=(6, 6)))
        in_dim = 9216
    elif args.layer == 5:
        layer_list = list(model.features)
        layer_list.append(model.avgpool)
        in_dim = 9216
    elif args.layer == 6:
        layer_list = list(model.features)
        layer_list.append(model.avgpool)
        layer_list.append(Flatten())
        layer_list.append(model.classifier[:3])
        in_dim = 4096
    elif args.layer == 7:
        layer_list = list(model.features)
        layer_list.append(model.avgpool)
        layer_list.append(Flatten())
        layer_list.append(model.classifier[:6])
        in_dim = 4096
    elif args.layer == 8:
        model = LastLayerModel(model)
        layer_list = list(model.children())
        in_dim = 1000
    else:
        raise ValueError('Not supported.')

    layer_list.append(Flatten())

    out_dim_dict = {'LHPPA': 131, 'RHEarlyVis': 285, 'LHRSC': 86, 'RHRSC': 143, 'LHLOC': 152, 'RHOPA': 187,
                    'LHEarlyVis': 210, 'LHOPA': 101, 'RHPPA': 200, 'RHLOC': 190}

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
    train_y = data['train_y'].item()[region]  # make this generic

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
    test_y = data['test_y'].item()[region]  # make this generic

    test_x_tensor = []
    for i in range(test_x.shape[0]):
        test_x_tensor.append(
            transforms.functional.normalize(transforms.functional.to_tensor(test_x[i]),
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]).view(-1, 3, 224, 224))

    test_x_tensor = torch.cat(test_x_tensor, dim=0)
    test_y_tensor = torch.from_numpy(test_y).float()
    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_y_tensor)

    return train_dataset, test_dataset


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

    # BAD BAD BAD
    job_idx = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    regions = ['LHPPA', 'RHEarlyVis', 'LHRSC', 'RHRSC', 'LHLOC', 'RHOPA', 'LHEarlyVis', 'LHOPA', 'RHPPA', 'RHLOC']
    layers = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    RR, LL = np.meshgrid(regions, layers)
    RR = RR.flatten()
    LL = LL.flatten()
    args.region = RR[job_idx]
    args.layer = LL[job_idx]
    # BAD BAD BAD

    model = load_model(args)

    data_file = '../data/' + args.subject + '_data.npz'
    train_dataset, test_dataset = dataset_generator(data_file, args.region)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, sampler=None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True, sampler=None)

    tr_losses = np.zeros(args.epochs)
    test_losses = np.zeros(args.epochs)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.L1Loss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    best_test_loss = 1000000

    for i in range(args.epochs):
        tr_loss = train(model, criterion, optimizer, train_loader)
        test_loss, output, target = validate(model, criterion, test_loader)

        tr_losses[i] = tr_loss
        test_losses[i] = test_loss

        if test_loss < best_test_loss:
            best_output = output
            best_target = target

        print('End of epoch:', i + 1)

    save_filename = args.subject + '_' + args.region + '_' +  str(args.layer) + '_' + args.model_name + '.tar'

    # plot_preds(best_output, best_target)

    np.savez(save_filename, tr_losses=tr_losses, test_losses=test_losses, best_output=best_output,
             best_target=best_target)