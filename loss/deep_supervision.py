import numpy as np
import pytorch_msssim
import torch
from torch import nn

from losses.neuralNDCG import neuralNDCG
from scipy.stats import kendalltau

from model.sp_blocks import *
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1

# TODO Put these in nnunetv2/training/loss/deep_supervision.py


class DD_Loss(nn.Module):
    def __init__(self, type_='dice', ret_mean=True):
        super(DD_Loss, self).__init__()
        self.smooth = 1e-5
        self.dilate_pool = Dil_Pool(3)
        self.type_ = type_
        self.ret_mean = ret_mean
        self.apply_nonlin = softmax_helper_dim1
        self.smooth = 1e-5

    def forward(self, x, y, dil_times=0, type_=None):
        x = self.apply_nonlin(x)
        if type_ is not None:
            self.type_ = type_

        do_bg = False
        if self.type_.find('do_bg') != -1:
            do_bg = True

        axes = tuple(range(2, x.ndim))
        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                for i in range(dil_times):
                    y = self.dilate_pool(y.float()).int()
                y_onehot = torch.zeros(x.shape, device=x.device)
                y_onehot.scatter_(1, y.long(), 1)

            if not do_bg:
                y_onehot = y_onehot[:, 1:]
            fg_sum_gt = y_onehot.sum(axes)
            # print(fg_sum_gt.shape)
            # fg_sum_n_gt = (1 - y_onehot[1:]).sum(axes)

        if not do_bg:
            x = x[:, 1:]
            for i in range(dil_times):
                # 膨胀
                x = self.dilate_pool(x.float())
        else:
            x_bg = -x[:, 0:1].clone()
            x_fg = x[:, 1:].clone()
            for i in range(dil_times):
                # 膨胀
                x_bg = self.dilate_pool(x_bg.float())
                x_fg = self.dilate_pool(x_fg.float())
            x_new = torch.concatenate([-x_bg, x_fg], dim=1)
            x = x_new

        if self.type_.find('recall') != -1:
            tp = x * y_onehot
            fg_sum_tp_pred = tp.sum(axes)
            # fg_sum_pred = x[:, 1:].sum(axes)

            loss_tp = (fg_sum_tp_pred + self.smooth) / torch.clip(fg_sum_gt + self.smooth, 1e-8)
            loss_tp = -(loss_tp.mean())
            loss = loss_tp
            return loss
        elif self.type_.find('dice') != -1:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
            dc = (2 * intersect + self.smooth) / (torch.clip(fg_sum_gt + sum_pred + self.smooth, 1e-8))
            return -(dc.mean())


class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(DeepSupervisionWrapper, self).__init__()
        assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
        self.weight_factors = list(weight_factors)
        self.loss = loss
        self.times = 1
        self.dilate_pool = Dil_Pool()
        self.soft_dil_loss = DD_Loss('dice')
        self.d3_weights = None
        self.open_times = 0
        self.deep_type = None
        self.epoch = 0
        self.d3x_start = 0
        self.rle_start = 0

    def update_epoch(self, epoch):
        self.epoch = epoch
        print(f'epoch: {epoch}')
        if self.deep_type is not None:
            if epoch == self.d3x_start:
                if self.deep_type.find('d3x') != -1:
                    print(f'start {self.deep_type}')
            if epoch == self.rle_start:
                if self.deep_type.find('rle') != -1:
                    print(f'start {self.deep_type}')

    def forward(self, *args):
        assert all([isinstance(i, (tuple, list)) for i in args]), \
            f"all args must be either tuple or list, got {[type(i) for i in args]}"
        # we could check for equal lengths here as well, but we really shouldn't overdo it with checks because
        # this code is executed a lot of times!

        if self.weight_factors is None:
            weights = [1., ] * len(args[0])
        else:
            weights = self.weight_factors
        if self.d3_weights is None:
            self.d3_weights = [weights[0], 1.]
        if len(args[0]) <= len(args[1]):
            loss_sum = sum(
                [weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])
        # elif len(args[0]) > len(args[1]):
        else:
            deep_type = args[0].pop(-1)
            if self.deep_type is None:
                print('*' * 10, 'loss_type', deep_type)
                if deep_type.find('d3x_3d') != -1:
                    print(f'weight: {args[0][-1]}')
                self.deep_type = deep_type
            if deep_type.find('dil_loss') != -1:
                dil_times = args[0].pop(-1)
                loss_sum = sum(
                    [weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])
                loss_sum += sum(
                    [weights[i] * self.soft_dil_loss(*inputs, dil_times=dil_times, type_=deep_type + ' dice')
                     for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])
            else:
                # Other
                loss_sum = sum(
                    [weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])

        return loss_sum

