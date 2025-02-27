# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn

from model.UMambaBot_3d import get_umamba_bot_3d
from cucim.skimage.filters import frangi
from cucim.skimage import morphology
import cupy
from skimage.filters import frangi as frangi_cpu
from skimage import morphology as morphology_cpu

from model.sp_blocks import Neigh_Block, Mean_Pool, Dil_Pool, Pro_Block

"""
model implementation of VA-UMamba
"""


class Empty_Decoder(nn.Module):
    def __init__(self, ds=True):
        super(Empty_Decoder, self).__init__()
        self.deep_supervision = ds


class Vessel_Enhance_Block(nn.Module):
    def __init__(self, min_max=None, min_size=200, frangi_th=0.2, cpu=False):
        super(Vessel_Enhance_Block, self).__init__()
        if min_max is None:
            min_max = [0, 500]
        self.min_max = min_max
        self.frangi_param = [0.5, 1., 5., np.arange(1., 2.5, 0.3), frangi_th]
        self.th_param = [0.1, 0.5]
        self.min_size = min_size
        self.cpu = cpu

    def get_th(self, image):
        image_ = image.clone()
        image_[image_ < self.th_param[0]] = 0
        image_[image_ > self.th_param[1]] = 0
        image_[image_ > 0] = 1
        return image_

    def get_frangi(self, x_):
        x = x_ * (self.min_max[1] - self.min_max[0]) + self.min_max[0]
        device_ = x.device
        # image = torch.detach().to_dlpack(x)
        if not self.cpu:
            image = torch.to_dlpack(x.clone())
            image = cupy.from_dlpack(image)
        else:
            image = x.clone().detach().cpu().numpy()
        batch_size = image.shape[0]
        frangi_img = []
        for i in range(batch_size):
            # print(image.device)
            if self.cpu:
                frangi_img.append(
                    np.expand_dims(
                        frangi_cpu(image[i][0], alpha=self.frangi_param[0], beta=self.frangi_param[1],
                                   gamma=self.frangi_param[2],
                                   sigmas=self.frangi_param[3], black_ridges=False),
                        0
                    )
                )
            else:
                frangi_img.append(
                    cupy.expand_dims(
                        frangi(image[i][0], alpha=self.frangi_param[0], beta=self.frangi_param[1],
                               gamma=self.frangi_param[2],
                               sigmas=self.frangi_param[3], black_ridges=False),
                        0
                    )
                )
        if self.cpu:
            frangi_img = np.concatenate(frangi_img, axis=0)
            frangi_img = np.expand_dims(frangi_img, 1)
            frangi_mask = frangi_img.copy()
            frangi_mask[frangi_mask < self.frangi_param[4]] = 0
            frangi_mask[frangi_mask > 1e-8] = 1
            frangi_mask = torch.from_numpy(frangi_mask).to(device=device_).bool()
        else:
            frangi_img = cupy.concatenate(frangi_img, axis=0)
            frangi_img = cupy.expand_dims(frangi_img, 1)
            frangi_mask = frangi_img.copy()
            frangi_mask[frangi_mask < self.frangi_param[4]] = 0
            frangi_mask[frangi_mask > 1e-8] = 1
            frangi_mask = frangi_mask.toDlpack()
            frangi_mask = torch.from_dlpack(frangi_mask).to(device=device_).bool()
        return frangi_mask

    def remove_small_region(self, x):
        device_ = x.device
        if self.cpu:
            image = x.clone().detach().cpu().numpy()
        else:
            image = torch.to_dlpack(x.clone())
            image = cupy.from_dlpack(image)
        batch_size = image.shape[0]
        out_img = []
        for i in range(batch_size):
            if self.cpu:
                case = image[i][0].astype(np.bool_)
                cleaned_image = morphology_cpu.remove_small_objects(case, min_size=self.min_size, connectivity=2)
                case *= cleaned_image
                out_img.append(
                    np.expand_dims(case, 0)
                )
            else:
                case = image[i][0].astype(cupy.bool_)
                cleaned_image = morphology.remove_small_objects(case, min_size=self.min_size, connectivity=2)
                case *= cleaned_image
                out_img.append(
                    cupy.expand_dims(case, 0)
                )

        if self.cpu:
            out_img = np.concatenate(out_img, axis=0)
            out_img = np.expand_dims(out_img, 1)
            out = out_img.astype(np.int8)
            out = torch.from_numpy(out).to(device=device_).bool()
        else:
            out_img = cupy.concatenate(out_img, axis=0)
            out_img = cupy.expand_dims(out_img, 1)
            out = out_img.astype(cupy.int8)
            out = out.toDlpack()
            out = torch.from_dlpack(out).to(device=device_).bool()
        return out

    def forward(self, image):
        th_mask = self.get_th(image)
        frangi_mask = self.get_frangi(image)
        frangi_mask = th_mask.bool() & frangi_mask.bool()
        frangi_mask = self.remove_small_region(frangi_mask.int())
        return frangi_mask


class HeatWeights_Block(nn.Module):
    def __init__(self, dilate_times=4, rate=None, deep=5):
        super(HeatWeights_Block, self).__init__()
        k = 3
        self.dilate_pool = Dil_Pool(k)
        self.deep = deep
        self.max_pool = nn.MaxPool3d(2, 2)

        if dilate_times is None:
            self.dilate_times = 4
        else:
            self.dilate_times = dilate_times
        if rate is None:
            # self.rate = [0.6, 0.4]
            self.rate = [0.3, 0.7]

    def get_ske(self, x):
        x = x.to(dtype=torch.float)
        x[x > 0] = 1
        x[x < 1e-8] = 0
        ske = x.clone().detach()
        # 计算权重

        # ske_ = torch.ones_like(ske).to(dtype=torch.float, device=x.device) * self.rate[0]
        factor = self.rate[1] / (self.dilate_times * 2 - 1)
        ske_ = factor * ske
        # ske_ += factor * ske
        # ske_i = ske.clone()

        # 外层膨胀
        ske_j = ske.clone()
        ske_j[ske_j > 0] = 1
        ske_j[ske_j < 1e-8] = 0
        for i in range(self.dilate_times - 1):
            ske_j = self.dilate_pool(ske_j)
            ske_j[ske_j > 0] = 1
            ske_j[ske_j < 1e-8] = 0
            ske_ += ske_j * factor

        # 内层腐蚀
        ske_i = -ske
        ske_i[ske_i < 0] = -1
        ske_i[ske_i > -1e-8] = 0
        for i in range(self.dilate_times - 1):
            ske_i = self.dilate_pool(ske_i)
            ske_i[ske_i < 0] = -1
            ske_i[ske_i > -1e-8] = 0
            ske_ += (-ske_i) * factor
        ske_ += self.rate[0]

        # ske_ = self.ske_sig(ske_)
        return ske_

    def forward(self, ske):
        # ske must be ds
        ske_outs = []
        ske_ = self.get_ske(ske)
        ske_outs.append(ske_)
        for i in range(1, self.deep):
            ske_outs.append(self.max_pool(ske_outs[-1]))

        return ske_outs


class Mamba3dNeigh(nn.Module):
    def __init__(self, in_chns, class_num, scale=1, ds=True, vessel=False, dil_loss=False,
                 pro_loss=False, neigh_dil=4, frangi_th=0.2, vessel_cpu=False, neigh=True, pro=True, dil_cig=None):
        super(Mamba3dNeigh, self).__init__()
        self.__class__.__name__ = 'Mamba3dNeigh'
        if dil_cig is None:
            dil_cig = [3, 1]
        self.dil_cig = dil_cig
        print(f'---config: n:{neigh_dil}, frangi:{frangi_th}---')
        self.mamba_net = get_umamba_bot_3d(in_chns, class_num, deep_supervision=True, neigh=neigh, pro=pro,
                                           neigh_dil=neigh_dil, vessel=True)
        self.decoder = Empty_Decoder(ds)

        # TODO best for loc: 4
        # dilate_times = 3
        self.heat_weights_block = HeatWeights_Block(dilate_times=neigh_dil)

        if not neigh:
            self.__class__.__name__ += 'WoN'

        if not pro:
            self.__class__.__name__ += 'WoP'

        self.vessel = vessel
        if self.vessel:
            self.__class__.__name__ += 'Vessel'
            # final_in_chns += 1
            # TODO: best for loc: 0.2
            # frangi_th = 0.2
            self.frangi_block = Vessel_Enhance_Block(frangi_th=frangi_th, cpu=vessel_cpu)

    def forward(self, x):
        vessel_weights = None
        if self.vessel:
            self.frangi_block = self.frangi_block.cuda()
            device_ = x.device
            vessel_enhance = self.frangi_block(x.cuda()).to(device=device_).float()
            vessel_weights = self.heat_weights_block(vessel_enhance)
        out = self.mamba_net(x, vessel_weights)

        if not self.decoder.deep_supervision:
            out = out[0]

        return out
