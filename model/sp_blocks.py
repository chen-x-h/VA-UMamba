from __future__ import division, print_function

import numpy as np
import torch
from torch import nn as nn

# cucim cuda==12.0 no 12.1
from cucim.skimage.filters import frangi
from cucim.skimage import morphology
import cupy
from skimage.filters import frangi as frangi_cpu
from skimage import morphology as morphology_cpu


class Pro_Block(nn.Module):
    def __init__(self, ch_in, reduction=4):
        super(Pro_Block, self).__init__()
        self.fcs = nn.ModuleList()
        for i in range(3):
            # shape_ = [None] * 3
            # shape_[i] = 1
            # self.avg_pools.append(
            #     nn.AdaptiveAvgPool3d(shape_)
            # )
            self.fcs.append(
                nn.Sequential(
                    nn.Conv3d(ch_in, ch_in // reduction, kernel_size=1, bias=False),
                    # nn.ReLU(inplace=True),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(ch_in // reduction, ch_in, kernel_size=1, bias=False),
                    # nn.Sigmoid()
                )
            )
        self.sig = nn.Sequential(
            nn.Conv3d(ch_in * 3, ch_in, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.had_record = True

    def forward(self, x):
        dim_pro = []
        for i in range(3):

            pro = torch.mean(x, dim=(i + 2)).unsqueeze(i + 2)
            pro = pro.expand_as(x)
            pro = pro + x
            # pro = self.avg_pools[i](x)
            pro = self.fcs[i](pro)
            if not self.had_record:
                import numpy as np
                import SimpleITK as sitk
                path_ = '/home/chenxianhao/Liver/Liver_Vessel_Seg/isimg/'
                image = pro.squeeze(0).squeeze(0).cpu().numpy()
                img_itk = sitk.GetImageFromArray(image.astype(np.float32))
                sitk.WriteImage(img_itk, path_ + f'pab0_{i}.nii.gz')
                print('---' + f'pab0_{i}.nii.gz')
            dim_pro.append(pro)

        y = torch.concatenate(dim_pro, dim=1)
        y = self.sig(y)

        if not self.had_record:
            import numpy as np
            import SimpleITK as sitk
            path_ = '/home/chenxianhao/Liver/Liver_Vessel_Seg/isimg/'
            image = y.squeeze(0).squeeze(0).cpu().numpy()
            img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            sitk.WriteImage(img_itk, path_ + f'pab1.nii.gz')
            print('---' + f'pab1.nii.gz')
        self.had_record = True

        # return x * y + x
        return y


class Neigh_Block(nn.Module):
    """
    NVPB v1
    """

    def __init__(self, in_chn, class_num=2, kernel=3, strike=1, padding=1, dil_times=4, conv_3d=nn.Conv3d):
        super(Neigh_Block, self).__init__()
        self.dil_times = dil_times
        self.class_num = class_num
        self.in_chn = in_chn
        k = 3

        self.mean_pool = Mean_Pool(k=k)
        self.max_pool = Dil_Pool(k=k)

        self.finals = nn.ModuleList()
        for i in range(3):
            self.finals.append(
                nn.Sequential(
                    # TODO
                    # act(),
                    conv_3d(in_chn, in_chn, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(in_chn, eps=1e-5, affine=True),
                    nn.LeakyReLU(inplace=True),
                    # nn.Sigmoid()
                )
            )

        self.final_conv = nn.Sequential(
            nn.Conv3d(in_chn, in_chn, kernel_size=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, change_param=None):

        if change_param is None:
            change_param = self.dil_times
        else:
            change_param = int(change_param * self.dil_times)
            # print(change_param, "debug")
        mid = x
        # TODO
        mid_mean = x.detach().clone()
        mid_max = x.detach().clone()

        out_mean = x.clone()
        out_max = x.clone()

        for i in range(change_param):
            mid_mean = self.mean_pool(mid_mean)
            mid_max = self.max_pool(mid_max)
            out_mean += mid_mean
            out_max += mid_max
        out_mean /= change_param + 1
        out_max /= change_param + 1
        out = self.finals[0](mid) + self.finals[1](out_max) + self.finals[2](out_mean)

        # if self.pro:
        #     out = self.pro_block(out)
        out = self.final_conv(out)

        return out


class Neigh_BlockV2(nn.Module):
    """
    NVPBv2
    """

    def __init__(self, in_chn, dil_times=3, mid_chn=2):
        super(Neigh_BlockV2, self).__init__()
        self.dil_times = dil_times
        self.in_chn = in_chn
        k = 3

        self.mean_pool = Mean_Pool(k=k)
        self.max_pool = Dil_Pool(k=k)
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_chn, mid_chn, kernel_size=1)
        )

        self.finals = nn.ModuleList()
        for i in range(3):
            self.finals.append(
                nn.Sequential(
                    # TODO
                    # act(),
                    nn.Conv3d(mid_chn, in_chn, kernel_size=1),
                    nn.InstanceNorm3d(in_chn, eps=1e-5, affine=True),
                    nn.LeakyReLU(inplace=True),
                    # nn.Sigmoid()
                )
            )

        self.final_conv = nn.Sequential(
            # conv_3d(in_chn * 3, in_chn * 3, kernel_size=3, padding=1),
            nn.Conv3d(in_chn, in_chn, kernel_size=1),
            nn.InstanceNorm3d(in_chn, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, change_param=None):

        if change_param is None:
            change_param = self.dil_times
        else:
            change_param = int(change_param * 5.)
            # print(change_param, "debug")
        mid = self.down_conv(x)
        # TODO
        mid_mean = mid.detach()
        mid_max = mid.detach()

        out_mean = mid.clone()
        out_max = mid.clone()

        for i in range(change_param):
            mid_mean = self.mean_pool(mid_mean)
            mid_max = self.max_pool(mid_max)
            out_mean += mid_mean
            out_max += mid_max
        out_mean /= change_param + 1
        out_max /= change_param + 1

        out = self.finals[0](mid) + self.finals[1](out_max) + self.finals[2](out_mean)
        out = self.final_conv(out)

        return out


class Neigh_BlockV3(nn.Module):
    """
    NVPBv3(latest)
    """

    def __init__(self, in_chn, dil_times=3, mid_chn=1):
        super(Neigh_BlockV3, self).__init__()
        self.dil_times = dil_times
        self.in_chn = in_chn
        k = 3

        self.mean_pool = Mean_Pool(k=k)
        self.max_pool = Dil_Pool(k=k)

        # self.down_conv = nn.Sequential(
        #     nn.Conv3d(in_chn, mid_chn, kernel_size=1),
        #     nn.InstanceNorm3d(mid_chn, eps=1e-5, affine=True),
        #     nn.LeakyReLU(inplace=True),
        # )

        mid_chn = in_chn
        self.finals = nn.ModuleList()
        for i in range(3):
            self.finals.append(
                nn.Sequential(
                    # TODO
                    # act(),
                    nn.Conv3d(mid_chn, in_chn, kernel_size=1),
                    nn.InstanceNorm3d(in_chn, eps=1e-5, affine=True),
                    nn.LeakyReLU(inplace=True),
                    # nn.Sigmoid()
                )
            )

        self.final_conv = nn.Sequential(
            # conv_3d(in_chn * 3, in_chn * 3, kernel_size=3, padding=1),
            # nn.Conv3d(in_chn, in_chn, kernel_size=1),
            nn.Conv3d(in_chn, in_chn, kernel_size=3, padding=1, groups=in_chn),
            nn.InstanceNorm3d(in_chn, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_chn, in_chn, kernel_size=1),
            nn.InstanceNorm3d(in_chn, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
        )

        self.had_record = True

    def forward(self, x, change_param=None):

        if change_param is None:
            change_param = self.dil_times
        else:
            change_param = int(change_param * 5.)
            # print(change_param, "debug")
        # mid = self.down_conv(x)
        mid = x.clone()
        # TODO
        mid_mean = mid.detach()
        mid_max = mid.detach()

        out_mean = mid.clone()
        out_max = mid.clone()

        if not self.had_record:
            import numpy as np
            import SimpleITK as sitk
            path_ = '/home/chenxianhao/Liver/Liver_Vessel_Seg/isimg/'
            image = x.squeeze(0).squeeze(0).cpu().numpy()
            img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            sitk.WriteImage(img_itk, path_ + f'nvpb_0.nii.gz')
            print('---' + f'nvpb_0.nii.gz')

        for i in range(change_param):
            mid_mean = self.mean_pool(mid_mean)
            mid_max = self.max_pool(mid_max)
            out_mean += mid_mean
            out_max += mid_max

            if not self.had_record:
                import numpy as np
                import SimpleITK as sitk
                path_ = '/home/chenxianhao/Liver/Liver_Vessel_Seg/isimg/'
                image = (out_max / (i + 2)).squeeze(0).squeeze(0).cpu().numpy()
                img_itk = sitk.GetImageFromArray(image.astype(np.float32))
                sitk.WriteImage(img_itk, path_ + f'nvpb_max_{i+1}.nii.gz')
                image = (out_mean / (i + 2)).squeeze(0).squeeze(0).cpu().numpy()
                img_itk = sitk.GetImageFromArray(image.astype(np.float32))
                sitk.WriteImage(img_itk, path_ + f'nvpb_mean_{i + 1}.nii.gz')
                print('---' + f'nvpb_{i+1}.nii.gz')
        out_mean /= change_param + 1
        out_max /= change_param + 1

        out = self.finals[0](mid) + self.finals[1](out_max) + self.finals[2](out_mean)
        out = self.final_conv(out)

        if not self.had_record:
            import numpy as np
            import SimpleITK as sitk
            path_ = '/home/chenxianhao/Liver/Liver_Vessel_Seg/isimg/'
            image = out.squeeze(0).squeeze(0).cpu().numpy()
            img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            sitk.WriteImage(img_itk, path_ + f'nvpb_f.nii.gz')
            print('---' + f'nvpb_f.nii.gz')

        if not self.had_record:
            self.had_record = True

        return out


class Mean_Pool(nn.Module):
    def __init__(self, k=3, padding_in=False):
        super(Mean_Pool, self).__init__()
        self.k = k
        self.padding_in = padding_in
        self.avg_pool = nn.AvgPool3d(kernel_size=k, stride=1, padding=0)
        if padding_in:
            self.avg_pool = nn.AvgPool3d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        if self.padding_in:
            return self.avg_pool(x)
        x_dil = nn.functional.pad(x, pad=[self.k // 2] * 6, mode='reflect')
        x_dil = self.avg_pool(x_dil)
        return x_dil


class Dil_Pool(nn.Module):
    def __init__(self, k=3, padding_in=False):
        super(Dil_Pool, self).__init__()
        self.k = k
        self.max_pool = nn.MaxPool3d(kernel_size=k, stride=1, padding=0)
        self.padding_in = padding_in
        if padding_in:
            self.max_pool = nn.MaxPool3d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        if self.padding_in:
            return self.max_pool(x)
        x_dil = nn.functional.pad(x, pad=[self.k // 2] * 6, mode='reflect')
        x_dil = self.max_pool(x_dil)
        return x_dil


def one_hot_tran(t, num_class=2):
    shape = list(t.shape)
    shape[1] = num_class
    x_onehot = torch.zeros(shape, device=t.device)
    t_long = t.long()
    t_long = torch.clamp(t_long, min=0, max=num_class-1)
    x_onehot.scatter_(1, t_long, 1)
    return x_onehot


def logit_tran(t):
    return torch.argmax(torch.softmax(t, dim=1), dim=1).unsqueeze(1)


def color_text(t):
    return f'\033[1;33m {t} \033[0m'


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

    def get_ske(self, x, d_times):
        # print(d_times, 'debug')
        x = x.to(dtype=torch.float)
        x[x > 0] = 1
        x[x < 1e-8] = 0
        ske = x.clone().detach()

        factor = self.rate[1] / (d_times * 2 - 1)
        ske_ = factor * ske

        ske_j = ske.clone()
        ske_j[ske_j > 0] = 1
        ske_j[ske_j < 1e-8] = 0
        for i in range(d_times - 1):
            ske_j = self.dilate_pool(ske_j)
            ske_j[ske_j > 0] = 1
            ske_j[ske_j < 1e-8] = 0
            ske_ += ske_j * factor

        ske_i = -ske
        ske_i[ske_i < 0] = -1
        ske_i[ske_i > -1e-8] = 0
        for i in range(d_times - 1):
            ske_i = self.dilate_pool(ske_i)
            ske_i[ske_i < 0] = -1
            ske_i[ske_i > -1e-8] = 0
            ske_ += (-ske_i) * factor
        ske_ += self.rate[0]

        return ske_

    def forward(self, ske, d_times=None):
        ske_outs = []
        if d_times is None:
            d_times = self.dilate_times
        else:
            d_times = int(5 * d_times + 1)
        ske_ = self.get_ske(ske, d_times)
        ske_outs.append(ske_)
        for i in range(1, self.deep):
            ske_outs.append(self.max_pool(ske_outs[-1]))

        return ske_outs


class Empty_Decoder(nn.Module):
    def __init__(self, ds=True):
        """
        for nnUnet
        :param ds:
        """
        super(Empty_Decoder, self).__init__()
        self.deep_supervision = ds


class Vessel_Enhance_Block(nn.Module):
    def __init__(self, min_max=None, min_size=200, frangi_th=0.2, cpu=False, param_change_rate=0.1):
        super(Vessel_Enhance_Block, self).__init__()
        if min_max is None:
            min_max = [0, 500]
        self.min_max = min_max
        self.frangi_param = [0.5, 1., 5., np.arange(1., 2.5, 0.3), frangi_th]
        self.th_param = [0.1, 0.5]
        self.frangi_param_base = [0.5, 1., 5., frangi_th, 0.1, 0.5]
        # TODO
        self.param_change_rate = param_change_rate
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

    def forward(self, image, frangi_param=None):
        """
        frangi_param: 6 param, 0.-1.
        """
        if frangi_param is not None and frangi_param[0] is not None:
            # TODO batch_Size must be 1
            self.frangi_param = [
                self.frangi_param_base[0] * (1. + (frangi_param[0] - 0.5) * self.param_change_rate),
                self.frangi_param_base[1] * (1. + (frangi_param[1] - 0.5) * self.param_change_rate),
                self.frangi_param_base[2] * (1. + (frangi_param[2] - 0.5) * self.param_change_rate),
                np.arange(1., 2.5, 0.3),
                self.frangi_param_base[3]
            ]
            # print(self.frangi_param, "debug")
            # print(self.th_param)

        th_mask = self.get_th(image)
        frangi_mask = self.get_frangi(image)
        frangi_mask = th_mask.bool() & frangi_mask.bool()
        frangi_mask = self.remove_small_region(frangi_mask.int())
        return frangi_mask

