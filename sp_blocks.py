import torch
from torch import nn as nn


class Pro_Block(nn.Module):
    def __init__(self, ch_in, reduction=4):
        """
        Projection Attention Block
        """
        super(Pro_Block, self).__init__()
        # self.avg_pools = nn.ModuleList()
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

    def forward(self, x):
        dim_pro = []
        for i in range(3):
            # TODO， mean？
            pro = torch.mean(x, dim=(i+2)).unsqueeze(i+2)
            pro = pro.expand_as(x)
            pro = pro + x
            # pro = self.avg_pools[i](x)
            pro = self.fcs[i](pro)
            dim_pro.append(pro)
        y = torch.concatenate(dim_pro, dim=1)
        y = self.sig(y)

        # return x * y + x
        return y


class Neigh_Block(nn.Module):
    """
    Neighboring Voxel Perception Block
    """

    def __init__(self, in_chn, class_num=2, kernel=3, strike=1, padding=1, dil_times=4):
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
                    nn.Conv3d(in_chn, in_chn, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(in_chn, eps=1e-5, affine=True),
                    nn.LeakyReLU(inplace=True),
                    # nn.Sigmoid()
                )
            )
        self.final_conv = nn.Sequential(
            nn.Conv3d(in_chn, in_chn, kernel_size=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        mid = x
        mid_mean = x.detach().clone()
        mid_max = x.detach().clone()

        out_mean = x.clone()
        out_max = x.clone()

        for i in range(self.dil_times):
            mid_mean = self.mean_pool(mid_mean)
            mid_max = self.max_pool(mid_max)
            out_mean += mid_mean
            out_max += mid_max
        out_mean /= self.dil_times + 1
        out_max /= self.dil_times + 1
        out = self.finals[0](mid) + self.finals[1](out_max) + self.finals[2](out_mean)

        out = self.final_conv(out)

        return out


class Mean_Pool(nn.Module):
    def __init__(self, k=3):
        super(Mean_Pool, self).__init__()
        self.k = k
        self.avg_pool = nn.AvgPool3d(kernel_size=k, stride=1, padding=0)

    def forward(self, x):
        x_dil = nn.functional.pad(x, pad=[self.k // 2] * 6, mode='reflect')
        x_dil = self.avg_pool(x_dil)
        return x_dil


class Dil_Pool(nn.Module):
    def __init__(self, k=3):
        super(Dil_Pool, self).__init__()
        self.k = k
        self.max_pool = nn.MaxPool3d(kernel_size=k, stride=1, padding=0)

    def forward(self, x):
        x_dil = nn.functional.pad(x, pad=[self.k // 2] * 6, mode='reflect')
        x_dil = self.max_pool(x_dil)
        return x_dil
