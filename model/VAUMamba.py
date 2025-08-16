from model.UMambaBot_3d import get_umamba_bot_3d_wrap
from model.sp_blocks import *

act_ = nn.LeakyReLU


class Loss_Wrap(nn.Module):
    def __init__(self, in_chns, class_num=2, scale=1, ds=True, base_model=None, rle_epoch=0,
                 loss_type='dice', dil_times=3):
        super(Loss_Wrap, self).__init__()
        # self.__class__.__name__ = 'RLELoss'
        self.loss_type = loss_type
        assert base_model is not None

        self.net = base_model
        self.__class__.__name__ = f'Loss{loss_type.replace(" ", "")}' + self.net.__class__.__name__
        self.rle_epoch = rle_epoch
        self.dil_times = dil_times
        print(f'-----loss: {loss_type}, dil: {dil_times}')

        self.decoder = Empty_Decoder(ds=ds)

    def forward(self, x):
        out = self.net(x)
        if self.decoder.deep_supervision:
            # You can do it you way
            if self.loss_type.find('dil') != -1:
                out.append(self.dil_times)
                out.append('dil_loss ' + self.loss_type)
            else:
                out.append(self.loss_type)
        else:
            out = out[0]

        return out


class VA_Wrap(nn.Module):
    def __init__(self, in_chns, class_num, scale=1, ds=True, vessel=True, dil_loss=False,
                 pro_loss=False, neigh_dil=4, frangi_th=0.2, vessel_cpu=False, neigh=True, pro=True, dil_cig=None,
                 base_model=None, neigh_block=Neigh_BlockV2, change_vessel=False):
        super(VA_Wrap, self).__init__()
        self.__class__.__name__ = 'VA'
        if dil_cig is None:
            dil_cig = [3, 1]
        self.dil_cig = dil_cig
        print(f'---config: n:{neigh_dil}, frangi:{frangi_th}---')
        # neigh在这边
        self.decoder = Empty_Decoder(ds)

        # dilate_times = 3
        self.heat_weights_block = HeatWeights_Block(dilate_times=5)

        self.net = base_model(in_chns, class_num, ds=True, neigh=neigh, pro=pro, vessel=vessel, scale=scale,
                              neigh_block=neigh_block, neigh_dil=neigh_dil)

        self.__class__.__name__ += self.net.__class__.__name__

        # 只对最后输出进行neigh？计算量也小
        self.neigh = neigh
        if neigh:
            self.__class__.__name__ += 'wN'

        if pro:
            self.__class__.__name__ += 'wP'

        self.vessel = vessel
        if self.vessel:
            self.frangi_block = Vessel_Enhance_Block(frangi_th=frangi_th, cpu=vessel_cpu)
            self.__class__.__name__ += 'wF'

        self.dil_loss = dil_loss
        if self.dil_loss:
            self.__class__.__name__ += 'DilLoss'

        self.pro_loss = pro_loss
        if self.pro_loss:
            self.__class__.__name__ += 'ProLoss'

        # for visualized img
        self.had_record = True

    def forward(self, x):
        # if self.neigh:
        #     x = self.neigh_block(x)

        vessel_weights = None
        if self.vessel:
            self.frangi_block = self.frangi_block.cuda()
            device_ = x.device
            vessel_enhance = self.frangi_block(x.cuda()).to(device=device_).float()
            vessel_weights = self.heat_weights_block(vessel_enhance)

            if not self.had_record:
                import numpy as np
                import SimpleITK as sitk
                path_ = '/home/chenxianhao/Liver/Liver_Vessel_Seg/isimg/'
                image = vessel_enhance.squeeze(0).squeeze(0).cpu().numpy()
                img_itk = sitk.GetImageFromArray(image.astype(np.float32))
                sitk.WriteImage(img_itk, path_ + f'vessel.nii.gz')

                image = vessel_weights[0].squeeze(0).squeeze(0).cpu().numpy()
                img_itk = sitk.GetImageFromArray(image.astype(np.float32))
                sitk.WriteImage(img_itk, path_ + f'vessel_weight.nii.gz')
                print('---' + f'vessel.nii.gz')

        if not self.had_record:
            import numpy as np
            import SimpleITK as sitk
            path_ = '/home/chenxianhao/Liver/Liver_Vessel_Seg/isimg/'
            image = x.squeeze(0).squeeze(0).cpu().numpy()
            img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            sitk.WriteImage(img_itk, path_ + f'input.nii.gz')
            print('---' + f'input.nii.gz')
            self.had_record = True

        out = self.net(x, vessel_weights)
        # if self.neigh:
        #     out[0] = self.neigh_block(out[0])

        if not self.decoder.deep_supervision:
            out = out[0]

        return out


def get_vaumamba(input_channels, output_channels, deep_supervision, dd_loss=False):
    """
    You can put this in nnUnet at nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py build_network_architecture
    Notice: you should change the nnUNetPlans.json refer to plan/* and set os.environ['nnUNet_compile'] = 'False'
    :param input_channels:
    :param output_channels:
    :param deep_supervision:
    :param dd_loss:
    :return:
    """
    # get a model with ddloss
    base_model = VA_Wrap(in_chns=1, class_num=2,
                         vessel=True, vessel_cpu=False, neigh=True, pro=True,
                         base_model=get_umamba_bot_3d_wrap, neigh_block=Neigh_BlockV3, neigh_dil=3)
    if not dd_loss:
        return base_model
    network = Loss_Wrap(input_channels, output_channels, scale=1, ds=deep_supervision, base_model=base_model,
                        loss_type='dil', dil_times=3)
    return network
