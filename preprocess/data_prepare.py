import glob
import os.path
import platform
import shutil

import SimpleITK as sitk
import numpy as np
import torch
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from tqdm import tqdm

from preprocess.pre_process import CT_normalize

data_path = '/home/chenxianhao/Liver/dataset/'
dataset_names = 'Dataset001_3dir'


def move_data(dataset_name=''):

    # TODO The path in pre_process
    source_path = data_path + 'liver_path/3d_val/*.nii.gz'
    datas = glob.glob(source_path)

    # 创建nnUnet数据库
    save_path = data_path + 'nnUnet/raw/' + dataset_name
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_path + 'imagesTr'):
        os.mkdir(save_path + 'imagesTr')
    if not os.path.exists(save_path + 'imagesTs'):
        os.mkdir(save_path + 'imagesTs')
    if not os.path.exists(save_path + 'labelsTr'):
        os.mkdir(save_path + 'labelsTr')

    val_list = ''
    val_list_ = []
    for case in datas:
        name = case.split('/')[-1]
        if name.find('gt') != -1:
            # if source_path.find('val') != -1:
            #     continue
            p_ = save_path + 'labelsTr/' + name.replace('F0', 'F_').replace('_gt', '')
        else:
            type_ = 'imagesTr/'
            p_ = save_path + type_ + name.replace('F0', 'F_').replace('.nii.gz', '_0000.nii.gz')
            val_list += '            \"' + name.replace('F0', 'F_').replace('.nii.gz', '\",\n')
            val_list_.append(name.replace('F0', 'F_').replace('.nii.gz', '_0000.nii.gz'))
        shutil.copy(case, p_)
    print(val_list)
    return val_list_


def move_data_liveronly(dataset_id=17):
    sou_path = '/home/chenxianhao/Liver/dataset/liver_path/tmp_dircadb/'
    imgs = glob.glob(sou_path+'image/*.nii.gz')
    labels = glob.glob(sou_path+'label_liver/*.nii.gz')
    dst = data_path + 'liver_path/3d_liveronly/'
    if not os.path.exists(dst):
        os.mkdir(dst)
    for img in imgs:
        name = img.split('/')[-1]
        img_itk = sitk.ReadImage(img)
        image = sitk.GetArrayFromImage(img_itk)
        image = image.astype(np.float32)
        image = CT_normalize(image)
        itk = sitk.GetImageFromArray(image.astype(np.float32))
        itk.SetSpacing(img_itk.GetSpacing())
        sitk.WriteImage(itk, dst + name.replace('_gt', ''))
        # shutil.copy(img, dst + name.replace('_gt', ''))
    for label in labels:
        name = label.split('/')[-1]
        img_itk = sitk.ReadImage(label)
        mask = sitk.GetArrayFromImage(img_itk)
        mask[mask > 0] = 1
        itk = sitk.GetImageFromArray(mask.astype(np.uint8))
        itk.SetSpacing(img_itk.GetSpacing())
        sitk.WriteImage(itk, dst + name)


def one_shot(id_=1):
    name = dataset_names[id_ - 1] + '/'
    val = move_data(name)
    # get_sim_mask(name)
    # channel_names = generate_neigh(1, dataset_name=name)
    # channel_names = {'0000': 'CT'}
    channel_names = {'0000': 'noNorm'}
    labels = {'background': 0, 'LV': 1}
    generate_dataset_json(output_folder=data_path + 'nnUnet/raw/' + name,
                          channel_names=channel_names,
                          labels=labels,
                          num_training_cases=len(val),
                          file_ending='.nii.gz',
                          )


if __name__ == '__main__':
    ids = [0]
    for i in ids:
        print('-------dataset: ', i)
        one_shot(i)
    print('finsh')
