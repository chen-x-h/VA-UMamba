import glob
import os
import pickle
import re

import SimpleITK as sitk
import dicom2nifti
import numpy as np
import scipy.io as sio
from scipy import ndimage
from skimage.measure import label
from skimage.measure import regionprops


# modify from https://github.com/lemoshu/MTCL


def _load(fp):
    pos = fp.rfind('.')
    suffix = fp[pos + 1:]
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))
    elif suffix == 'mat':
        return sio.loadmat(fp)


def save(dst, data):
    with open(dst + '.pkl', 'wb') as f:
        pickle.dump(data, f)


MIN_BOUND = 0.0
MAX_BOUND = 500.0
liver_hu = 100


def find_idx(case):
    return int(re.split('[_|.]', case.split('/')[-1])[1])


def combine_vessel_mask(mask1, mask2):
    mask = mask1 + mask2
    mask[mask > 1e-8] = 1
    return mask


def liver_ROI(mask_npy):
    # regionprops tutorial: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_regionprops.html
    labeled_img, num = label(mask_npy, return_num=True)
    print(labeled_img.shape)
    print('There are {} regions'.format(num))
    # print(np.max(labeled_img))
    if num > 0:
        regions = regionprops(labeled_img, cache=True)
        for prop in regions:
            box = prop.bbox  # Bounding box (min_row, min_col, max_row, max_col)
            area = prop.area  # Number of pixels of the region
            ratio = prop.extent  # Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols)
            print(box)
            print(area)
            print(ratio)
            # print(centroid)
            if area >= 500:
                return box


def CT_normalize(nii_data, max_=MAX_BOUND, min_=MIN_BOUND):
    """
    normalize
    Our values currently range from 0 to around 500.
    Anything above 400 is not interesting to us,
    as these are simply bones with different radiodensity.
    """
    nii_data = nii_data.astype(np.float32)

    nii_data = (nii_data - min_) / (max_ - min_)
    nii_data[nii_data > 1] = 1.
    nii_data[nii_data < 0] = 0.
    return nii_data


def crop_ROI(npy_data, box):
    xmin, xmax = box[1], box[4]
    ymin, ymax = box[2], box[5]
    zmin, zmax = box[0], box[3]
    z, x, y = npy_data.shape
    if xmin - 5 < 0:
        xmin = 5
    if xmin + 315 > x:
        xmin = x - 315
    if ymin - 5 < 0:
        ymin = 5
    if ymin + 315 > x:
        ymin = y - 315

    # crop to z x 320 x 320
    npy_data_aftercrop = npy_data[zmin:zmax, xmin - 5:xmin + 315, ymin - 5:ymin + 315]
    print('crop size:', npy_data_aftercrop.shape)
    return npy_data_aftercrop


def normalize_after_prob(nii_data):
    """
    normalize
    Our values currently range from -1024 to around 500.
    Anything above 400 is not interesting to us,
    as these are simply bones with different radiodensity.
    """
    # Default: [0, 400]
    MIN_BOUND = np.min(nii_data)
    MAX_BOUND = np.max(nii_data)

    nii_data = (nii_data - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    nii_data[nii_data > 1] = 1.
    nii_data[nii_data < 0] = 0.
    return nii_data


def choose_mask(mask_npy):
    target = [1]
    ix = np.isin(mask_npy, target)  # bool array
    # print(ix)
    idx = np.where(ix)
    idx_inv = np.where(~ix)  # inverse the bool array
    # print(idx)
    mask_npy[idx] = 1
    mask_npy[idx_inv] = 0
    return mask_npy


def dircadb_gen_1(data_path, tmp_dircadb):
    """
    dicom to nii
    :param data_path
    :param tmp_dircadb
    :return:
    """
    dircadb_path = data_path + '3Dircadb1/'
    row = 'PATIENT_DICOM/'
    ROI = 'MASKS_DICOM/liver/'
    mask1 = 'MASKS_DICOM/venoussystem/'
    mask2 = 'MASKS_DICOM/portalvein/'
    path_list = [row, ROI, mask1, mask2]

    row_path = tmp_dircadb + 'image/'
    ROI_baseDir_path = tmp_dircadb + 'label_liver/'
    vessel_msk1_baseDir_path = tmp_dircadb + 'label_venacava/'
    vessel_msk2_baseDir_path = tmp_dircadb + 'label_portalvein/'
    path_list_dis = [row_path, ROI_baseDir_path, vessel_msk1_baseDir_path, vessel_msk2_baseDir_path]
    for p in path_list_dis:
        if not os.path.exists(p):
            os.mkdir(p)

    for num in range(1, 21):
        path_ = dircadb_path + '3Dircadb1.%d/' % num
        for i in range(4):
            print(f'run {num}, {i}')
            path = path_ + path_list[i]
            if not os.path.exists(path) and i == 2:
                path = path_ + 'MASKS_DICOM/venacava/'
            if not os.path.exists(path):
                print('not found!')
                continue
            dicom2nifti.convert_directory(path, path_list_dis[i])
            os.rename(path_list_dis[i] + 'none.nii.gz', path_list_dis[i] + 'image_' + str(num) + '_gt.nii.gz')


def duel_dir_20(data_path, tmp_dircadb):
    # some error on case 20
    path = tmp_dircadb
    dicom2nifti.convert_directory(data_path + '3Dircadb1/3Dircadb1.20/LABELLED_DICOM/', path)
    vesselmsk_itk = sitk.ReadImage(path + 'none.nii.gz')
    vessel_labels = [1073, 1057, 33, 49]
    vesselmsk = sitk.GetArrayFromImage(vesselmsk_itk)
    vesselmsk[vesselmsk == 1] = 0
    for label_ind in vessel_labels:
        vesselmsk[vesselmsk == label_ind] = 1
    vesselmsk[vesselmsk != 1] = 0
    vesselmsk_itk_new = sitk.GetImageFromArray(vesselmsk)
    vesselmsk_itk_new.SetSpacing(vesselmsk_itk.GetSpacing())
    vesselmsk_itk_new.SetDirection(vesselmsk_itk.GetDirection())
    sitk.WriteImage(vesselmsk_itk_new, tmp_dircadb + 'label_venacava/image_20_gt.nii.gz')


def toRAI(image):
    image = ndimage.rotate(image, 180, axes=(1, 0), order=1, reshape=False)
    # image = ndimage.rotate(image, 180, axes=(0, 1), order=1, reshape=False)
    return image


def img_resample(itk, new_spacing=None, label_re=False):
    if new_spacing is None:
        new_spacing = [0.8, 0.8, 1.0]
    original_spacing = itk.GetSpacing()
    # print(original_spacing)
    original_size = itk.GetSize()
    # print(data.GetOrigin(), data.GetDirection())
    new_shape = [
        int(np.round(original_spacing[0] * original_size[0] / new_spacing[0])),
        int(np.round(original_spacing[1] * original_size[1] / new_spacing[1])),
        int(np.round(original_spacing[2] * original_size[2] / new_spacing[2])),
    ]
    resmaple = sitk.ResampleImageFilter()
    resmaple.SetInterpolator(sitk.sitkLinear)
    if label_re:
        resmaple.SetInterpolator(sitk.sitkNearestNeighbor)
    resmaple.SetDefaultPixelValue(0)
    resmaple.SetOutputSpacing(new_spacing)
    # resmaple.SetOutputOrigin(reference_image.GetOrigin())
    resmaple.SetOutputDirection([1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0])

    resmaple.SetSize(new_shape)
    new_itk = resmaple.Execute(itk)
    return new_itk


def dircadb_gen_2(d3_path, tmp_dircadb):
    """
    roi
    :param d3_path dst path
    :param tmp_dircadb
    :return:
    """
    msk_baseDir = tmp_dircadb + 'label_liver/'
    vessel_msk1_baseDir = tmp_dircadb + 'label_venacava/'
    vessel_msk2_baseDir = tmp_dircadb + 'label_portalvein/'
    val_img_Dir = tmp_dircadb + 'image/*.nii.gz'
    val_img_path = sorted(glob.glob(val_img_Dir))
    # val_img_path = [val_img_path[11]]
    for case in val_img_path:
        print(case)
        img_itk = sitk.ReadImage(case)
        # origin = img_itk.GetOrigin()
        img_itk = img_resample(img_itk)
        spacing = img_itk.GetSpacing()
        direction = img_itk.GetDirection()
        image = sitk.GetArrayFromImage(img_itk)
        # dir_m = np.array(direction).reshape((3, 3))
        # image = dir_m @ image
        image = toRAI(image)

        # change to mask path
        idx = find_idx(case)
        label_file_name = 'image_' + str(idx) + '_gt.nii.gz'
        msk_path = os.path.join(msk_baseDir, label_file_name)
        vessel_msk1_path = os.path.join(vessel_msk1_baseDir, label_file_name)
        vessel_msk2_path = os.path.join(vessel_msk2_baseDir, label_file_name)
        # msk_path = case.replace(".nii.gz", "_gt.nii.gz")
        if os.path.exists(msk_path):
            print(msk_path)
            msk_itk = sitk.ReadImage(msk_path)
            msk_itk = img_resample(msk_itk)
            mask_ = sitk.GetArrayFromImage(msk_itk)
            mask_ = toRAI(mask_)
            if np.max(mask_) > 1:  # fix the bug that some labels are valued 1 not 255
                mask = mask_
                mask[mask > 0] = 1
            else:
                mask = mask_

            vesselmsk1_itk = sitk.ReadImage(vessel_msk1_path)
            vesselmsk1_itk = img_resample(vesselmsk1_itk, label_re=True)
            vessel_mask1 = sitk.GetArrayFromImage(vesselmsk1_itk)
            vessel_mask1 = toRAI(vessel_mask1)
            if np.max(vessel_mask1) == 255:
                vessel_mask1 = vessel_mask1 / 255

            # vessel_mask1 = mask * vessel_mask1

            vesselmsk2_itk = sitk.ReadImage(vessel_msk2_path)
            vesselmsk2_itk = img_resample(vesselmsk2_itk)
            vessel_mask2 = sitk.GetArrayFromImage(vesselmsk2_itk, label_re=True)
            vessel_mask2 = toRAI(vessel_mask2)
            if np.max(vessel_mask2) == 255:
                vessel_mask2 = vessel_mask2 / 255
            # combine two vessel labels
            vessel_mask = combine_vessel_mask(vessel_mask1, vessel_mask2)

            # crop the liver area
            # Get the liver area
            box = liver_ROI(mask_)  # (xmin, ymin, zmin, xmax, ymax, zmax)
            # start cropping
            image = crop_ROI(image, box)
            mask = crop_ROI(mask, box)
            vessel_mask = crop_ROI(vessel_mask, box)

            if image.shape != mask.shape:
                print("Error")

            print('mask shape:', mask.shape)

            # 肝脏密度统一
            liver = mask * image
            liver_mean = liver_hu - liver.sum() / mask.sum()
            print(f'range: {liver.min()}-{liver.max()}')
            print(f'liver: {liver_mean}')
            add_ = mask * liver_mean
            image += add_.astype(np.int16)

            image = CT_normalize(image, max_=500, min_=0)

            print('image shape:', image.shape)
            image = image.astype(np.float32)

            img_itk = sitk.GetImageFromArray(image)
            img_itk.SetSpacing(spacing)
            img_itk.SetDirection([1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0])
            vessel_label_itk = sitk.GetImageFromArray(vessel_mask.astype(np.float32))
            vessel_label_itk.SetSpacing(spacing)
            vessel_label_itk.SetDirection([1.0, 0.0, 0.0,
                                           0.0, 1.0, 0.0,
                                           0.0, 0.0, 1.0])
            p_ = d3_path
            sitk.WriteImage(img_itk, p_ + 'image_{}.nii.gz'.format(str(idx)))
            sitk.WriteImage(vessel_label_itk,
                            p_ + 'image_{}_gt.nii.gz'.format(str(idx)))
            print('---------------')


if __name__ == '__main__':
    data_path = ''
    tmp_dircadb = ''
    d3_path = ''
    dircadb_gen_1(data_path=data_path, tmp_dircadb=tmp_dircadb)
    duel_dir_20(data_path=data_path, tmp_dircadb=tmp_dircadb)
    dircadb_gen_2(d3_path=d3_path, tmp_dircadb=tmp_dircadb)

    print('finsh')

