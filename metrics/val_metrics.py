import json
import os
import time

import SimpleITK as sitk
import numpy as np
from medpy import metric
from skimage.morphology import skeletonize_3d


def sen(pred, gt):
    smooth = 1e-8
    tp = pred * gt
    tp = np.sum(tp)
    fg = np.sum(gt)
    return (tp + smooth) / (fg + smooth)


def acc(pred, gt):
    right = np.sum(pred == gt)
    return right / gt.size


def metric_td(preds, gts, names, metric_fun_=None, class_num_=None):
    metrics_dict = {}
    class_num = int(np.max(gts[0]))
    if class_num_ is not None:
        class_num = class_num_

    gt_sum = [0] * (class_num - 1)
    pred_sum = [0] * (class_num - 1)
    sum_out = [0.] * (class_num - 1)
    metrics_dict['sum'] = {}
    for i in range(len(preds)):
        print(f'case_{i} ----- {names[i]}')
        case_d = {'name': names[i]}
        for j in range(1, class_num):
            gt_s = gts[i].copy()
            gt_s[gt_s != j] = 0
            gt_s[gt_s == j] = 1
            pred_s = preds[i].copy()
            pred_s[pred_s != j] = 0
            pred_s[pred_s == j] = 1
            gt_s = skeletonize_3d(gt_s.astype(np.float32)) / 255.0
            # pred_s = skeletonize_3d(pred_s.astype(np.float32)) / 255.0
            gt_s = gt_s.astype(np.uint8)
            pred_s = pred_s.astype(np.uint8)
            pred_s *= gt_s
            gt_len = int(np.sum(gt_s))
            pred_len = int(np.sum(pred_s))
            case_d[f'{j}_gt'] = gt_len
            case_d[f'{j}_pred'] = pred_len
            smooth = 1e-8
            case_d[f'{j}_td'] = (pred_len + smooth) / (gt_len + smooth)
            gt_sum[j - 1] += gt_len
            pred_sum[j - 1] += pred_len
            sum_out[j - 1] += case_d[f'{j}_td']
        # print(case_d)

        metrics_dict[f'case_{i}'] = case_d

    for i in range(1, class_num):
        # metrics_dict['sum'][f'sum_{i}'] = pred_sum[i-1] / gt_sum[i-1]
        metrics_dict['sum'][f'sum_{i}'] = sum_out[i - 1] / len(preds)

    # print(metrics_dict)
    print('td: ', metrics_dict['sum'])

    return metrics_dict


def metric_nor(preds, gts, names, metric_fun_=metric.hd95, class_num_=None):
    metrics_dict = {}
    class_num = int(np.max(gts[0]))
    if class_num_ is not None:
        class_num = class_num_
    # gt_sum = [0] * class_num
    # pred_sum = [0] * class_num
    sum_out = [0.] * (class_num - 1)
    metrics_dict['sum'] = {}
    for i in range(len(preds)):
        print(f'case_{i} ----- {names[i]}')
        case_d = {'name': names[i]}
        for j in range(1, class_num):
            # for mul class
            gt_s = gts[i].copy()
            gt_s[gt_s != j] = 0
            gt_s[gt_s == j] = 1
            pred_s = preds[i].copy()
            pred_s[pred_s != j] = 0
            pred_s[pred_s == j] = 1
            gt_s = (gt_s == 1)
            pred_s = (pred_s == 1)
            # TODO avoid div:0
            if np.sum(pred_s) == 0:
                pred_s[0][1] = 1
            if np.sum(gt_s) == 0:
                gt_s[0][0] = 1
                print(f'case_{i} is 0')

            met = metric_fun_(pred_s, gt_s)
            case_d[f'{j}_{metric_fun_.__name__}'] = met
            sum_out[j - 1] += met

        # print(case_d)

        metrics_dict[f'case_{i}'] = case_d

    for i in range(1, class_num):
        # metrics_dict['sum'][f'sum_{i}'] = pred_sum[i-1] / gt_sum[i-1]
        metrics_dict['sum'][f'sum_{i}'] = sum_out[i - 1] / len(preds)
    print(f'{metric_fun_.__name__}', metrics_dict['sum'])

    # print(metrics_dict)

    return metrics_dict


def val_frame(data_id, val_fold, metric=None, metric_fun_=metric.hd95, class_num_=2, gt_path=''):
    val_cases = [d for d in os.listdir(val_fold)
                 if d not in ['summary.json'] and not os.path.isdir(os.path.join(val_fold, d))]
    preds = []
    gts = []
    for case in val_cases:
        pred = sitk.ReadImage(os.path.join(val_fold, case))
        pred = sitk.GetArrayFromImage(pred)
        gt = sitk.ReadImage(os.path.join(gt_path, case))
        gt = sitk.GetArrayFromImage(gt)
        preds.append(pred)
        gts.append(gt)

    metrics = {}
    if metric is not None:
        metrics = metric(preds, gts, val_cases, metric_fun_=metric_fun_, class_num_=class_num_)
    return metrics


def do_my_metric(name, data_id=None, class_num=2, only_td=False, not_nn_path=None):
    """
    You can do this in nnUnet dataset
    :param name:
    :param data_id:
    :param class_num:
    :param only_td:
    :param not_nn_path:
    :return:
    """
    val_fold = f'/home/chenxianhao/Liver/dataset/nnUnet/results/{name}/fold_0/validation'
    save_fold = f'/home/chenxianhao/Liver/dataset/nnUnet/results/{name}/fold_0/my_metric'
    if not_nn_path is not None:
        val_fold = not_nn_path + '/validation'
        save_fold = not_nn_path + '/my_metric'
    if not os.path.exists(save_fold):
        os.mkdir(save_fold)
    # gt_id = 1
    gt_id = int(name.split('_')[1])
    if data_id is not None:
        gt_id = data_id

    time_ = time.strftime('%Y-%m-%d-%H%M%S', time.localtime())

    if only_td:
        metric_fun = metric_td
        metrics = val_frame(gt_id, val_fold, metric=metric_fun, class_num_=class_num)
        # anl_sum[f'{metric_fun.__name__}'] = metrics['sum']
        with open(save_fold + f"/0_new_metrics_{metric_fun.__name__}_{time_}.json", 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        return

    anl_sum = {}

    metric_fun = metric.binary.dc
    metrics = val_frame(gt_id, val_fold, metric=metric_nor, class_num_=class_num, metric_fun_=metric_fun)
    anl_sum[f'{metric_fun.__name__}'] = metrics['sum']
    with open(save_fold + f"/metrics_{metric_fun.__name__}_{time_}.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    metric_fun = acc
    metrics = val_frame(gt_id, val_fold, metric=metric_nor, class_num_=class_num, metric_fun_=metric_fun)
    anl_sum[f'{metric_fun.__name__}'] = metrics['sum']
    with open(save_fold + f"/metrics_{metric_fun.__name__}_{time_}.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    metric_fun = sen
    metrics = val_frame(gt_id, val_fold, metric=metric_nor, class_num_=class_num, metric_fun_=metric_fun)
    anl_sum[f'{metric_fun.__name__}'] = metrics['sum']
    with open(save_fold + f"/metrics_{metric_fun.__name__}_{time_}.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    metric_fun = metric_td
    metrics = val_frame(gt_id, val_fold, metric=metric_fun, class_num_=class_num)
    anl_sum[f'{metric_fun.__name__}'] = metrics['sum']
    with open(save_fold + f"/metrics_{metric_fun.__name__}_{time_}.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    metric_fun = metric.binary.hd95
    metrics = val_frame(gt_id, val_fold, metric=metric_nor, class_num_=class_num, metric_fun_=metric_fun)
    anl_sum[f'{metric_fun.__name__}'] = metrics['sum']
    with open(save_fold + f"/metrics_{metric_fun.__name__}_{time_}.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    with open(save_fold + f"/anl_sum_{time_}.json", 'w', encoding='utf-8') as f:
        json.dump(anl_sum, f, indent=4, ensure_ascii=False)

    print(anl_sum)
    return anl_sum


if __name__ == '__main__':

    print('end')
