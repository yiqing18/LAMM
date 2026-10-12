import torch
import cv2
import numpy as np
from collections import OrderedDict
from loguru import logger
from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous
import pdb


# --- METRICS ---

def compute_precision(data):
    thre = 5
    T_name = data['T_name']
    pts0 = data['mkpts0_f'] 
    pts1 = data['mkpts1_f']
    number = pts0.shape[0]

    H = data['T_0to1'] 

    Y = H.squeeze(0).float() @ torch.hstack((pts0,torch.ones(number,1).cuda())).T  
    Y_final = Y[0:2,:] / Y[2,:]
    pre_errs = torch.sqrt(sum(torch.pow((Y_final - pts1.T), 2)))

    data.update({'Pre': pre_errs})



# --- METRIC AGGREGATION ---

def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs
    
def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    prec_ = []
    for errs in errors:
        correct_mask = errs < thresholds[0]
        prec_.append(np.mean(correct_mask))

    precs.append(np.mean(prec_))

    if ret_dict:
        return {f'Pre@{t}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs

def aggregate_metrics(metrics, epi_err_thr=5e-4):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(np.array(metrics['Pre'], dtype=object)[unq_ids], dist_thresholds, True)

    return precs
