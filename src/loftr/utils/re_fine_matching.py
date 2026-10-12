import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
import pdb
import numpy as np
from kornia.geometry.subpix import dsnt

INF = 1e9


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch
    
    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand

class ReFineMatching(nn.Module):
    def __init__(self):
        super().__init__()
        self.thr = 0.05
        self.temperature = 0.1
        self.train_coarse_percent = 0.2
        self.train_pad_num_gt_min = 200


    def forward(self, feat_l0, feat_l1, data, mask_c0=None, mask_c1=None):

        # N, ww, C = feat_l0.shape
        feat_l0, feat_l1 = map(lambda feat: feat / feat.shape[-1]**.5,
                               [feat_l0, feat_l1])

        sim_matrix = torch.einsum("ndc,nsc->nds", feat_l0, feat_l1)/ self.temperature
        conf_matrix = F.softmax(sim_matrix,1) * F.softmax(sim_matrix,2)
        data.update({'conf_matrix_fine': conf_matrix})
        
        data.update(**self.get_coarse_match(conf_matrix, data))

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data):

        _device = conf_matrix.device
        c_b_ids, c_i_ids, c_j_ids = data['b_ids'], data['i_ids'], data['j_ids']
  
        mask = conf_matrix > self.thr

        if mask.sum() == 0:
            mask[0,0,0] = 1
            conf_matrix[0,0,0] = 1
        mask = mask * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])\
                    * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0])

        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        mconf_t = data['t_conf'][b_ids]
        mask_t = data['mask']

        n_b_ids, n_i_ids, n_j_ids = c_b_ids[b_ids], c_i_ids[b_ids], c_j_ids[b_ids]

        if self.training:

            if 'mask0' not in data:
                num_candidates_max = mask_t.size(0) * max(
                    mask_t.size(1), mask_t.size(2))
            else:
                num_candidates_max = compute_max_candidates(
                    data['mask0'], data['mask1'])
            num_matches_train = int(num_candidates_max *
                                    self.train_coarse_percent)
            num_matches_pred = len(n_b_ids)
            assert self.train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - self.train_pad_num_gt_min, ),
                    device=_device)

            gt_pad_indices = torch.randint(
                    len(data['spv_b_ids']),
                    (max(num_matches_train - num_matches_pred,
                        self.train_pad_num_gt_min), ),
                    device=_device)
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)  

            n_b_ids, n_i_ids, n_j_ids, mconf_t = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]],
                                    dim=0),
                *zip([n_b_ids, data['spv_b_ids']], [n_i_ids, data['spv_i_ids']],
                    [n_j_ids, data['spv_j_ids']], [mconf_t, mconf_gt]))     


        coarse_matches = {'b_ids_t': n_b_ids, 'i_ids_t': n_i_ids, 'j_ids_t': n_j_ids}

        scale = data['hw0_i'][0] / data['hw0_c'][0]
        mkpts0_c = torch.stack(
            [n_i_ids % data['hw0_c'][1], n_i_ids // data['hw0_c'][1]],
            dim=1) * scale
        mkpts1_c = torch.stack(
            [n_j_ids % data['hw1_c'][1], n_j_ids // data['hw1_c'][1]],
            dim=1) * scale

        try:
            coarse_matches.update({
                'm_bids': n_b_ids[mconf_t != 0],  
                'mkpts0_c': mkpts0_c[mconf_t != 0],
                'mkpts1_c': mkpts1_c[mconf_t != 0],
                'mconf': mconf_t[mconf_t != 0],
            })
        except:
            n_b_ids_cpu = n_b_ids.cpu()
            mconf_t_cpu = mconf_t.cpu()

        return coarse_matches
