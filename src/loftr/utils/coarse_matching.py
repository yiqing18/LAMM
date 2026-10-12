import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
import pdb

INF = 1e9

def mask_border(m, b: int, v):

    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):

    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand

def torch_isin(elements, test_elements):
    elements = elements.view(-1, 1)  
    test_elements = test_elements.view(1, -1)
    return torch.any(torch.eq(elements, test_elements), dim=1)


class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.thr = config['thr']
        self.border_rm = config['border_rm']

        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']

        self.match_type = config['match_type']
        if self.match_type == 'dual_softmax':
            self.temperature = config['dsmax_temperature']
        elif self.match_type == 'sinkhorn':
            try:
                from .superglue import log_optimal_transport
            except ImportError:
                raise ImportError("download superglue.py first!")
            self.log_optimal_transport = log_optimal_transport
            self.bin_score = nn.Parameter(
                torch.tensor(config['skh_init_bin_score'], requires_grad=True))
            self.skh_iters = config['skh_iters']
            self.skh_prefilter = config['skh_prefilter']
        else:
            raise NotImplementedError()

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
                               [feat_c0, feat_c1])
        
        if self.match_type == 'dual_softmax':
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                      feat_c1) / self.temperature
            if mask_c0 is not None:
                sim_matrix.masked_fill_(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                    -INF)
            
            conf_matrix0 = F.softmax(sim_matrix,1)
            conf_matrix1 = F.softmax(sim_matrix,2)
            conf_matrix = conf_matrix0 * conf_matrix1

        data.update({'conf_matrix0': conf_matrix0,
                     'conf_matrix1': conf_matrix1,
                     'conf_matrix': conf_matrix})

        data.update(**self.get_coarse_match(conf_matrix0, conf_matrix1, conf_matrix, data))

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix0, conf_matrix1, conf_matrix, data):
    
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        _device = conf_matrix0.device
        
        mask_m = conf_matrix > self.thr
        mask_m = rearrange(mask_m, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                         **axes_lengths)
        if 'mask0' not in data:
            mask_border(mask_m, self.border_rm, False)
        else:
            mask_border_with_padding(mask_m, self.border_rm, False,
                                     data['mask0'], data['mask1'])
        mask_m = rearrange(mask_m, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                         **axes_lengths)

        # 2. mutual nearest
        mask_m = mask_m \
            * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
            * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        b_ids_m, i_ids_m, j_ids_m = mask_m.nonzero(as_tuple=True)

        b_ids, i_ids, j_ids = b_ids_m, i_ids_m, j_ids_m
        mnn_mconf = conf_matrix[b_ids, i_ids, j_ids]
            

        un_tensor_list = False
        mask = torch.logical_or((conf_matrix1 > self.thr) * (conf_matrix1 == conf_matrix1.max(dim=2, keepdim=True)[0]),
                            (conf_matrix0 > self.thr) * (conf_matrix0 == conf_matrix0.max(dim=1, keepdim=True)[0]))

        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                        **axes_lengths)
        
        if 'mask0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False,
                                    data['mask0'], data['mask1'])
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                        **axes_lengths)

        b_ids, i_ids, j_ids = mask.nonzero(as_tuple=True)
        mconf = torch.maximum(conf_matrix1[b_ids, i_ids, j_ids], conf_matrix0[b_ids, i_ids, j_ids])
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids, 't_conf': mconf,'mask':mask,'mnn_conf':mnn_mconf}

        unique_tensor0, inverse_indices0, counts0 = torch.unique(i_ids, dim=0, sorted=False, return_inverse=True, return_counts=True)
        unique_tensor1, inverse_indices1, counts1 = torch.unique(j_ids, dim=0, sorted=False, return_inverse=True, return_counts=True)

   
        duplicate_values0 = unique_tensor0[counts0 > 1]
  
        if duplicate_values0.shape[0] > 1:
            un_tensor_list = True

        duplicate_values1 = unique_tensor1[counts1 > 1]
        if duplicate_values1.shape[0] > 1:
            un_tensor_list = True
        
        if self.training:
 
            if 'mask0' not in data:
                num_candidates_max = mask.size(0) * max(
                    mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(
                    data['mask0'], data['mask1'])
            num_matches_train = int(num_candidates_max *
                                    self.train_coarse_percent)  
            num_matches_pred = len(b_ids)
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


            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]],
                                       dim=0),
                *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],
                     [j_ids, data['spv_j_ids']], [mconf, mconf_gt])) 

        data.update({
            'repeat_index_':un_tensor_list,
        })

        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]],
            dim=1) * scale1

        coarse_matches.update({
            'm_bids': b_ids[mconf != 0],  
            'b_ids_t':b_ids,
            'i_ids_t':i_ids,
            'j_ids_t':j_ids,
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mconf': mconf[mconf != 0],
        })
        return coarse_matches
