from math import log
from loguru import logger

import torch
from einops import repeat
from kornia.utils import create_meshgrid
from einops.einops import rearrange
import torch.nn.functional as F
from .geometry import warp_kpts, warp_kpts_refine




@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


@torch.no_grad()
def spvs_coarse(data, config):

    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['LOFTR']['RESOLUTION'][0]
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale0' in data else scale
    scalep = 8
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
  

    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)
    grid_pt1_i = scale1 * grid_pt1_c

    if 'mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])


    w_pt0_i, valid_mask0 = warp_kpts(grid_pt0_i, data['T_0to1'],H0,W0)   
    w_pt1_i, valid_mask1 = warp_kpts(grid_pt1_i, data['T_1to0'],H0,W0)
    
    w_pt0_i[~valid_mask0] = 0
    w_pt1_i[~valid_mask1] = 0
    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0

    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0


    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0 
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    arange_1 = torch.arange(h0*w0, device=device)[None].repeat(N, 1) 
    arange_0 = torch.arange(h0*w0, device=device)[None].repeat(N, 1)
    arange_1[nearest_index1 == 0] = 0
    arange_0[nearest_index0 == 0] = 0
    arange_b = torch.arange(N, device=device).unsqueeze(1)


    conf_matrix_gt = torch.zeros(N, h0*w0, h1*w1, device=device)
    conf_matrix_gt[arange_b, arange_1, nearest_index1] = 1
    conf_matrix_gt[arange_b, nearest_index0, arange_0] = 1

    conf_matrix_gt[:, 0, 0] = False

    b_ids, i_ids, j_ids = conf_matrix_gt.nonzero(as_tuple=True)


    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
    correct_0to1 = loop_back == torch.arange(h0*w0, device=device)[None].repeat(N, 1)
    correct_0to1[:, 0] = False  


    conf_matrix_gt_mnn = torch.zeros(N, h0*w0, h1*w1, device=device)
    mnn_b_ids, mnn_i_ids = torch.where(correct_0to1 != 0)
    mnn_j_ids = nearest_index1[mnn_b_ids, mnn_i_ids]

    conf_matrix_gt_mnn[mnn_b_ids, mnn_i_ids, mnn_j_ids] = 1
  

    data.update({ 'conf_matrix_gt': conf_matrix_gt,'conf_matrix_gt_mnn': conf_matrix_gt_mnn})


    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids,
    })


    data.update({
        'spv_w_pt0_i': w_pt0_i,
        'spv_pt1_i': grid_pt1_i,
        'spv_w_pt1_i': w_pt1_i,
        'spv_pt0_i': grid_pt0_i,
    })


def compute_supervision_coarse(data, config):
    assert len(set(data['dataset_name'])) == 1, "Do not support mixed datasets training!"
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth','vissar']:
        spvs_coarse(data, config)
    else:
        raise ValueError(f'Unknown data source: {data_source}')


@torch.no_grad()
def spvs_refine(data, config):

    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = 4 
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale1' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
    scale_f_c = config['LOFTR']['RESOLUTION'][0] // scale
    W_f = config['LOFTR']['TEMP_WINDOW_SIZE']

    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']

    if len(b_ids) == 0:
        data.update({"conf_matrix_f_gt": torch.zeros(1,W_f*W_f,W_f*W_f, device=device)})
        return

    grid_pt0_c = create_meshgrid(h0, w0, False, device).repeat(N, 1, 1, 1)
    grid_pt0_i = scale0* grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).repeat(N, 1, 1, 1)
    grid_pt1_i = scale1 * grid_pt1_c

    stride_f = data['hw0_t'][0] // data['hw0_c'][0]

    grid_pt0_i = rearrange(grid_pt0_i, 'n h w c -> n c h w')
    grid_pt0_i = F.unfold(grid_pt0_i, kernel_size=(W_f, W_f), stride=stride_f, padding=W_f//2)
    grid_pt0_i = rearrange(grid_pt0_i, 'n (c ww) l -> n l ww c', ww=W_f**2)
    grid_pt0_i = grid_pt0_i[b_ids, i_ids]

    grid_pt1_i = rearrange(grid_pt1_i, 'n h w c -> n c h w')
    grid_pt1_i = F.unfold(grid_pt1_i, kernel_size=(W_f, W_f), stride=stride_f, padding=W_f//2)
    grid_pt1_i = rearrange(grid_pt1_i, 'n (c ww) l -> n l ww c', ww=W_f**2)
    grid_pt1_i = grid_pt1_i[b_ids, j_ids]

    w_pt0_i = warp_kpts_refine(grid_pt0_i, data['T_0to1'])
    w_pt1_i = warp_kpts_refine(grid_pt1_i, data['T_1to0'])
    w_pt0_f = w_pt0_i / scale1
    w_pt1_f = w_pt1_i / scale0

    mkpts0_c_scaled_to_f = torch.stack(
        [i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]],
        dim=1) * scale_f_c - W_f//2
    mkpts1_c_scaled_to_f = torch.stack(
        [j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]],
        dim=1) * scale_f_c - W_f//2
    
    w_pt0_f = w_pt0_f - mkpts1_c_scaled_to_f[:,None,:]
    w_pt1_f = w_pt1_f - mkpts0_c_scaled_to_f[:,None,:]


    w_pt0_f_round = w_pt0_f[:, :, :].round().long()
    w_pt1_f_round = w_pt1_f[:, :, :].round().long()
    M = w_pt0_f.shape[0]

    nearest_index1 = w_pt0_f_round[..., 0] + w_pt0_f_round[..., 1] * W_f
    nearest_index0 = w_pt1_f_round[..., 0] + w_pt1_f_round[..., 1] * W_f

    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_f_round, W_f, W_f)] = 0
    nearest_index0[out_bound_mask(w_pt1_f_round, W_f, W_f)] = 0

    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
    correct_0to1 = loop_back == torch.arange(W_f*W_f, device=device)[None].repeat(M, 1)
    correct_0to1[:, 0] = False  

  
    conf_matrix_f_gt = torch.zeros(M, W_f*W_f, W_f*W_f, device=device)
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]
    conf_matrix_f_gt[b_ids, i_ids, j_ids] = 1

    data.update({"conf_matrix_f_gt": conf_matrix_f_gt})

@torch.no_grad()
def spvs_fine(data, config):

    w_pt0_i, pt1_i = data['spv_w_pt0_i'], data['spv_pt1_i']
    w_pt1_i, pt0_i = data['spv_w_pt1_i'], data['spv_pt0_i']
    scale = config['LOFTR']['RESOLUTION'][1]
    radius = config['LOFTR']['FINE_WINDOW_SIZE'] // 2

    b_ids, i_ids, j_ids = data['b_ids_t'], data['i_ids_t'], data['j_ids_t']

    scale = scale * data['scale1'][b_ids] if 'scale0' in data else scale
    expec_f_gt0 = (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]) / scale / radius  
    expec_f_gt1 = (w_pt1_i[b_ids, j_ids] - pt0_i[b_ids, i_ids]) / scale / radius  
    data.update({"expec_f_gt0": expec_f_gt0,"expec_f_gt1": expec_f_gt1})


def compute_supervision_fine(data, config):
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth','vissar']:
        spvs_refine(data, config)
        spvs_fine(data, config)
    else:
        raise NotImplementedError
    

