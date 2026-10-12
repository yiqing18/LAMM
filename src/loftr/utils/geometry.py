import torch
import pdb


@torch.no_grad()
def warp_kpts(kpts0, T_0to1, h, w):

    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) 
    T_0to1 = T_0to1.to(torch.float32)
    w_kpts0_cam = (T_0to1 @ kpts0_h.transpose(2, 1)).transpose(2,1)  

    w_kpts0 = w_kpts0_cam[:, :, :2] / (w_kpts0_cam[:, :, [2]] + 1e-4)  

    covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
        (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)

    return w_kpts0, covisible_mask

def warp_kpts_refine(kpts0, T_0to1):


    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) 
    T_0to1 = T_0to1.to(torch.float32)
    w_kpts0_cam = (T_0to1 @ kpts0_h.transpose(2, 1)).transpose(2,1)   
    w_kpts0 = w_kpts0_cam[:, :, :2] / (w_kpts0_cam[:, :, [2]] + 1e-4)  


    return w_kpts0
