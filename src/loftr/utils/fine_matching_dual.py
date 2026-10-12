import math
import torch
import torch.nn as nn

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid


class FineMatching(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, feat_f0, feat_f1, data):

        M, WW, C = feat_f0.shape
        W = int(math.sqrt(WW))
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        if 'mkpts1_rc' not in data:
            data.update({
                'mkpts1_rc':data['mkpts1_c']
            })

        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"

            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
                'sig': 0,
            })
            return

        unique_tensor0, inverse_indices0, counts0 = torch.unique(data['mkpts0_c'], dim=0, sorted=False, return_inverse=True, return_counts=True)

        duplicate_values0 = unique_tensor0[counts0 > 1]
        
        if duplicate_values0.shape[0] > 1:
            duplicate_indices0 = [torch.nonzero(inverse_indices0 == idx).squeeze().tolist() for idx in torch.nonzero(counts0 > 1).squeeze()]
            tensor_list0 = torch.cat([torch.tensor(sublist) for sublist in duplicate_indices0])
        else:
            tensor_list0 = None

        feat_f1_picked = feat_f1_picked = feat_f1[:, WW//2, :]
        sim_matrix1 = torch.einsum('mc,mrc->mr', feat_f1_picked, feat_f0)
        softmax_temp = 1. / C**.5
        heatmap1 = torch.softmax(softmax_temp * sim_matrix1, dim=1).view(-1, W, W)

        coords_normalized1 = dsnt.spatial_expectation2d(heatmap1[None], True)[0]  
        grid_normalized = create_meshgrid(W, W, True, heatmap1.device).reshape(1, -1, 2)  

        var1 = torch.sum(grid_normalized**2 * heatmap1.view(-1, WW, 1), dim=1) - coords_normalized1**2  
        std1 = torch.sum(torch.sqrt(torch.clamp(var1, min=1e-10)), -1)  
    
        feat_f0_picked = feat_f0_picked = feat_f0[:, WW//2, :]
        sim_matrix0 = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
        heatmap0 = torch.softmax(softmax_temp * sim_matrix0, dim=1).view(-1, W, W)

        coords_normalized0 = dsnt.spatial_expectation2d(heatmap0[None], True)[0]  

        var0 = torch.sum(grid_normalized**2 * heatmap0.view(-1, WW, 1), dim=1) - coords_normalized0**2  
        std0 = torch.sum(torch.sqrt(torch.clamp(var0, min=1e-10)), -1)  

        data.update({'expec_f0': torch.cat([coords_normalized0, std0.unsqueeze(1)], -1),
                     'expec_f1': torch.cat([coords_normalized1, std1.unsqueeze(1)], -1)})

        self.get_fine_match(coords_normalized0, coords_normalized1, tensor_list0, data)

    @torch.no_grad()
    def get_fine_match(self, coords_normed0, coords_normed1, list0, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale
        scale1 = scale
        mkpts0 = data['mkpts0_c']
        mkpts1 = data['mkpts1_c']

        mkpts0_f = mkpts0
        mkpts1_f = mkpts1 + (coords_normed0 * (W // 2) * scale1)[:len(data['mconf'])]           

        mkpts0_f_t = mkpts0 + (coords_normed1 * (W // 2) * scale1)[:len(data['mconf'])]
        if list0 is not None:
            mkpts0_f[list0] = mkpts0_f_t[list0]
            mkpts1_f[list0] = mkpts1[list0]
        
        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f,
        })
