import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat
import numpy
import pdb


class TempPreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cat_c_feat = config['fine_concat_coarse_feat']
        self.W = self.config['temp_window_size']

        d_model_c = self.config['coarse']['d_model']
        d_model_t = self.config['temp']['d_model']
        self.d_model_t = d_model_t
        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_t, bias=True)
            self.merge_feat = nn.Linear(2*d_model_t, d_model_t, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_t0, feat_t1, feat_c0, feat_c1, data):

        W = self.W
        stride = data['hw0_t'][0] // data['hw0_c'][0]

        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, self.W**2, self.d_model_t, device=feat_t0.device)
            feat1 = torch.empty(0, self.W**2, self.d_model_t, device=feat_t0.device)
            return feat0, feat1

        unique_tensor0, inverse_indices0, counts0 = torch.unique(data['i_ids'], dim=0, sorted=False, return_inverse=True, return_counts=True)

  
        duplicate_values0 = unique_tensor0[counts0 > 1]
        t=0
  
        if duplicate_values0.shape[0] > 1:
            duplicate_indices0 = [torch.nonzero(inverse_indices0 == idx).squeeze().tolist() for idx in torch.nonzero(counts0 > 1).squeeze()]
            tensor_list0 = torch.cat([torch.tensor(sublist) for sublist in duplicate_indices0])
        else:
            tensor_list0 = None
            t=1


        unique_tensor1, inverse_indices1, counts1 = torch.unique(data['j_ids'], dim=0, sorted=False, return_inverse=True, return_counts=True)

        duplicate_values1 = unique_tensor1[counts1 > 1]
        if duplicate_values1.shape[0] > 1:
            duplicate_indices1 = [torch.nonzero(inverse_indices1 == idx).squeeze().tolist() for idx in torch.nonzero(counts1 > 1).squeeze()]
            tensor_list1 = torch.cat([torch.tensor(sublist) for sublist in duplicate_indices1])
        else:
            tensor_list1 = None
            t=t+2
   
        if t==0:
            tensor_list = torch.cat((tensor_list0, tensor_list1))
            un_tensor_list = torch.unique(tensor_list).cuda()
        elif t==1:
            un_tensor_list = tensor_list1
        elif t==2:
            un_tensor_list = tensor_list0
        else:
            un_tensor_list=None
        feat_t0_unfold = F.unfold(feat_t0, kernel_size=(W, W), stride=stride, padding=W//2)  
        feat_t0_unfold = rearrange(feat_t0_unfold, 'n (c ww) l -> n l ww c', ww=W**2)   
        feat_t1_unfold = F.unfold(feat_t1, kernel_size=(W, W), stride=stride, padding=W//2)
        feat_t1_unfold = rearrange(feat_t1_unfold, 'n (c ww) l -> n l ww c', ww=W**2)

        feat_t0_unfold = feat_t0_unfold[data['b_ids'], data['i_ids']]  

        feat_t1_unfold = feat_t1_unfold[data['b_ids'], data['j_ids']]
        feat_c1_win = feat_c1[data['b_ids'], data['j_ids']]

        if self.cat_c_feat:
            feat_c_win = self.down_proj(torch.cat([feat_c0[data['b_ids'], data['i_ids']],
                                                   feat_c1_win], 0))  
            feat_ct_win = self.merge_feat(torch.cat([
                torch.cat([feat_t0_unfold, feat_t1_unfold], 0),  
                repeat(feat_c_win, 'n c -> n ww c', ww=W**2), 
            ], -1))
            feat_t0_unfold, feat_t1_unfold = torch.chunk(feat_ct_win, 2, dim=0)  
        data.update({
            'repeat_index':un_tensor_list,
            't_list0':tensor_list0,
            't_list1':tensor_list1,
        })

        return feat_t0_unfold, feat_t1_unfold
