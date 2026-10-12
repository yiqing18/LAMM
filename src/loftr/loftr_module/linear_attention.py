"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import torch
from torch.nn import Module, Dropout
import torch.nn.functional as F
import torch.nn as nn
import pdb
from einops.einops import repeat

class layernorm2d(nn.Module):
     
     def __init__(self,dim) :
         super().__init__()
         self.dim=dim
         self.affine=nn.parameter.Parameter(torch.ones(dim), requires_grad=True)
         self.bias=nn.parameter.Parameter(torch.zeros(dim), requires_grad=True) 
    
     def forward(self,x):
        mean,std=x.mean(dim=1,keepdim=True),x.std(dim=1,keepdim=True)
        return self.affine[None,:,None,None]*(x-mean)/(std+1e-6)+self.bias[None,:,None,None]

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):

        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()

class RefineAttention(Module):
    def __init__(self, d_model):
        super().__init__()
        self.feature_map = elu_feature_map
        self.d_model = d_model

    def forward(self, query, key, value, temp=1, mask_sample=None):

        bs=query.shape[0]
        QK = torch.einsum("bhdkm,bhdgm->bhkgm", query, key)

        if mask_sample is not None:
            num_head,number_n=QK.shape[1],QK.shape[3]
            QK.masked_fill_(~(mask_sample[:,:,:,None]).expand(-1,num_head,-1,number_n,-1).bool(), float(-1e8))

        softmax_temp = temp / query.size(2)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=-1)

        queried_values = torch.einsum("bhkgm,bhdgm->bhdkm", A, value).contiguous().view(bs,self.d_model,-1)
 
        return queried_values


class FullAttention(Module):
    def __init__(self,d_model,nhead):
        super().__init__()
        self.d_model=d_model
        self.nhead=nhead

    def forward(self, q, k,v , mask0=None, mask1=None, temp=1):

        bs=q.shape[0]
        q,k,v=q.view(bs,self.nhead,self.d_model//self.nhead,-1),k.view(bs,self.nhead,self.d_model//self.nhead,-1),v.view(bs,self.nhead,self.d_model//self.nhead,-1)

        QK = torch.einsum("nhdl,nhds->nhls", q, k)
        if mask0 is not None:
            QK.masked_fill_(~(mask0[:,None, :, None] * mask1[:, None, None]).bool(), float(-1e8))

        softmax_temp = temp / q.size(2)**.5  
        A = torch.softmax(softmax_temp * QK, dim=-1)
        queried_values = torch.einsum("nhls,nhds->nhdl", A, v).contiguous().view(bs,self.d_model,-1)
        return queried_values
    
class HierachicalAttention(Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.fullattention = FullAttention(d_model, nhead)
        self.merge_head = nn.Sequential(
            nn.Conv1d(d_model*3, d_model, kernel_size=1,bias=False),
            nn.ReLU(True),
            nn.Conv1d(d_model, d_model, kernel_size=1,bias=False),
        )
        self.linearattetion = LinearAttention()
        self.nlevel = 3
        self.temp=nn.parameter.Parameter(torch.tensor(1.),requires_grad=True)

    def forward(self, query, key, value,mask0, mask1):
        bs = query.shape[0]
        ds0=[4,4]
        h0=60
        w0=80
        message_list = []
        query, key, value = query.permute(0,2,1).contiguous(),key.permute(0,2,1).contiguous(),value.permute(0,2,1).contiguous()

        sub_sample0,sub_sample1=[ds0,2,1],[ds0,2,1]  # [[4,4],2,1]
        q_list=[F.avg_pool2d(query.view(bs,-1,h0,w0),kernel_size=sub_size,stride=sub_size) for sub_size in sub_sample0] 
        k_list=[F.avg_pool2d(key.view(bs,-1,h0,w0),kernel_size=sub_size,stride=sub_size) for sub_size in sub_sample1]
        v_list=[F.avg_pool2d(value.view(bs,-1,h0,w0),kernel_size=sub_size,stride=sub_size) for sub_size in sub_sample1] 
        

        if mask0 is not None:
            mask0,mask1=mask0.view(bs,1,h0,w0),mask1.view(bs,1,h0,w0)
            mask0_list=[-F.max_pool2d(-mask0,kernel_size=sub_size,stride=sub_size) for sub_size in sub_sample0]
            mask1_list=[-F.max_pool2d(-mask1,kernel_size=sub_size,stride=sub_size) for sub_size in sub_sample1]

        else:
            mask0_list=mask1_list=[None,None,None]
        sample = [4,2,1]
        message_list.append(self.fullattention(q_list[0],k_list[0],v_list[0],mask0_list[0], mask1_list[0],self.temp).view(bs,self.d_model,h0//sample[0],w0//sample[0]))
        
        for index in range(1,self.nlevel):
            q,k,v,q_mask, kv_mask=q_list[index],k_list[index],v_list[index],mask0_list[index], mask1_list[index]

            message_list.append(self.linearattetion(q,k,v,q_mask, kv_mask).view(bs,self.d_model,h0//sample[index],w0//sample[index]))

        all_message=torch.cat([F.upsample(message_list[idx],scale_factor=sub_sample0[idx],mode='nearest') \
                    for idx in range(self.nlevel)],dim=1).view(bs,-1,h0*w0) 
        
        all_message=self.merge_head(all_message).view(bs,-1,h0,w0) 
        return all_message
