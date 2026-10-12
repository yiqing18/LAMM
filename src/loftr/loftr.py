import torch
import torch.nn as nn
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformerCS, LocalFeatureTransformer,FinePreprocess,TempPreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching_dual import FineMatching
from .utils.re_fine_matching import ReFineMatching
from src.utils.plotting import *

class LoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = build_backbone(config,False)
        self.pos_encoding = PositionEncodingSine(  
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])    
        self.loftr_coarse = LocalFeatureTransformerCS(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])

        self.temp_preprocess = TempPreprocess(config)
        self.loftr_temp = LocalFeatureTransformer(config['temp'])
        self.temp_matching = ReFineMatching()
        
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()

    def forward(self, data):

        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']: 
            feats_c, feats_t, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0), data)
            (feat_c0, feat_c1), (feat_f0, feat_f1), (feat_t0, feat_t1) = feats_c.split(data['bs']), feats_f.split(data['bs']), feats_t.split(data['bs'])
        else: 
            (feat_c0, feat_f0, feat_t0), (feat_c1, feat_f1, feat_t1) = self.backbone(data['image0'], data), self.backbone(data['image1'], data)

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:],
            'hw0_t': feat_t0.shape[2:], 'hw1_t': feat_t1.shape[2:],
        })


        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
 
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')
        
        mask_c0 = mask_c1 = None  
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        feat_c0,feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)
       
        if data['repeat_index_']==True: 
            feat_t0_unfold, feat_t1_unfold = self.temp_preprocess(feat_t0, feat_t1, feat_c0, feat_c1, data) 

            if feat_t0_unfold.size(0) != 0:
                feat_t0_unfold, feat_t1_unfold = self.loftr_temp(feat_t0_unfold, feat_t1_unfold)
                self.temp_matching(feat_t0_unfold, feat_t1_unfold, data)

        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        
        if feat_f0_unfold.size(0) != 0:  
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)


    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
