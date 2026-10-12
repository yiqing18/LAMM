from loguru import logger

import torch
import torch.nn as nn
import pdb
from skimage.metrics import structural_similarity as ssim
from itertools import permutations


class LoFTRLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['loftr']['loss']
        self.match_type = self.config['loftr']['match_coarse']['match_type']
        self.sparse_spvs = self.config['loftr']['match_coarse']['sparse_spvs']  # True
        
        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr'] # 1
        self.c_pos_w = self.loss_config['pos_weight'] # 1
        self.c_neg_w = self.loss_config['neg_weight'] # 1
        # fine-level
        self.fine_type = self.loss_config['fine_type']

    def compute_coarse_dual_loss(self, data, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        # pdb.set_trace()
        conf1 = data["conf_matrix1"]
        conf0 = data["conf_matrix0"]
        conf_gt0 = data["conf_matrix_gt"]
        # conf_gt1 = data["conf_matrix_gt1"]
        pos_mask0, neg_mask0 = conf_gt0 == 1, conf_gt0 == 0
        # pos_mask1, neg_mask1 = conf_gt1 == 1, conf_gt1 == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask0.any():  # assign a wrong gt
            pos_mask0[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.


        # elif self.loss_config['coarse_type'] == 'focal':
        conf0 = torch.clamp(conf0, 1e-6, 1-1e-6)   # This
        conf1 = torch.clamp(conf1, 1e-6, 1-1e-6)
        alpha = self.loss_config['focal_alpha']  # 0.25
        gamma = self.loss_config['focal_gamma']  # 2
        
        # if self.sparse_spvs:
        pos_conf0 = conf0[pos_mask0]
        pos_conf1 = conf1[pos_mask0]
        # pdb.set_trace()
        loss_pos0 = - alpha * torch.pow(1 - pos_conf0, gamma) * pos_conf0.log()
        # calculate losses for negative samples
        loss_pos0 += - alpha * torch.pow(1 - pos_conf1, gamma) * pos_conf1.log()

        if weight is not None:
            # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
            # but only through manually setting corresponding regions in sim_matrix to '-inf'.
            loss_pos0 = loss_pos0 * weight[pos_mask0]    
            # loss_pos1 = loss_pos1 * weight[pos_mask1] 
        loss =  c_pos_w * loss_pos0.mean() 
        return loss
    
    def compute_coarse_loss(self, data, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        # pdb.set_trace()
        conf = data["conf_matrix"]
        conf_gt = data["conf_matrix_gt_mnn"]
        pos_mask0, neg_mask0 = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask0.any():  # assign a wrong gt
            pos_mask0[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask0.any():
            neg_mask0[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        # elif self.loss_config['coarse_type'] == 'focal':
        conf = torch.clamp(conf, 1e-6, 1-1e-6)   # This
        alpha = self.loss_config['focal_alpha']  # 0.25
        gamma = self.loss_config['focal_gamma']  # 2
        
        # if self.sparse_spvs:
        pos_conf = conf[pos_mask0]
        # pdb.set_trace()
        loss_pos0 = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
        # calculate losses for negative samples


        if weight is not None:
            # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
            # but only through manually setting corresponding regions in sim_matrix to '-inf'.
            loss_pos0 = loss_pos0 * weight[pos_mask0]     
        loss =  c_pos_w * loss_pos0.mean()
        return loss
        
    def compute_fine_loss(self, expec_f, expec_f_gt):
        if self.fine_type == 'l2_with_std':  # This
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.fine_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr #torch.linalg.norm矩阵的行绝对值之和的最大值

        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1. / torch.clamp(std, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if not correct_mask.any():
            if self.training:  # this seldomly happen during training, since we pad prediction with gt
                               # sometimes there is not coarse-level gt at all.
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
                weight[0] = 0.
            else:
                return torch.tensor(0)

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss
    
    def match_and_compute_distance(self, j_pred_group, j_gt_group,w):
        """匹配最近的GT和Pred值,并计算最小距离"""
        # 如果一方只有1个值，直接复制对齐
        if len(j_pred_group) == 1 or len(j_gt_group) == 1:
            j_pred_padded = j_pred_group.expand(max(len(j_pred_group), len(j_gt_group)))
            j_gt_padded = j_gt_group.expand(max(len(j_pred_group), len(j_gt_group)))
            return torch.norm(j_pred_padded.float() - j_gt_padded.float(), p=2)
        
        # 如果双方都多于1个值但数量不同，复制较少一方的第一个值填充
        if len(j_pred_group) != len(j_gt_group):
            target_len = max(len(j_pred_group), len(j_gt_group))
            if len(j_pred_group) < target_len:
                j_pred_group = torch.cat([j_pred_group, j_pred_group[0].expand(target_len - len(j_pred_group))])
            else:
                j_gt_group = torch.cat([j_gt_group, j_gt_group[0].expand(target_len - len(j_gt_group))])
        
        j_gt_=torch.stack([j_gt_group%w,j_gt_group//w],dim=1)  # 序列变二维  [4354,2]
        j_pred_=torch.stack([j_pred_group%w,j_pred_group//w],dim=1)
        # pdb.set_trace()
        distance=torch.sqrt(((j_gt_-j_pred_)**2)).float().mean()
        
        # # 计算所有可能的匹配组合，选择总距离最小的方式
        # min_distance = float('inf')
        # for pred_perm in permutations(j_pred_group):
        #     for gt_perm in permutations(j_gt_group):
        #         current_distance = torch.norm(torch.stack(pred_perm).float() - torch.stack(gt_perm).float(), p=2)
        #         if current_distance < min_distance:
        #             min_distance = current_distance
        return distance
    
    def get_distance(self, i_pred, i_gt, j_pred, j_gt, w):
        # 找到 i_pred 和 i_gt 中的公共序号
        common_indices = torch.tensor(list(set(i_pred.tolist()) & set(i_gt.tolist()))).cuda()

                # 初始化存储每组距离的列表
        distances = []

        # 对每个公共序号处理
        for idx in common_indices:
            # 获取当前序号在 i_pred 和 i_gt 中的位置
            pred_pos = (i_pred == idx).nonzero().squeeze()
            gt_pos = (i_gt == idx).nonzero().squeeze()
            
            # 获取对应的 j_pred 和 j_gt 值
            j_pred_group = j_pred[pred_pos] if pred_pos.dim() > 0 else j_pred[pred_pos].unsqueeze(0)
            j_gt_group = j_gt[gt_pos] if gt_pos.dim() > 0 else j_gt[gt_pos].unsqueeze(0)
            
            # 填充较短的一方（用 0）
            max_len = max(len(j_pred_group), len(j_gt_group))
            if len(j_pred_group) < max_len:
                padding = torch.zeros(max_len - len(j_pred_group), dtype=j_pred_group.dtype).cuda()
                j_pred_group = torch.cat([j_pred_group, padding])
            if len(j_gt_group) < max_len:
                padding = torch.zeros(max_len - len(j_gt_group), dtype=j_gt_group.dtype).cuda()
                j_gt_group = torch.cat([j_gt_group, padding])
            
            # 计算当前组的欧式距离
            distance = self.match_and_compute_distance(j_pred_group, j_gt_group,w)
            distances.append(distance)

        # 合并所有组的距离（求和或平均）
        if not distances:
            total_distance = torch.tensor(0.0)
        else:
            total_distance = torch.sum(torch.stack(distances))
        return total_distance

    
    def compute_refine_loss(self,coarse_corr_gt,i_pred, j_pred,w,h):
        #coarse_corr_gt:[[batch_indices],[left_indices],[right_indices]]
        #flow_list: [L,B,H,W,4] w1 80
        j_gt = coarse_corr_gt[2]
        i_gt = coarse_corr_gt[1]

        refine_loss_j = self.get_distance(i_pred, i_gt, j_pred, j_gt,w)
        refine_loss_i = self.get_distance(j_pred, j_gt, i_pred, i_gt,w)
        # # 计算欧式距离（忽略 NaN 值）
        # # valid_mask = ~torch.isnan(j_pred_values) & ~torch.isnan(j_gt_values)
        # j_gt_=torch.stack([j_gt_values%w,j_gt_values//w],dim=1)  # 序列变二维  [4354,2]
        # j_pred_=torch.stack([j_pred_values%w,j_pred_values//w],dim=1)
        # # pdb.set_trace()
        # refine_loss_j=torch.sqrt(((j_gt_-j_pred_)**2)).float().mean()

        # 找到 i_pred 和 i_gt 中的公共序号
        # common_indices_j = torch.tensor(list(set(j_pred.tolist()) & set(j_gt.tolist()))).cuda()

        # # 对每个公共序号处理
        # for idx in common_indices_j:
        #     # 获取当前序号在 i_pred 和 i_gt 中的位置
        #     pred_pos = (i_pred == idx).nonzero().squeeze()
        #     gt_pos = (i_gt == idx).nonzero().squeeze()
            
        #     # 获取对应的 j_pred 和 j_gt 值
        #     j_pred_group = j_pred[pred_pos] if pred_pos.dim() > 0 else j_pred[pred_pos].unsqueeze(0)
        #     j_gt_group = j_gt[gt_pos] if gt_pos.dim() > 0 else j_gt[gt_pos].unsqueeze(0)
            
        #     # 填充较短的一方（用 0）
        #     max_len = max(len(j_pred_group), len(j_gt_group))
        #     if len(j_pred_group) < max_len:
        #         padding = torch.zeros(max_len - len(j_pred_group), dtype=j_pred_group.dtype)
        #         j_pred_group = torch.cat([j_pred_group, padding])
        #     if len(j_gt_group) < max_len:
        #         padding = torch.zeros(max_len - len(j_gt_group), dtype=j_gt_group.dtype)
        #         j_gt_group = torch.cat([j_gt_group, padding])
            
        #     # 计算当前组的欧式距离
        #     distance = self.match_and_compute_distance(j_pred_group, j_gt_group,w)
        #     distances.append(distance)

        # # 合并所有组的距离（求和或平均）
        # refine_loss_i = torch.sum(torch.stack(distances))

        # 计算欧式距离（忽略 NaN 值）
        # # valid_mask = ~torch.isnan(j_pred_values) & ~torch.isnan(j_gt_values)
        # i_gt_=torch.stack([i_gt_values%w,i_gt_values//w],dim=1)  # 序列变二维  [4354,2]
        # i_pred_=torch.stack([i_pred_values%w,i_pred_values//w],dim=1)
        # # pdb.set_trace()
        # refine_loss_i=torch.sqrt(((i_gt_-i_pred_)**2)).float().mean()



        # i_unique, i_index, i_counts = torch.unique(i_pred, sorted=False, return_inverse=True, return_counts=True)
        # j_unique, j_index, j_counts = torch.unique(j_pred, sorted=False, return_inverse=True, return_counts=True)

        # # pdb.set_trace()
        # #填充gt
        #  # # 只提取重复的行（计数 > 1）
        # i_values0 = i_unique[i_counts > 1]
        # j_values0 = j_unique[j_counts > 1]

        # mask0to1 = (i_gt == i_pred.view(-1, 1)).nonzero()[:, 1]  # 和gt一致的index
        # index = i_gt[mask0to1]
        # mask1to0 = (i_unique == index.view(-1, 1)).nonzero()[:, 1]

        # pdb.set_trace()      
        # pdb.set_trace()
        # j_gt_=torch.stack([j_gt[mask0to1]%w,j_gt[mask0to1]//w],dim=1)  # 序列变二维  [4354,2]
        # j_pred_=torch.stack([j_pred[mask1to0]%w,j_pred[mask1to0]//w],dim=1)
        # # pdb.set_trace()
        # refine_loss_j=torch.sqrt(((j_gt_-j_pred_)**2)).float().mean()


        # mask0to1_j = (j_gt == j_pred.view(-1, 1)).nonzero()[:, 1]  # 和gt一致的index
        # index_j = j_gt[mask0to1_j]
        # mask1to0_j = (j_pred == index_j.view(-1, 1)).nonzero()[:, 1]
        # # pdb.set_trace()
        # i_gt_=torch.stack([i_gt[mask0to1_j]%w,i_gt[mask0to1_j]//w],dim=1)  # 序列变二维  [4354,2]
        # i_pred_=torch.stack([i_pred[mask1to0_j]%w,i_pred[mask1to0_j]//w],dim=1)
        # # pdb.set_trace()
        # refine_loss_i=torch.sqrt(((i_gt_-i_pred_)**2)).float().mean()

        refine_loss = refine_loss_j + refine_loss_i
        return refine_loss
    
    def compute_refine_matching_loss(self, data):
        """ Point-wise Focal Loss with 0 / 1 confidence as gt.
        Args:
        data (dict): {
            conf_matrix_fine (torch.Tensor): (N, W_f^2, W_f^2) 
            conf_matrix_f_gt (torch.Tensor): (N, W_f^2, W_f^2) 
            }
        """
        
        conf_matrix_fine = data['conf_matrix_fine']
        conf_matrix_f_gt = data['conf_matrix_f_gt']
        pos_mask, neg_mask = conf_matrix_f_gt > 0, conf_matrix_f_gt == 0
        pos_w, neg_w = self.c_pos_w, self.c_neg_w

        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            neg_w = 0.

        conf_matrix_fine = torch.clamp(conf_matrix_fine, 1e-6, 1-1e-6)
        alpha = self.loss_config['focal_alpha']
        gamma = self.loss_config['focal_gamma']

        loss_pos = - alpha * torch.pow(1 - conf_matrix_fine[pos_mask], gamma) * (conf_matrix_fine[pos_mask]).log()
        # loss_pos *= conf_matrix_f_gt[pos_mask]
        loss_neg = - alpha * torch.pow(conf_matrix_fine[neg_mask], gamma) * (1 - conf_matrix_fine[neg_mask]).log()

        return pos_w * loss_pos.mean() + neg_w * loss_neg.mean()
    
    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)
        loss = 0
        ## refin loss
        # pdb.set_trace()
        if data.get('repeat_index') is not None: 
            # 1. coarse-level loss
            loss_c = self.compute_coarse_dual_loss(data,weight=c_weight)
            loss = loss_c * self.loss_config['coarse_weight']
            # loss += loss_p * 0.1
            loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})
            loss_refine = self.compute_refine_matching_loss(data)
            # coarse_corr=[data['spv_b_ids'],data['spv_i_ids'],data['spv_j_ids']]
            # loss_refine = self.compute_refine_loss(coarse_corr,data['i_ids'],data['j_ids'],\
            #                                     data['hw0_c'][1], data['hw0_c'][0])
            # # loss_refine=loss_refine*self.refine_weight
            loss += loss_refine * 0.1
            loss_scalars.update({"loss_r": loss_refine.clone().detach().cpu()})
        else:
            # 1. coarse-level loss
            loss_c = self.compute_coarse_loss(data,weight=c_weight)
            loss = loss_c * self.loss_config['coarse_weight']
            # loss += loss_p * 0.1
            loss_refine=torch.tensor(1., dtype=torch.float32)
            loss += loss_refine * 0.1
            loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})
            loss_scalars.update({"loss_r": loss_refine.clone().detach().cpu()})
        # else:
        #     # assert self.training is False
        #     loss_scalars.update({'loss_r': torch.tensor(1.)})  # 1 is the upper bound

        # 2. fine-level loss
        # pdb.set_trace()
        # if data['sig']== 0:
        if 'expec_f0' in data:
            loss_f0 = self.compute_fine_loss(data['expec_f0'], data['expec_f_gt0'])
            loss_f1 = self.compute_fine_loss(data['expec_f1'], data['expec_f_gt1'])
            loss_f = loss_f0+loss_f1
        else:
            loss_f=torch.tensor(1., dtype=torch.float32)
        # else:
            # loss_f = self.compute_fine_loss(data['expec_f'], data['expec_f_gt1'])
        if loss_f is not None:
            loss += loss_f * self.loss_config['fine_weight']
            loss_scalars.update({"loss_f":  loss_f.clone().detach().cpu()})
        else:
            assert self.training is False
            loss_scalars.update({'loss_f': torch.tensor(1.)})  # 1 is the upper bound

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
