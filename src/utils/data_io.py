import torch
from torch import nn
import numpy as np
import cv2
import torch.nn.functional as F
from yacs.config import CfgNode as CN

def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def upper_config(dict_cfg):
    if not isinstance(dict_cfg, dict):
        return dict_cfg
    return {k.upper(): upper_config(v) for k, v in dict_cfg.items()}


class DataIOWrapper(nn.Module):

    def __init__(self, model, config, ckpt=None):
        super().__init__()

        self.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
        torch.set_grad_enabled(False)
        self.model = model
        self.config = config
        self.img0_size =  640
        self.img1_size = 640
        self.df = 16
        self.padding = False
        self.coarse_scale = 0.125

        if ckpt:
            ckpt_dict = torch.load(ckpt)
            self.model.load_state_dict(ckpt_dict['state_dict'], strict= False)
            self.model = self.model.eval().to(self.device)

    def preprocess_image(self, img, device, resize=None, df=None, padding=None, cam_K=None, dist=None, gray_scale=True):
        if gray_scale and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = img.shape[:2]
        
        img = cv2.resize(img, (640, 480))

        if len(img.shape) == 2: 
            img_tensor = torch.from_numpy(img)[None][None].cuda().float() / 255.0
        else:  
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
        return img, img_tensor
    
    def from_cv_imgs(self, img0, img1, K0=None, K1=None, dist0=None, dist1=None):
        image0, img0_tensor = self.preprocess_image(
            img0, self.device, resize=self.img0_size, df=self.df, padding=self.padding, cam_K=K0, dist=dist0)
        image1, img1_tensor = self.preprocess_image(
            img1, self.device, resize=self.img1_size, df=self.df, padding=self.padding, cam_K=K1, dist=dist1)
        mkpts0, mkpts1 = self.match_images(img0_tensor, img1_tensor)
        matches = np.concatenate([mkpts0, mkpts1], axis=1)
        data = {'matches':matches,
                'mkpts0':mkpts0,
                'mkpts1':mkpts1,
                'img0':image0,
                'img1':image1
                }

        return data

    def from_paths(self, img0_pth, img1_pth, K0=None, K1=None, dist0=None, dist1=None, read_color=False):
        
        imread_flag = cv2.IMREAD_COLOR if read_color else cv2.IMREAD_GRAYSCALE

        img0 = cv2.imread(img0_pth, imread_flag)
        img1 = cv2.imread(img1_pth, imread_flag)
        return self.from_cv_imgs(img0, img1, K0=K0, K1=K1, dist0=dist0, dist1=dist1)

    def match_images(self, image0, image1):
        batch = {'image0': image0, 'image1': image1}
        self.model(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        return mkpts0, mkpts1
    
    def pad_bottom_right(self, inp, pad_size, ret_mask=False):
        assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
        mask = None
        if inp.ndim == 2:
            padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
            padded[:inp.shape[0], :inp.shape[1]] = inp
            if ret_mask:
                mask = np.zeros((pad_size, pad_size), dtype=bool)
                mask[:inp.shape[0], :inp.shape[1]] = True
        elif inp.ndim == 3:
            padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
            padded[:, :inp.shape[1], :inp.shape[2]] = inp
            if ret_mask:
                mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
                mask[:, :inp.shape[1], :inp.shape[2]] = True
        else:
            raise NotImplementedError()
        return padded, mask

