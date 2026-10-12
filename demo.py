import os
import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import matplotlib

os.sys.path.append("../")  
from src.loftr import LoFTR, default_cfg
from src.config.default import get_cfg_defaults
from src.utils.data_io import DataIOWrapper, lower_config


torch.set_grad_enabled(False)

os.chdir(sys.path[0])  

config = get_cfg_defaults()
config = lower_config(config)

matcher = LoFTR(config=config["loftr"])

ckpt = "weights/multimodal.ckpt"

matcher = DataIOWrapper(matcher, config=config, ckpt=ckpt)

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, 
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):

    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c='lime', linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c="r", s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c="r", s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(os.path.join(vis_result_path, f"{path}_overlay.jpg"), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


if __name__=='__main__':
    vis_path = "test_data/VIS"
    sar_path = "test_data/OTH"
    result_path = "./results"

    vis_result_path = os.path.join(result_path, "registration_vis")
    os.makedirs(vis_result_path, exist_ok=True)

    for path in os.listdir(vis_path):
        print(path)
        id = path.split('.')[0]
        im1_path = os.path.join(vis_path, path).replace('\\', '/')
        im2_path = os.path.join(sar_path, path).replace('\\', '/')

        # Create model
        s_time = time.time()
        output_data = matcher.from_paths(im1_path, im2_path, read_color=True)

        # Matched keypoints
        kpts1 = output_data['mkpts0']
        kpts2 = output_data['mkpts1']

        image0 = cv2.imread(im1_path, cv2.IMREAD_COLOR)
        image1 = cv2.imread(im2_path, cv2.IMREAD_COLOR)
        img0 = cv2.resize(image0, (640,480))
        img1 = cv2.resize(image1, (640,480))
       
        e_time = time.time()
        ourtime = e_time-s_time

        mkpts0 = kpts1
        mkpts1 = kpts2

        if len(mkpts0) >= 4:
            ret_H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)
        else:
            inliers = None
            ret_H = None

        point1 = mkpts0[np.where(inliers)[0]]
        point2 = mkpts1[np.where(inliers)[0]]
    
        text = [
        'LAMM',
        'Matches: {}'.format(len(point1))]

        if len(img0.shape) == 3:
            _img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        else:
            _img0 = img0
        if len(img1.shape) == 3:
            _img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        else:
            _img1 = img1
            

        fig_match = make_matching_figure(_img0, _img1, point1, point2, text=text, dpi=125, path = id)

        # Registration
        if inliers is not None and ret_H is not None:

            h_ref, w_ref = img1.shape[:2]
            
            warped_img0 = cv2.warpPerspective(img0, ret_H, (w_ref, h_ref))
            
            blended_image = cv2.addWeighted(img1, 0.4, warped_img0, 0.6, 0)
            image0_resized = cv2.resize(img0, (w_ref, h_ref))
            
            cv2.putText(image0_resized, "VIS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img1, "OTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(blended_image, "Registered Overlay", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            concat_result = np.concatenate((image0_resized, img1, blended_image), axis=1)
            
            save_name = os.path.join(vis_result_path, f"{id}_overlayres.jpg")
            cv2.imwrite(save_name, concat_result)
            print(f"Saved registration overlay: {save_name}")
  

        else:
            print(f"[{id}] Not enough matches for RANSAC!")
            point1 = []
            point2 = []