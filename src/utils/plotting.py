import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pdb
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision.utils import make_grid
from matplotlib.colors import LinearSegmentedColormap
import torch

transparency = 0.4
sz = 2
def _compute_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'scannet':
        thr = 5e-4
    elif dataset_name == 'megadepth':
        thr = 1e-4
    elif dataset_name == 'vissar':
        thr = 5e-4
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr

def make_temp_figure1(raw_img, name):

    dpi = 75
    raw_img0 = (raw_img[0,0].cpu().numpy() * 255).round().astype(np.int32)
    raw_img1 = (raw_img[1,0].cpu().numpy() * 255).round().astype(np.int32)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(raw_img0)
    axes[1].imshow(raw_img1)
    for i in range(2):   
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    path = name
    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    return fig

def make_temp_figure(raw_img, img,name, cha):
    # draw image pair
    dpi = 75
    raw_img0 = raw_img[0,0].cpu().numpy()
    raw_img1 = raw_img[1,0].cpu().numpy()
    if len(img.shape)<4:
        img0 = F.interpolate(img[0].unsqueeze(0).unsqueeze(0), scale_factor=8, mode='bilinear', align_corners=True)
        img1 = F.interpolate(img[1].unsqueeze(0).unsqueeze(0), scale_factor=8, mode='bilinear', align_corners=True)
    else:
        img0 = F.interpolate(img[0].unsqueeze(0), scale_factor=8, mode='bilinear', align_corners=True)
        img1 = F.interpolate(img[1].unsqueeze(0), scale_factor=8, mode='bilinear', align_corners=True)
    img0 = img0[0,cha].detach().cpu().numpy()
    img1 = img1[0,cha].detach().cpu().numpy()
 
    cmap_jet = plt.cm.get_cmap('jet')

    colors = cmap_jet(np.linspace(0, 1, 256))
    colors[:, 3] = np.where(colors[:, 1] < 0.5, 1, 0)
    new_cmap = LinearSegmentedColormap.from_list('custom_jet', colors)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(raw_img0,cmap='gray')
    axes[0].imshow(img0,alpha=transparency,cmap=new_cmap, interpolation='nearest')
    axes[1].imshow(raw_img1,cmap='gray')
    axes[1].imshow(img1,alpha=transparency,cmap=new_cmap, interpolation='nearest')

    for i in range(2):  
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    path = name+str(cha)+'.png'
    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()


def grid_figure(raw_img, img, name):
    dpi = 75
    raw_img0 = raw_img[0,0].cpu().numpy()
    raw_img1 = raw_img[1,0].cpu().numpy()
    img0 = F.interpolate(img[0].unsqueeze(0), scale_factor=8, mode='bilinear', align_corners=True)
    img1 = F.interpolate(img[1].unsqueeze(0), scale_factor=8, mode='bilinear', align_corners=True)
    for id in range(img.shape[1]):
        img00 = img0[0,id].detach().cpu().numpy()
        img11 = img1[0,id].detach().cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
        axes[0].imshow(raw_img0,cmap='gray')
        axes[0].imshow(img00,alpha=transparency,cmap='jet', interpolation='nearest')
        axes[1].imshow(raw_img1,cmap='gray')
        axes[1].imshow(img11,alpha=transparency,cmap='jet', interpolation='nearest')

        for i in range(2):   
            axes[i].get_yaxis().set_ticks([])
            axes[i].get_xaxis().set_ticks([])
            for spine in axes[i].spines.values():
                spine.set_visible(False)
        plt.tight_layout(pad=1)
        
        path = name+str(id)+'.png'
        plt.savefig(path, bbox_inches='tight', pad_inches=0)



def make_raw_figure(raw_img, name):
    # draw image pair
    dpi = 75
    raw_img0 = (raw_img[0,0].cpu().numpy() * 255).round().astype(np.int32)
    raw_img1 = (raw_img[1,0].cpu().numpy() * 255).round().astype(np.int32)

    plt.figure()
    plt.imshow(raw_img0,cmap='gray')
    plt.axis('off')
    path3 = name+'-1.png'
    plt.savefig(path3,bbox_inches='tight',pad_inches=0.0)

    plt.figure()
    plt.imshow(raw_img1,cmap='gray')
    plt.axis('off')
    path4 = name+'-2.png'
    plt.savefig(path4,bbox_inches='tight',pad_inches=0.0)

def make_single_figure(raw_img, img,name):
    # draw image pair
    dpi = 75
    raw_img0 = (raw_img[0,0].cpu().numpy() * 255).round().astype(np.int32)
    img0 = F.interpolate(img, scale_factor=8, mode='bilinear', align_corners=False)
    img0 = (img0[0,0].detach().cpu().numpy() * 255).round().astype(np.int32)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0+raw_img0)
    axes[1].imshow(img0)
    for i in range(2):   
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    path = name
    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    return fig

def make_fig_point(img0, img1, point1, point2, conf, path):
    dpi = 75
    img0 = (img0[0,0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (img1[0,0].cpu().numpy() * 255).round().astype(np.int32)
    conf = conf.cpu().numpy()
    colors = [plt.cm.jet(confidence) for confidence in conf]
    point1 = point1.detach().cpu().numpy()
    point2 = point2.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0)
    axes[0].scatter(point1[:,0], point1[:,1], c=colors, cmap='viridis')
    axes[1].imshow(img1)
    axes[1].scatter(point2[:,0], point2[:,1], c=colors, cmap='viridis')
    for i in range(2):   
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    return fig


def make_deltafeature(img0, img1, img00, img11, path):
    # draw image pair
    dpi = 75
    img00 = (img00[0,0].cpu().numpy() * 255).round().astype(np.int32)
    img11 = (img11[0,0].cpu().numpy() * 255).round().astype(np.int32)
    img0 = img0.reshape(60,80,5,-1)[:,:,0]
    img1 = img1.reshape(60,80,5,-1)[:,:,0]
    img0 = F.interpolate(img0.permute(2,0,1).unsqueeze(0), scale_factor=8, mode='bilinear', align_corners=False)
    img1 = F.interpolate(img1.permute(2,0,1).unsqueeze(0), scale_factor=8, mode='bilinear', align_corners=False)
    img0 = img0.squeeze(0).permute(1,2,0)
    img1 = img1.squeeze(0).permute(1,2,0)
    img0 = (img0[:,:,0].detach().cpu().numpy() * 255).round().astype(np.int32)
    img1 = (img1[:,:,0].detach().cpu().numpy() * 255).round().astype(np.int32)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0+img00)
    axes[1].imshow(img1+img11)
    for i in range(2):   
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    return fig

def make_anglefeature(img00, img11, img0, img1, point1, point2, path):
    # draw image pair
    dpi = 75
    img00 = (img00[0,0].cpu().numpy() * 255).round().astype(np.int32)
    img11 = (img11[0,0].cpu().numpy() * 255).round().astype(np.int32)
    point1 = point1.detach().cpu().numpy()
    point2 = point2.detach().cpu().numpy()
    img0 = img0.reshape(60,80,-1)
    img1 = img1.reshape(60,80,-1)
    img0 = F.interpolate(img0.permute(2,0,1).unsqueeze(0), scale_factor=8, mode='bilinear', align_corners=False)
    img1 = F.interpolate(img1.permute(2,0,1).unsqueeze(0), scale_factor=8, mode='bilinear', align_corners=False)
    img0 = img0.squeeze(0).permute(1,2,0)
    img1 = img1.squeeze(0).permute(1,2,0)
    img0 = (img0[:,:,0].detach().cpu().numpy() * 255).round().astype(np.int32)
    img1 = (img1[:,:,0].detach().cpu().numpy() * 255).round().astype(np.int32)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img00,cmap='gray')
    axes[0].imshow(img0,alpha=transparency)
    axes[0].scatter(point1[:,0], point1[:,1], c='red')
    axes[1].imshow(img11,cmap='gray')
    axes[1].imshow(img1,alpha=transparency)
    axes[1].scatter(point2[:,0], point2[:,1], c='red')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.savefig(str(path)+'.svg', bbox_inches='tight', pad_inches=0, format='svg')
    plt.close()


def make_conffeature(img00, img11, img, point1, point2, path):
    # draw image pair
    dpi = 75
    img00 = (img00[0,0].cpu().numpy() * 255).round().astype(np.int32)
    img11 = (img11[0,0].cpu().numpy() * 255).round().astype(np.int32)
    point1 = point1.detach().cpu().numpy()
    point2 = point2.detach().cpu().numpy()
    img0 = F.interpolate(img[0].unsqueeze(0), scale_factor=8, mode='bilinear', align_corners=True)
    img1 = F.interpolate(img[1].unsqueeze(0), scale_factor=8, mode='bilinear', align_corners=True)
    img0 = (img0[0,0].detach().cpu().numpy() * 255).round().astype(np.int32)
    img1 = (img1[0,0].detach().cpu().numpy() * 255).round().astype(np.int32)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img00,cmap='gray')
    axes[0].imshow(img0,alpha=transparency,cmap='jet', interpolation='nearest')
    axes[0].scatter(point1[:,0], point1[:,1], s=sz, c='red')
    axes[1].imshow(img11,cmap='gray')
    axes[1].imshow(img1,alpha=transparency,cmap='jet', interpolation='nearest')
    axes[1].scatter(point2[:,0], point2[:,1], s=sz, c='red')
    for i in range(2):   
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.savefig(str(path)+'.svg', bbox_inches='tight', pad_inches=0, format='svg')
    return fig

def make_posencoding(img0, img1, path):
    # draw image pair
    dpi = 75
    img0 = (img0[0,0].detach().cpu().numpy() * 255).round().astype(np.int32)
    img1 = (img1[0,0].detach().cpu().numpy() * 255).round().astype(np.int32)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0)
    axes[1].imshow(img1)
    for i in range(2):   
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    return fig

# --- VISUALIZATION --- #

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, 
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
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
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c='r', s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c='r', s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    plt.savefig("/home/results/"+str(path[0][0]), bbox_inches='tight', pad_inches=0)
    return fig


def _make_evaluation_figure(data, b_id, alpha='dynamic'):
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)
    data_name = data['pair_names']
    
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)

    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()
    
    figure = make_matching_figure(img0, img1, kpts0, kpts1, path=data_name)
    return figure

def _make_confidence_figure(data, b_id):
    # TODO: Implement confidence figure
    raise NotImplementedError()


def make_matching_figures(data, config, mode='evaluation'):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence','train']  # 'confidence'
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            fig = _make_evaluation_figure(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA)
        elif mode == 'train':
            fig = _make_evaluation_figure(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA)
        elif mode == 'confidence':
            fig = _make_confidence_figure(data, b_id)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
    figures[mode].append(fig)
    return figures


def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)
