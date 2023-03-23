import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Sampler
from torchvision import transforms
import time
import os, sys
import numpy as np
import math
import torch

import cv2 

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def colorize(value, vmin=None, vmax=None, cmap='Greys'):
    value = value.cpu().numpy()[:, :, :]
    value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value*0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']


class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


def flip_lr(image):
    """
    Flip image horizontally

    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Image to be flipped

    Returns
    -------
    image_flipped : torch.Tensor [B,3,H,W]
        Flipped image
    """
    assert image.dim() == 4, 'You need to provide a [B,C,H,W] image to flip'
    return torch.flip(image, [3])


def fuse_inv_depth(inv_depth, inv_depth_hat, method='mean'):
    """
    Fuse inverse depth and flipped inverse depth maps

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_hat : torch.Tensor [B,1,H,W]
        Flipped inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    fused_inv_depth : torch.Tensor [B,1,H,W]
        Fused inverse depth map
    """
    if method == 'mean':
        return 0.5 * (inv_depth + inv_depth_hat)
    elif method == 'max':
        return torch.max(inv_depth, inv_depth_hat)
    elif method == 'min':
        return torch.min(inv_depth, inv_depth_hat)
    else:
        raise ValueError('Unknown post-process method {}'.format(method))


def post_process_depth(depth, depth_flipped, method='mean'):
    """
    Post-process an inverse and flipped inverse depth map

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_flipped : torch.Tensor [B,1,H,W]
        Inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    inv_depth_pp : torch.Tensor [B,1,H,W]
        Post-processed inverse depth map
    """
    
    B, C, H, W = depth.shape
    inv_depth_hat = flip_lr(depth_flipped)
    inv_depth_fused = fuse_inv_depth(depth, inv_depth_hat, method=method)
    xs = torch.linspace(0., 1., W, device=depth.device,
                        dtype=depth.dtype).repeat(B, C, H, 1)
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    mask_hat = flip_lr(mask)
    return mask_hat * depth + mask * inv_depth_hat + \
           (1.0 - mask - mask_hat) * inv_depth_fused




def depth_value_to_depth_image(depth_values, is_gt = False):
    depth_values = cv2.normalize(depth_values, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    depth = (depth_values * 255).astype(np.uint8)
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
    return depth

def normalize_image(image):
    input_image = np.asarray(image, dtype=np.float32) / 255.0
    input_image = np.expand_dims(input_image, axis=0)
    input_image = np.transpose(input_image, (0, 3, 1, 2))
    
    return input_image

#####################################################################################################################
def wait_frame_fps(wait_frame, time_frame, last_time):
    time_sleep_frame = max(0, wait_frame - time_frame)
    time.sleep(time_sleep_frame)

    return 1/(time.perf_counter()-last_time)


#####################################################################################################################

def plot_results(fig_list):
    # title_list = ["Input", "AdaBin", "NewCRFs", "OURs"]
    # myorder = [0, 3, 1, 2]
    # mylist = [fig_list[i] for i in myorder]

    # disp = np.hstack(fig_list)
    # h,w = disp.shape[0],disp.shape[1]

    # # Removes toolbar and status bar
    # cv2.namedWindow('Deltax', flags=cv2.WINDOW_GUI_NORMAL)
    # cv2.putText(disp, title_list[0], (30,50), cv2.FONT_HERSHEY_SIMPLEX, 
    #                1, (255,0,0), 2, cv2.LINE_AA)
    # cv2.putText(disp, title_list[1], (int(w/4)+30,50), cv2.FONT_HERSHEY_SIMPLEX, 
    # 1, (255,0,0), 2, cv2.LINE_AA)
    # cv2.putText(disp, title_list[2], (int(w/2)+30,50), cv2.FONT_HERSHEY_SIMPLEX, 
    #                1, (255,0,0), 2, cv2.LINE_AA)
    # cv2.putText(disp, title_list[3], (650+650+650,50), cv2.FONT_HERSHEY_SIMPLEX, 
    # 1, (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow('Deltax', fig_list)
