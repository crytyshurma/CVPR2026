from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
from pathlib import Path
import os
import logging
import xlwt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from option import args


def create_dir(args):
    save_dir = Path(args.path_save)
    save_dir.mkdir(exist_ok=True)

    val_dir = save_dir.joinpath('test')
    val_dir.mkdir(exist_ok=True)

    return save_dir, val_dir


def cal_metrics(gt_LF, reconstructed_LF):
        if gt_LF.ndim == 3:
            view_point_num, _, _ = gt_LF.shape
            chanel_num = 1
        else:
            view_point_num, _, _, chanel_num = gt_LF.shape

        ssim = np.zeros((view_point_num), np.float32)
        psnr = np.zeros((view_point_num), np.float32)
        mse = np.zeros((view_point_num), np.float32)

        for t in  range(view_point_num):
            sub_psnr = peak_signal_noise_ratio(reconstructed_LF[t], gt_LF[t])
            sub_mse = mean_squared_error(reconstructed_LF[t], gt_LF[t])

            if chanel_num == 1:
                sub_ssim = structural_similarity(reconstructed_LF[t,...,0], gt_LF[t,...,0], multichannel=False)
            else:
                sub_ssim = structural_similarity(reconstructed_LF[t], gt_LF[t], multichannel=True)

            psnr[t] = sub_psnr
            ssim[t] = sub_ssim
            mse[t] = sub_mse

        return psnr, ssim, mse


def vis_event(event_frame, event_max_count, cbar=False):
    H, W, _ = event_frame.shape
    dpi = 100

    fig = plt.figure(figsize=(W/dpi, H/dpi))
    ax = fig.add_subplot(111)
    ax.axis("off")
    img = ax.imshow(event_frame, cmap="bwr", norm=Normalize(vmin=-event_max_count, vmax=event_max_count))
    if cbar == True:
        plt.colorbar(img, orientation="vertical")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    plt.close()
    return fig