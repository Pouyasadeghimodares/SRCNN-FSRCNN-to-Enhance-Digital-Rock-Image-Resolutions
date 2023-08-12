"""
IN THE NAME OF GOD

Subject:  Utils (Definition  Necessary Functions)

Created on Sun Jan 23 09:12:28 2022

@Author: Pouya Sadeghi
"""
#==========================================================================================================================================
"""
Import Necessary Packages (Libraries)
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.backends.backend_pdf import PdfPages
import os
import pathlib
import argparse
#==========================================================================================================================================

inp = {
    'db_path': ".\\db/Q01.h5",
    'learning_rate': 1e-3,
    'batch_size': 40,
    'optimizer': 'Adam'
     }
#==========================================================================================================================================

# For Test/Continue_Train chkPath_load at below should be set too.
# --------------------------
DatafolderName = os.path.basename(inp['db_path'])
chkp_Path = f'./_results/8.{DatafolderName[:-3]}_{inp["optimizer"]}_b{inp["batch_size"]}_lr{inp["learning_rate"]}_LR2HR-model3'
pathlib.Path(chkp_Path).mkdir(parents=True, exist_ok=True)

# *****************************************************  Function: This Function Calculate PSNR between two images. *****************************************************
# def calc_psnr(img1, img2):
#     return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def calc_psnr (img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore, PSNR have no importance.
        return 100
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

# *********************************************************  Function: This Class is for Calculate AverageMeter. *********************************************************
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# *******************************************************  Function: This function is used for plotting Loss animation. ******************************************************
def plot_loss_animation(iteration_list, loss, fig=None):
    # plots an animation of loss during training/validation
    def create_fig():
        # enable interactive mode
        plt.ion()
        # fig = plt.figure(figsize=(10,5))
        fig, ax1 = plt.subplots(1)
        fig.set_figheight(3)
        fig.set_figwidth(3)
        fig.suptitle('Loss during Training')
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("Loss")
        ax1.plot(iteration_list, loss, label="Train")
        # ax1.legend()
        plt.tight_layout()
        return fig

    if len(iteration_list) == 1:
        fig = create_fig()
    else:
        # updating the value of x and y
        fig.axes[0].lines[0].set_xdata(iteration_list)
        fig.axes[0].lines[0].set_ydata(loss)
        # recompute the ax.dataLim
        fig.axes[0].relim()
        # update ax.viewLim using the new dataLim
        fig.axes[0].autoscale_view()
        # plt.legend()

    # re-drawing the figure
    fig.canvas.draw()
    # to flush the GUI events
    fig.canvas.flush_events()
    plt.tight_layout()
    time.sleep(0.1)

    fig.savefig(f'{chkp_Path}/Loss_Train.png', dpi=300)
    np.save(f'{chkp_Path}/loss.npy', loss)
    np.save(f'{chkp_Path}/iteration_list.npy', iteration_list)
    return fig

def plot_loss_animation_valid(iteration_list, loss, fig=None):
    # plots an animation of loss during training/validation
    def create_fig():
        # enable interactive mode
        plt.ion()
        # fig = plt.figure(figsize=(10,5))
        fig, ax1 = plt.subplots(1)
        fig.set_figheight(3)
        fig.set_figwidth(3)
        fig.suptitle('Loss during Validation')
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("Loss")
        ax1.plot(iteration_list, loss, label="Validation")
        # ax1.legend()
        plt.tight_layout()
        return fig

    if len(iteration_list) == 1:
        fig = create_fig()
    else:
        # updating the value of x and y
        fig.axes[0].lines[0].set_xdata(iteration_list)
        fig.axes[0].lines[0].set_ydata(loss)
        # recompute the ax.dataLim
        fig.axes[0].relim()
        # update ax.viewLim using the new dataLim
        fig.axes[0].autoscale_view()
        # plt.legend()

    # re-drawing the figure
    fig.canvas.draw()
    # to flush the GUI events
    fig.canvas.flush_events()
    plt.tight_layout()
    time.sleep(0.1)

    fig.savefig(f'{chkp_Path}/Loss_Validation.png', dpi=300)
    np.save(f'{chkp_Path}/loss.npy', loss)
    np.save(f'{chkp_Path}/iteration_list.npy', iteration_list)
    return fig
# ************************************  Function: This Function is used for Plotting Input Image and Reconstructed Image and Ground truth Image. ************************************
def plot_imgs(inp, outp, gt, epoch):
    # input image, reconstructed image, ground truth
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(f'Epoch = {epoch}')
    ax1.imshow(inp.cpu().detach().numpy()[0, 0, :, :])
    ax1.set_title('Input')
    ax1.set_xlim([0, 64])
    ax1.set_ylim([0, 64])
    ax2.imshow(outp.cpu().detach().numpy()[0, 0, :, :])
    ax2.set_title('Reconstructed')
    ax2.set_xlim([0, 128])
    ax2.set_ylim([0, 128])
    ax3.imshow(gt.cpu().detach().numpy()[0, 0, :, :])
    ax3.set_title('Ground truth')
    ax3.set_xlim([0, 128])
    ax3.set_ylim([0, 128])
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{chkp_Path}/{epoch}.png', dpi=300)
    time.sleep(0.1)
    plt.close()

# **************************************************  Function: This Function for Creating a PDF  File with Several Pages. ****************************************************
def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

#==========================================================================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-file', type=str, required=True)
    args = parser.parse_args(['--image-file', 'C:\\Users/Mehrgan/PycharmProjects/pythonProject/Final/Image Folder/'])

def plot_imgs_test(inp, outp, gt, epoch):
    # input image, reconstructed image, ground truth
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(f'Sample = {epoch}')
    ax1.imshow(inp.cpu().detach().numpy()[0, 0, :, :])
    ax1.set_title('Input')
    ax1.set_xlim([0, 64])
    ax1.set_ylim([0, 64])
    ax2.imshow(outp.cpu().detach().numpy()[0, 0, :, :])
    ax2.set_title('Reconstructed')
    ax2.set_xlim([0, 128])
    ax2.set_ylim([0, 128])
    ax3.imshow(gt.cpu().detach().numpy()[0, 0, :, :])
    ax3.set_title('Ground truth')
    ax3.set_xlim([0, 128])
    ax3.set_ylim([0, 128])
    plt.tight_layout()
    # plt.show()
    # plt.savefig(f'{chk_Path}/{epoch}.png', dpi=300)
    plt.savefig(f'{args.image_file}/{epoch}.png', dpi=300)
    time.sleep(0.1)
    plt.close(fig)
#==========================================================================================================================================
