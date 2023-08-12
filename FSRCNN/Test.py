"""
IN THE NAME OF GOD

Subject: Converts LR images to HR images

Created on Sun Jan 23 09:12:28 2022

@Author: Pouya Sadeghi
"""

#==========================================================================================================================================
"""
SECTION : Import Necessary Packages (Libraries)
"""
from Model import FSRCNN
from Utils import calc_psnr
from Dataset import Image_Dataset
from Utils import plot_imgs_test
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
from torch.utils.data import Dataset
import copy
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import time
from datetime import timedelta, datetime
import pandas as pd
import torch; torch.manual_seed(0)
import torch.utils
import torch.distributions
from torchsummary import summary
import os
import pathlib
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
import math
# mpl.style.use('grayscale')
mpl.style.use('classic')
from matplotlib.backends.backend_pdf import PdfPages
# from skimage.metrics import structural_similarity
from tabulate import tabulate
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'figure.max_open_warning': 0})
from math import log10, sqrt
import argparse
import torch
import torch.backends.cudnn as cudnn
import imageio
import matplotlib.image

#==========================================================================================================================================
if __name__ == '__main__':
    inp = {
        'db_path': ".\\db/Q01.h5",
        'LRsample_shape': (1, 64, 64),  # (Channel, Height, Width)
        'HRsample_shape': (1, 128, 128),  # (Channel, Height, Width)
        'outputs-dir': 'E:\\FSRCNN/New folder/',
        'scale': 2,
        'learning_rate': 1e-3,
        'batch_size': 40,
        'num_epochs': 100,
        'num_workers': 8,
        'seed': 123,
        'optimizer': 'Adam',
        'trDB_size': -1,  # -1 to use all Images available in the db
    }

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(inp['seed'])

    # For Test/Continue_Train chkPath_load at below should be set too.
    # --------------------------
    DatafolderName = os.path.basename(inp['db_path'])
    chkp_Path = f'./_results/8.{DatafolderName[:-3]}_{inp["optimizer"]}_b{inp["batch_size"]}_lr{inp["learning_rate"]}_LR2HR-model3'
    pathlib.Path(chkp_Path).mkdir(parents=True, exist_ok=True)


    from colorama import Fore

    print(Fore.YELLOW + """"
    *********************************************************************************************************************************************************
    ********************************************************* Loading dataset ***********************************************************************************
    *********************************************************************************************************************************************************
    """)
    with h5py.File(inp['db_path'], "r") as hf:
        # Split the Data into Training / Test
        # Split the Data into Features / Targets
        train_x = np.array(hf["LR/Train"]['3D_image'][:inp['trDB_size']])
        train_y = np.array(hf["HR/Train"]['3D_image'][:inp['trDB_size']])
        test_x = np.array(hf["LR/Test"]['3D_image'][:inp['trDB_size']])
        test_y = np.array(hf["HR/Test"]['3D_image'][:inp['trDB_size']])

    Train_x = train_x[0:9990]
    Train_y = train_y[0:9990]
    Eval_x = train_x[9990:11200]
    Eval_y = train_y[9990:11200]
    Test_x = test_x[0:2800]
    Test_y = test_y[0:2800]

    Train_Dataset = Image_Dataset(Train_x, Train_y, inp['LRsample_shape'], inp['HRsample_shape'], device)
    print('TrainDB:', Train_Dataset.img_x.shape[0])
    train_dataloader = DataLoader(dataset=Train_Dataset,
                                  batch_size=inp['batch_size'],
                                  shuffle=True)
                                # num_workers = inp['num_workers'],
                                # pin_memory = True)

    Eval_Dataset = Image_Dataset(Eval_x, Eval_y, inp['LRsample_shape'], inp['HRsample_shape'], device)
    print('ValidationDB:', Eval_Dataset.img_x.shape[0])
    Eval_dataloader = DataLoader(dataset=Eval_Dataset,
                                 batch_size=1)

    Test_Dataset = Image_Dataset(Test_x, Test_y, inp['LRsample_shape'], inp['HRsample_shape'], device)
    print('TestDB:', Test_Dataset.img_x.shape[0])
    Test_dataloader = DataLoader(dataset=Test_Dataset,
                                 batch_size=1)

    print(Fore.GREEN + """"
    *********************************************************************************************************************************************************
     ***************************************************************Start of Testing Network********************************************************************** 
     *********************************************************************************************************************************************************""")

#==========================================================================================================================================
"""
SECTION : Test Model
"""

#============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    args = parser.parse_args(['--weights-file', 'C:\\Users/Mehrgan/PycharmProjects/pythonProject/Final/New folder/best.pth',
                                            '--image-file', 'C:\\Users/Mehrgan/PycharmProjects/pythonProject/Final/Image Folder2/'])

    # DatafolderName = os.path.basename(args.image_file)
    # chk_Path = f'./Image Output/8.{DatafolderName[:-3]}_{inp["optimizer"]}_b{inp["batch_size"]}_lr{inp["learning_rate"]}_LR2HR-model3'

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FSRCNN(scale_factor=args.scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    # i = 0
    # for x, y in Test_dataloader:
    #     with torch.no_grad():
    #         sr = model(x).clamp(0.0, 1.0)
    #         sr = sr.cpu().numpy()
    #         # sr = np.uint8(255 * (sr - np.min(sr)) / (np.max(sr) - np.min(sr)))
    #         # imageio.imwrite(f'{i}_Super_Resolution.png', sr[0][0])
    #         matplotlib.image.imsave(f'{i}_Super_Resolution.png', sr[0][0])
    #         hr = y.cpu().numpy()
    #         # hr = np.uint8(255 * (hr - np.min(hr)) / (np.max(hr) - np.min(hr)))
    #         # imageio.imwrite(f'{i}_High_Resolution.png', hr[0][0])
    #         matplotlib.image.imsave(f'{i}_High_Resolution.png', hr[0][0])
    #         lr = x.cpu().numpy()
    #         # lr = np.uint8(255 * (lr - np.min(lr)) / (np.max(lr) - np.min(lr)))
    #         # imageio.imwrite(f'{i}_Low_Resolution.png', lr[0][0])
    #         matplotlib.image.imsave(f'{i}_Low_Resolution.png', lr[0][0])
    #         i+=1
    i=0
    # epoch_psnr_test = AverageMeter()
    PSNR_Final = np.zeros(2800)
    for x, y in Test_dataloader:
        with torch.no_grad():
            sr = model(x).clamp(0.0, 1.0)
            # plot reconstructe
            if i % 10 == 0:
                plot_imgs_test(x, sr, y, i)
            PSNR = calc_psnr(y, sr)
            PSNR_Final[i] = PSNR
            i+=1
            table = [["PSNR", PSNR]]
            print(tabulate(table, headers=["\nPeak Signal-to-Noise ratio", "\nValue of PSNR"]))
            print(Fore.RED + '==========================================')

    df = pd.DataFrame({"PSNR": [PSNR_Final]})
    df.to_csv("output_test.csv")
print("Finish")
print(Fore.RED + '==========================================')
#==========================================================================================================================================
"""
END
"""
#==========================================================================================================================================