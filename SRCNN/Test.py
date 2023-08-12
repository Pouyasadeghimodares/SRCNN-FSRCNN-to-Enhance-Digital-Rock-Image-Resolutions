"""
IN THE NAME OF GOD

Subject: Converts LR images to HR images

Created on Sun Jan 23 09:12:28 2022

@Author: Pouya Sadeghi
"""

#=================================================================================================================
"""
SECTION 1 : Import Necessary Packages (Libraries)
"""
from Model import SRCNN
from Utils import calc_psnr, plot_imgs_test
from Dataset import Image_Dataset
import numpy as np
import h5py
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import torch; torch.manual_seed(0)
import os
import pathlib
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
mpl.style.use('grayscale')
from tabulate import tabulate
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'figure.max_open_warning': 0})
import argparse
import torch
import torch.backends.cudnn as cudnn
from colorama import Fore

#=================================================================================================================

if __name__ == '__main__':
    inp = {
        'db_path': ".\\images/db/db.h5",
        'LRsample_shape': (1, 64, 64),  # (Channel, Height, Width)
        'HRsample_shape': (1, 128, 128),  # (Channel, Height, Width)
        'scale': 2,
        'learning_rate': 1e-3,
        'batch_size': 15,
        'num_epochs': 500,
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

    Train_x = train_x[0:8970]
    Train_y = train_y[0:8970]
    Eval_x = train_x[8970:11520]
    Eval_y = train_y[8970:11520]
    Test_x = train_x[11520:]
    Test_y = train_y[11520:]

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
     ***************************************************************Start of Tessting Network********************************************************************** 
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
    args = parser.parse_args(['--weights-file', 'E:\\SRCNN/OUTPUT1/New folder/best.pth',
                                            '--image-file', 'E:\\SRCNN/OUTPUT1/Image Folder/'])

    # DatafolderName = os.path.basename(args.image_file)
    # chk_Path = f'./Image Output/8.{DatafolderName[:-3]}_{inp["optimizer"]}_b{inp["batch_size"]}_lr{inp["learning_rate"]}_LR2HR-model3'

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN(scale_factor=args.scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    i = 0
    model.eval()
    # epoch_psnr_test = AverageMeter()
    PSNR = np.zeros(1279)
    for x, y in Test_dataloader:
        with torch.no_grad():
            sr = model(x).clamp(0.0, 1.0)
            # plot reconstructe
            if i % 10 == 0:
                plot_imgs_test(x, sr, y, i)
            PSNR = calc_psnr(y, sr)
            i+=1
            table = [["PSNR", PSNR]]
            print(tabulate(table, headers=["\nPeak Signal-to-Noise ratio", "\nValue of PSNR"]))
            print(Fore.RED + '==========================================')

    df = pd.DataFrame({"PSNR": [PSNR]})
    df.to_csv("output_test.csv")
print("Finish")
print(Fore.RED + '==========================================')
#=================================================================================================================
"""
END
"""
#=================================================================================================================