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
#==========================================================================================================================================

"""
*************** Load and Setup Data for Training ***************
The Dataset Separated in Two Parts from Original Datasets:
x_train  = 16000 Low Resolution Images
y_train  = 16000 High Resolution Images
"""
# ************************************************************* Class For Data Augmentation **********************************************************************
class Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_x, img_y, img_x_shape, img_y_shape, device):
        self.img_x = img_x
        self.img_y = img_y
        self.img_x_shape = img_x_shape
        self.img_y_shape = img_y_shape
        self.device = device

    def __len__(self):
        return (len(self.img_x))

    def __getitem__(self, i):
        device = torch.device(self.device)
        img_x = torch.tensor(self.img_x[i], dtype=torch.float, device=device)
        img_y = torch.tensor(self.img_y[i], dtype=torch.float, device=device)

        # img_x = (img_x-0.5)/0.5 # change to [-1,1] ***************
        # img_y = (img_y-0.5)/0.5 # change to [-1,1] ***************

        img_x = img_x.reshape(self.img_x_shape)
        img_y = img_y.reshape(self.img_y_shape)

        return img_x, img_y

#==========================================================================================================================================