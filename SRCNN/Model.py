"""
IN THE NAME OF GOD

Subject: Converts LR images to HR images

Created on Sun Jan 23 09:12:28 2022

@Author: Pouya Sadeghi
"""

#=================================================================================================================
"""
SECTION : Import Necessary Packages (Libraries)
"""
import torch.nn as nn
#=================================================================================================================
"""
SECTION : Build Model
Super Resolution Convolutional Neural Network: 
The proposed SRCNN is different from FSRCNN mainly in three aspects.
First, SRCNN adopts the original low-resolution image as input with
bicubic interpolation. Second, the non-linear mapping step in 
SRCNN is replaced by three steps in FSRCNN, namely the shrinking, 
mapping, and expanding step. Third, FSRCNN adopts smaller filter sizes 
and a deeper network structure. These improvements provide FSRCNN 
with better performance but lower computational cost than SRCNN.

Three Sensitive Variable in SRCNN:
-- The LR feature dimension : < d >
-- The Number of Shrinking Filters: < s >
-- The Mapping Depth: < m >

First Part        :  Conv(9, d, 1)       ,    represents the Feature Extraction.
Second Part   :  Conv(5, s, d)        ,    represents the Shrinking.
Third Part      :  Conv(5, 1, s) * m  ,    represents the Mapping.
"""

# ****************************************************** Function: This Class Defines Structure of SRCNN Model. *******************************************************

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=9, padding=4);
        self.relu1 = nn.ReLU();
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1);
        self.relu2 = nn.ReLU();
        self.conv3 = nn.Conv2d(64, 1, kernel_size=5, padding=2);
        self.relu3 = nn.Sigmoid();

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)

        return out

