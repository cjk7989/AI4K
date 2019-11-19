import numpy as np
import random
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import cv2

sourcePath = './data/high/'
savedPath = './data/high2low/'
filelist = sorted(os.listdir(sourcePath))
for filename in filelist:
    img = cv2.imread(sourcePath + filename)
    img = cv2.resize(img, (960, 540))
    cv2.imwrite(savedPath + filename, img)