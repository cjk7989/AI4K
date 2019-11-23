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
import torch.nn.init as init

from train import Net

low_files = os.listdir('./data/test/')
checkpoint_dir = "./checkpoint/"
epoch = 4
batch_size = 5

def test_path(p): return f"./data/test/{p}"
low_files = list(map(test_path, low_files))

class TestDataset(Dataset):
    def __init__(self, image_paths, transform):
        super().__init__()
        self.paths = image_paths
        self.len = len(self.paths)
        self.transform = transform
        
    def __len__(self): return self.len
    
    def __getitem__(self, index): 
        low_path = self.paths[index]
        low = Image.open(low_path).convert('RGB') # 540 * 960
        name = low_path.split('/')[-1]
        low = self.transform(low)
        return (low, name)

transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))
])

test_ds = TestDataset(low_files, transform)
test_dl = DataLoader(test_ds, batch_size=batch_size)

checkpoint = torch.load(checkpoint_dir+"model_epoch%03d.pth" % (epoch))

model = nn.DataParallel(Net()).cuda()
model.load_state_dict(checkpoint['net'])

model.eval()
with torch.no_grad():
    for test_X, name in test_dl:
        test_X = test_X.cuda()
        test_preds = model(test_X)[0].cpu().numpy()
        for i in range(batch_size):
            im = test_preds[i, :, :, :]
            im = np.swapaxes(np.swapaxes(im, 0, 2), 0, 1) # H * W * C
            im[:,:,[0,2]] = im[:,:,[2,0]] # RGB
            cv2.imwrite('./data/result/'+name[i], im * 255)
        del test_preds
    