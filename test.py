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

low_files = os.listdir('./data/test/')
checkpoint_dir = "./checkpoint/"
epoch = 1
batch_size = 10

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
            
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.pixel_shuffle = nn.PixelShuffle(4)

        self.conv4 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        init.orthogonal_(self.conv0.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight)
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

    def _block(self, x):
        x1 = self.relu(self.conv1(x))
        x1 = (self.relu(self.conv1(x1)) + x) * 0.5
        return x1

    def forward(self, x):
        x = self.relu(self.conv0(x))
        x = self._block(x)
        x = self._block(x)
        x = self._block(x)
        x = self.relu(self.conv1(x))
        x1 = self.conv2(x)
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(x)
        x = self.conv4(x)
        return (x, x1)

test_ds = TestDataset(low_files, transform)
test_dl = DataLoader(test_ds, batch_size=batch_size)

checkpoint = torch.load(checkpoint_dir+"model_epoch%03d.pth" % (epoch))

model = nn.DataParallel(Net()).cuda()
model.load_state_dict(checkpoint['net'])

model.eval()
with torch.no_grad():
    for test_X, name in test_dl:
        test_X = test_X.cuda()
        test_preds, _ = model(test_X).cpu().numpy()
        for i in range(batch_size):
            im = test_preds[i, :, :, :]
            im = np.swapaxes(np.swapaxes(im, 0, 2), 0, 1) # H * W * C
            im[:,:,[0,2]] = im[:,:,[2,0]] # RGB
            cv2.imwrite('./data/result/'+name[i], im * 255)
        del test_preds
    