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
import torch.nn.init as init

class TrainDataset(Dataset):
    def __init__(self, image_paths, transform):
        super().__init__()
        self.paths = image_paths
        self.len = len(self.paths)
        self.transform = transform
        
    def __len__(self): return self.len
    
    def __getitem__(self, index): 
        low_path = self.paths[index]
        low = Image.open(low_path).convert('RGB') # 960 * 540
        low = self.transform(low)
        high_path = "./data/high/" + low_path.split('/')[-1]
        high = Image.open(high_path).convert('RGB') # 3840 * 2160
        high = self.transform(high)
        high2low_path = "./data/high2low/" + low_path.split('/')[-1]
        high2low = Image.open(high2low_path).convert('RGB')
        high2low = self.transform(high2low)
        return (low, high, high2low)

transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))
])

class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out, stride):
        super(ResBlk, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
        )

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
            )

    def forward(self, input):
        out = self.block(input)+self.extra(input)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Sequential(
            ResBlk(32, 32, 1),
            ResBlk(32, 32, 1),
            ResBlk(32, 64, 1),
            ResBlk(64, 64, 1),
            ResBlk(64, 64, 1),
            ResBlk(64, 64, 1),
            ResBlk(64, 64, 1),
            ResBlk(64, 64, 1)
        )

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
            
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.pixel_shuffle = nn.PixelShuffle(4)

        self.conv4 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = self.relu(self.conv0(x))
        x = self.conv1(x)
        x1 = self.conv2(x)
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(x)
        x = self.conv4(x)
        return (x, x1)

if __name__ == "__main__":

    low_files = os.listdir('./data/low/')
    checkpoint_dir = "./checkpoint/"

    def train_path(p): return f"./data/low/{p}"
    low_files = list(map(train_path, low_files))

    model = nn.DataParallel(Net()).cuda()
    #model = Net().cuda()
    losses = []
    epoches = 5
    start = time.time()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    start_epoch = 2
    if start_epoch > 0:
        checkpoint = torch.load(checkpoint_dir+"model_epoch%03d.pth"%(start_epoch-1))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch, epoches):
        random.shuffle(low_files)

        train_files = low_files[:10000]
        valid = low_files[10000:15000]

        train_ds = TrainDataset(train_files, transform)
        train_dl = DataLoader(train_ds, batch_size=5)

        valid_ds = TrainDataset(valid, transform)
        valid_dl = DataLoader(valid_ds, batch_size=5)

        epoch_loss = 0
        model.train()
        for X, y, m in train_dl:
            X = X.cuda()
            y = y.cuda()
            m = m.cuda()
            preds, low = model(X)
            loss = loss_fn(preds, y) + loss_fn(low, m)
            del preds
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss
            print('.', end='', flush=True)
        del X, y, m
            
        epoch_loss = epoch_loss / len(train_dl)
        print("\nEpoch: {}, train loss: {:.6f}, time: {}".format(epoch, epoch_loss, time.time() - start), flush=True)
        
        model.eval()
        with torch.no_grad():
            val_epoch_loss = 0
            for val_X, val_y, val_m in valid_dl:
                val_X = val_X.cuda()
                val_y = val_y.cuda()
                val_m = val_m.cuda()
                val_preds, val_low = model(val_X)
                val_loss = loss_fn(val_preds, val_y) + loss_fn(val_low, val_m)
                del val_preds

                val_epoch_loss += val_loss      
            del val_X, val_y, val_m

            val_epoch_loss = val_epoch_loss / len(valid_dl)
            print("Epoch: {}, valid loss: {:.6f}, time: {}\n".format(epoch, val_epoch_loss, time.time() - start), flush=True)
        
        state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        torch.save(state, checkpoint_dir+"model_epoch%03d.pth"%(epoch))
        del state

    '''
    checkpoint = torch.load(dir)

    model.load_state_dict(checkpoint['net'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    start_epoch = checkpoint['epoch'] + 1
    '''