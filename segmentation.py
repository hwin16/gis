import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import os

class VGG16(nn.Module): 
    def __init__(self):

        super().__init__()

        # layer 1
        self.conv1_1 = nn.Conv2d(6, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        # layer 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        # layer 3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 1)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        # layer 4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 1)
        self.pool4 = nn.MaxPool2d(2, stride=2)

        # layer 5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 1)
        self.pool5 = nn.MaxPool2d(2, stride=2)

        # layer 6
        self.convt6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.convt7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.convt8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.convt9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.convt10 = nn.ConvTranspose2d(32, 2, 2, stride=2)


    def forward(self, x):
        x = x.cuda()

        # layer 1, input: 3x224x224, output: 64x112x112
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        # layer 2, input: 64x112x112, output: 128x56x56
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        # layer 3, input: 128x56x56, output: 256x28x28
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        # layer 4, input: 256x28x28, output: 512x14x14 
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool4(x)

        # layer 5, input: 512x14x14, output: 512x7x7
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.pool5(x)

        # layer 6
        x = F.relu(self.convt6(x))
        x = F.relu(self.convt7(x))
        x = F.relu(self.convt8(x))
        x = F.relu(self.convt9(x))
        x = F.relu(self.convt10(x))

        return x

class ChampaignRaster(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        self.X = os.listdir('../data/vgg16/X')
        self.y = os.listdir('../data/vgg16/y')
        size = int(len(self.X) * 0.7)
        
        if split == 'train':
            self.X = self.X[0:size]
            self.y = self.y[0:size]
        else:
            self.X = self.X[size:len(self.X)]
            self.y = self.y[size:len(self.y)]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        x = self.X[i]
        rst_x = rio.open('../data/vgg16/X/' + x)
        band_x = rst_x.read()
        tensor_x = torch.from_numpy(band_x).cuda()
        tensor_x = tensor_x.type(torch.cuda.FloatTensor)
        
        y = self.y[i]
        rst_y = rio.open('../data/vgg16/y/' + y)
        band_y = rst_y.read()
        tensor_y = torch.from_numpy(band_y).cuda()
        tensor_y = tensor_y.type(torch.cuda.LongTensor)
        return tensor_x, tensor_y


epochs = 3
lr = 0.01
momentum = 0.05
batch_size = 10

train_dataset = ChampaignRaster()
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

network = VGG16().cuda()
optimizer = optim.SGD(network.parameters(), lr=lr, momentum=momentum)
crit = nn.NLLLoss()

print(len(train_loader))

# train
def train():
    network.train()
    for idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = F.log_softmax(network(data), dim=1)
        target = target.view(-1, 224, 224)
        loss = crit(output, target)
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print(f'Loss: {loss.item()}')

train()
