# importing libraries
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from model import ConvolutionalNetwork

# constants
DATA = './data/raw'
VERSION = '0.0'

# prep the data
transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(root = DATA, train = True, download = True, transform = transform)
test_data = datasets.FashionMNIST(root = DATA, train = False, download = True, transform = transform)

class_names = ['T-shirt', 'Trouser', 'Sweater', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

train_loader = DataLoader(train_data, batch_size = 10, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 10, shuffle = True)

# pre-analysis
for images, labels in train_loader:
    break;

print('Label: ', labels.numpy())
print('Class: ', *np.array([class_names[i] for i in labels]))

im = make_grid(images, nrow=10)
plt.figure(figsize = (12, 4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()

# checking the number of parameters of Model
torch.manual_seed(101)
model = ConvolutionalNetwork()

def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item: >6}')
    
    print(f'______\n{sum(params): >6}')

# count_parameters(model)

# defining loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

epochs = 5
start_time = time.time()

train_losses = []
train_correct = []

for i in range(epochs):
    trn_corr = 0

    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 800 == 0:
            print(f'epoch: {i: 2} batch: {b: 4} [{10 * b: 6}/8000] loss: {loss.item(): 10.8f} \ acc: {trn_corr.item() * 100/ (10 * b): 7.3f}')

    train_losses.append(loss.item())
    train_correct.append(trn_corr)

print(f'\n Duration: {time.time() - start_time: .0f} seconds')

# saving the model and analysis
torch.save(model.state_dict(), f'./models/model.S.{VERSION}.{int(time.time())}.pt')
pd.DataFrame(np.array(train_losses)).to_csv('./data/interim/train_losses.csv', header = None, index = False)
pd.DataFrame(np.array(train_correct)).to_csv('./data/interim/train_correct.csv', header = None, index = False)
