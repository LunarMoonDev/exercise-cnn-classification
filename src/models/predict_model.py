# importing libraries
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ConvolutionalNetwork

# constants
DATA = './data/raw'
MODEL_NAME = './models/model.S.0.0.1667304561.pt'

# prep the data
transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(root = DATA, train = True, download = True, transform = transform)
test_data = datasets.FashionMNIST(root = DATA, train = False, download = True, transform = transform)

class_names = ['T-shirt', 'Trouser', 'Sweater', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

train_loader = DataLoader(train_data, batch_size = 10, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 10, shuffle = True)

# prep the model
torch.manual_seed(101)
model = ConvolutionalNetwork()
model.load_state_dict(torch.load(MODEL_NAME))
criterion = nn.CrossEntropyLoss()

tst_losses = 0
tst_corr = 0

model.eval()
with torch.no_grad():
    for b, (X_test, y_test) in enumerate(test_loader):
        y_val = model(X_test)

        predicted = torch.max(y_val.data, 1)[1]
        tst_corr += (predicted == y_test).sum()

    tst_losses = criterion(y_val, y_test)

print(f'Test accuracy: {tst_corr: 3}/10000 = {tst_corr/100: .3f}%')

