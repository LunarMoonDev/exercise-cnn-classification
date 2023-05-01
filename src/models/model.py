# importing libraries
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNetwork(nn.Module):
    '''
        basic model for implementing CNN with two conv layers
        and hard coded parameters
    '''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 24, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 24, 120)
        self.fc2 = nn.Linear(120, 10)
    
    def forward(self, X):
        '''
            connects the layers in __init__ with given input X

            @param X, 28 x 28 x 3 image
        '''

        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 4 * 4 * 24)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        
        return F.log_softmax(X, dim = 1)
