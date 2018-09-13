import pandas as pd
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch import device, from_numpy, cat
import torch
from warnings import warn
import pdb

device = device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Preprocessing:
    """ Class for preprocessing the data for the Google Trends notebooks. Allows the user to 
    normalize the data with minmax.
    """
    def __init__(self, data):
        self.data = data
        self.functions = {'minmax': self.minmax}
    
    def prep(self, previous, future, enrichMode=None, normalize=None):
        """ This function assumes index is dates
        Allowed enrich modes: one-hot, embedding (Used with seasons)
        Allowed normalizations: minmax 
        """
        if enrichMode =='embedding':
            data = self.data.copy()
            data.insert(loc=1, column='season', value = [self.seasons(ind) for ind in data.index])
            data = data.as_matrix().astype(float)
            enriched = 1
        elif enrichMode =='one-hot':
            data = self.data.copy()
            data.insert(loc=1, column='season', value = [self.seasons(ind) for ind in data.index])
            data = self.oneHot(data).as_matrix().astype(float)
            enriched = 4
        else:
            data = self.data.as_matrix().astype(float).copy()
            enriched = 0

        if normalize:
            data[:, 0] = self.functions[normalize](data[:,0])
        
        rows = data.shape[0]

        inp = np.zeros(shape=(rows, previous+1+enriched))
        for i in range(0, previous+1):
            inp[list(range(rows-i)),i] = data[i:,0].reshape(-1,rows-i)
        
        if enriched > 0:
            for i in range(enriched):
                inp[:,previous+1+i] = data[:,i+1] 

        out = np.zeros(shape=(rows, future))
        out[list(range(rows-previous-future)), 0] = data[(previous+future):,0].reshape(-1)
            
        inp = inp[:-(i+future)]
        out = out[:-(i+future)]

        raw = [(from_numpy(inp[j]), from_numpy(out[j])) for j in range(inp.shape[0])]
        return raw

    def minmax(self, data):
        maximum = max(data)
        minimum = min(data)
        return (data - minimum)/ (maximum - minimum)

    def seasons(self, date):
        """ Helper function to encode seasons. 0 is winter, 1 is spring, 2 is summer and 3 is fall.
        """
        seasons = {12: 0, 1: 0, 2: 0,
            3: 1, 4: 1, 5: 1,
            6:2, 7:2, 8:2,
            9:3, 10:3, 11:3}
        
        return(seasons[date.to_pydatetime().month])

    def oneHot(self, data):
        rows = data.shape[0]
        results = np.zeros((rows, 4))
        for idx, i in enumerate(data.season.values):
            results[idx, i] = 1

        data.insert(loc=2, column='winter', value = results[:,0])
        data.insert(loc=3, column='spring', value = results[:,1])
        data.insert(loc=4, column='summer', value = results[:,2])
        data.insert(loc=5, column='fall', value = results[:,3])
        data = data.drop(['season'], axis=1)
        return data

class Processing:
    """ Class which containts the functions for training and testing a model.
    """
    def __init__(self, model, epochs, opt, loss):
        self.model = model.to(device)
        self.num_epochs = epochs
        self.opt = opt
        self.loss = loss

    def trainWithVisual(self, train_loader):
        """Trains the model while retaining the model's error in each epoch. This function is used as a diagnostic tool while training the model 
        """
        idx = 0
        self.model.train()
        results = pd.DataFrame(np.nan, index= list(range(self.num_epochs)), columns=['epoch', 'err'])
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for i, (x, y) in enumerate(train_loader):
                x = x.float().to(device)
                y = y.float().to(device)

                # Forward pass
                output = self.model(x)
                l = self.loss(output, y)
                running_loss += l.item()

                # Backward and optimize
                self.opt.zero_grad()
                l.backward()
                self.opt.step()
            
            results.iloc[idx] = [epoch, running_loss/(i+1)]
            idx += 1
        return results

    def train(self, train_loader):
        self.model.train()
        for _ in range(self.num_epochs):
            for i, (x, y) in enumerate(train_loader):
                x = x.float().to(device)
                y = y.float().to(device)

                # Forward pass
                output = self.model(x)
                l = self.loss(output, y)

                # Backward and optimize
                self.opt.zero_grad()
                l.backward()
                self.opt.step()

    def test(self, test_loader):
        idx = 0
        results = pd.DataFrame(np.nan, index= list(range(len(test_loader))), columns=['actual', 'pred'])
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for i, (x, y) in enumerate(test_loader):
                x = x.float().to(device)
                y = y.float().to(device)

                # Forward pass
                output = self.model(x)
                l = self.loss(output, y)
                
                running_loss += l.item()
                
                results.iloc[idx,0] = y.numpy()
                results.iloc[idx,1] = output.numpy()
                idx += 1
                
        return results, running_loss/(i+1)


class FCNet(nn.Module):
    """ A Fully connected network  
    """
    def __init__(self, units, activation=F.relu):
        super().__init__()
        self.act = activation
        self.lins = nn.ModuleList([nn.Linear(units[x], units[x+1]) 
                                   for x,_ in enumerate(units) if x < len(units)-2])
        self.out = nn.Linear(units[-2], units[-1])
        
    def forward(self, x):
        for l in self.lins: x = self.act(l(x))
        return self.out(x)

class ConvNet(nn.Module):
    """ Convolutional Neural Network with a convolutional operator at the head and Fully connected layers in the rest.

    The formulas for convOut and start are taken from https://pytorch.org/docs/stable/nn.html (L_out) and might require 
    changing if we use padding in the convolution operator OR in the average pool operator AND dilation in the 
    convolution operator.
    """
    def __init__(self, units, convKernel, poolKernel, convChannels=7, activation=F.relu, convStride=1, poolStride=1):
        convOut = int((units[0] - (convKernel - 1) - 1) / convStride + 1)
        if poolKernel > convOut:
            raise ValueError('Pool kernel size should be smaller then that of the Convolution')
        if poolKernel + poolStride > convOut+1:
            warn('All of the information will not be retained after the pooling function is applied.')
        
        super().__init__()
        self.conv = nn.Conv1d(1, convChannels, kernel_size=convKernel, stride=convStride)
        self.pool = nn.AvgPool1d(kernel_size= poolKernel, stride=poolStride)
        
        if convOut-poolKernel == 0:
            start = 1 * convChannels
        else:
            start = int((convOut-poolKernel)/poolStride + 1) * convChannels
        self.layers = nn.ModuleList([nn.Linear(start, units[1])])

        for i in range(1, len(units)-2):
            self.layers.append(nn.Linear(units[i], units[i+1]))
        self.out = nn.Linear(units[-2], units[-1])       
        self.act = activation
        
    def forward(self, x):
        x = self.pool(self.conv(x.unsqueeze(1))).view(x.shape[0],-1)
        for l in self.layers: x = self.act(l(x))
        return self.out(x)

class FCNetEmbed(nn.Module):
    def __init__(self, units, cont_idx, cat_idx, embed_detail, activation=F.relu):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, s) for c,s in embed_detail])
        # initilization based on fast.ai code
        for emb in self.embs: 
            x = emb.weight.data
            sc = 2/(x.size(1)+1)
            x.uniform_(-sc, sc)
    
        self.lins = nn.ModuleList([nn.Linear(units[x], units[x+1]) 
                                   for x,_ in enumerate(units) if x < len(units)-2])
        self.out = nn.Linear(units[-2], units[-1])
        self.cont_idx = cont_idx
        self.cat_idx = cat_idx
        self.act = activation
                
    def forward(self, x):
        x_ = x[:,self.cont_idx]
        x_cat = x[:,self.cat_idx]
        
        embs = (e(x_cat[:,i].type(torch.long)) for i,e in enumerate(self.embs))
        for e in embs: x_ = cat([x_, e], 1)
        
        for l in self.lins: x_ = self.act(l(x_))
        
        return self.out(x_)
