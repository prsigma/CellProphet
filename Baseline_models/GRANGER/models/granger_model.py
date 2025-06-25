# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
from models.utils import get_lable_dic,to_causal_arr,filtered_predict_pair
from copy import deepcopy
import torch.optim as optim
from tqdm import tqdm
import time

class NBLoss(nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, x, mean, disp, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        mean = mean * scale_factor
        t1 = torch.lgamma(disp[0][1]+eps) + torch.lgamma(x[0][1]+1.0) - torch.lgamma(x[0][1]+disp[0][1]+eps)
        t2 = (disp[0][1]+x[0][1]) * torch.log(1.0 + (mean/(disp[0][1]+eps))) + (x[0][1] * (torch.log(disp[0][1]+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2
        nb_case = nb_final 
        result =  nb_case      
        if ridge_lambda > 0:
            ridge = ridge_lambda
            result += ridge       
        result = torch.mean(result)
        return result
class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class GRU(nn.Module):
    def __init__(self, num_series, hidden):

        super(GRU, self).__init__()
        self.p = num_series
        self.hidden = hidden
        # Set up network.
        self.gru = nn.GRU(num_series, hidden, batch_first=True)
        self.gru.flatten_parameters()
        self.relu=nn.ReLU()
        self.linear = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch):
        #Initialize hidden states
        device = self.gru.weight_ih_l0.device
        return torch.zeros(1, batch, self.hidden, device=device)
               
    def forward(self, X, z, connection, mode = 'train'):
        
        X=X[:,:,np.where(connection!=0)[0]]
        device = self.gru.weight_ih_l0.device
        tau = 0
        if mode == 'train':
          X_right, hidden_out = self.gru(torch.cat((X[:,0:1,:],X[:,11:-1,:]),1), z)
          X_right=self.relu(X_right)
          X_right = self.linear(X_right)
          return X_right, hidden_out
          
class GRANGER(nn.Module):
    def __init__(self, x_np, num_series, connection, hidden, device=None):
        super(GRANGER, self).__init__()
        # 使用传入的device，如果没有传入则使用cuda:0（如果可用）
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"GRANGER模型使用设备: {self.device}")
        
        self.p = num_series
        self.hidden = hidden
        self.gru_left = nn.GRU(num_series, hidden, batch_first=True,bidirectional=False)
        self.gru_left.flatten_parameters()     
        self.fc_mu = nn.Linear(hidden, hidden)
        self.fc_std = nn.Linear(hidden, hidden)
        self.connection = connection
        self.gru_1 = nn.GRU(256,256,bidirectional=False)#double layers
        self._dec_mean = torch.nn.Sequential(torch.nn.Linear(hidden,x_np.shape[1]), MeanAct())
        self._dec_disp = torch.nn.Sequential(torch.nn.Linear(hidden,x_np.shape[1]), DispAct())
        self.nb_loss=NBLoss().to(self.device)
        # Set up networks.
        self.networks = nn.ModuleList([
            GRU(int(connection[:,i].sum()), hidden) for i in range(num_series)])
        

    def forward(self, X, noise = None, mode = 'train', phase = 0):
        # 确保输入tensor在正确的设备上
        if X.device != self.device:
            X = X.to(self.device)

        if phase == 0:
            X = torch.cat((torch.zeros(X.shape, device=self.device)[:,0:1,:], X), 1)
            if mode == 'train':
                hidden_0 = torch.zeros(1, X.shape[0], self.hidden, device=self.device)
                out, h_t = self.gru_left(X[:,1:11,:], hidden_0.detach()) # 1 32 256
                out,h_t = self.gru_1(h_t,hidden_0.detach())
                mu = self.fc_mu(h_t)
                log_var = self.fc_std(h_t)
                _mean=self._dec_mean(h_t)
                _disp=self._dec_disp(h_t)
                sigma = torch.exp(0.5*log_var)
                z = torch.randn(size = mu.size(), device=self.device)
                z = z.type_as(mu)
                z = mu + sigma*z
                pred = [self.networks[i](X, z, self.connection[:,i])[0]
                      for i in range(self.p)]
                return pred, log_var, mu,_mean,_disp       
            if mode == 'test':
                X_seq = torch.zeros(X[:,:1,:].shape, device=self.device)
                h_0 = torch.randn(size = (1, X_seq[:,-2:-1,:].size(0),self.hidden), device=self.device)
                ht_last =[]
                for i in range(self.p):
                    ht_last.append(h_0)
                for i in range(int(20/1)+1):
                    ht_new = []
                    for j in range(self.p):
                        # out, h_t = self.gru_out[j](X_seq[:,-1:,:], ht_last[j])
                        # out = self.fc[j](out)
                        out, h_t = self.networks[j](X_seq[:,-1:,:], ht_last[j], self.connection[:,j])
                        if j == 0:
                            X_t = out
                        else:
                            X_t = torch.cat((X_t,out),-1)
                        ht_new.append(h_t)
                    ht_last = ht_new
                    if i ==0:
                        X_seq = X_t
                    else:
                        X_seq = torch.cat([X_seq,X_t],dim = 1)           
                return X_seq
            
        if phase == 1:
            X = torch.cat((torch.zeros(X.shape, device=self.device)[:,0:1,:], X), 1)
            if mode == 'train':
                hidden_0 = torch.zeros(1, X.shape[0], self.hidden, device=self.device)
                out, h_t = self.gru_left(X[:,1:11,:], hidden_0.detach())
                mu = self.fc_mu(h_t)
                log_var = self.fc_std(h_t)
                _mean=self._dec_mean(h_t)
                _disp=self._dec_disp(h_t)
                sigma = torch.exp(0.5*log_var)
                z = torch.randn(size = mu.size(), device=self.device)
                z = z.type_as(mu) 
                z = mu + sigma*z
                pred = [self.networks[i](X, z, self.connection[:,i])[0]
                      for i in range(self.p)]
                return pred, log_var, mu,_mean,_disp
            if mode == 'test':
                X_seq = torch.zeros(X[:,:1,:].shape, device=self.device)
                h_0 = torch.randn(size = (1, X_seq[:,-2:-1,:].size(0),self.hidden), device=self.device)
                ht_last =[]
                for i in range(self.p):
                    ht_last.append(h_0)
                for i in range(int(20/1)+1):
                    ht_new = []
                    for j in range(self.p):
                        # out, h_t = self.gru_out[j](X_seq[:,-1:,:], ht_last[j])
                        # out = self.fc[j](out)
                        out, h_t = self.networks[j](X_seq[:,-1:,:], ht_last[j], self.connection[:,j])
                        if j == 0:
                            X_t = out
                        else:
                            X_t = torch.cat((X_t,out),-1)
                        ht_new.append(h_t)
                    ht_last = ht_new
                    if i ==0:
                        X_seq = X_t + 0.1*noise[:,i:i+1,:] 
                    else:
                        X_seq = torch.cat([X_seq,X_t+0.1*noise[:,i:i+1,:]],dim = 1)
                return X_seq

    def GC(self, threshold=True):
        # to learn Granger causality and return a GC matrix
        # GC = [torch.norm(net.gru.weight_ih_l0,dim=0) 
        #       for net in self.networks]
        GC = [torch.norm(net.gru.weight_ih_l0, p=1,dim=0) 
              for net in self.networks]
        GC = torch.stack(GC)
        # print(GC)
        
        # 注释掉以下代码，因为使用的是unlabeled data（没有refNetwork.csv）
        # label_pair= get_lable_dic('example_data/mCAD-2000-1/refNetwork.csv')#format label file,input it's path
        # predict_pair=None
        # if threshold:
        #     np.savetxt('GC_cell.csv', GC.detach().cpu().numpy(), delimiter=',',fmt='%.20f')#GC matrix file path
        #     predict_pair = to_causal_arr('GC_cell.csv','example_data/mCAD-2000-1/ExpressionData.csv')    #ExpressionData file path
        #     filtered_predict_pair(label_pair,predict_pair)
        #     return (torch.abs(GC) > 0).int() 
        # else:
        #     return GC
        
        if threshold:
            return (torch.abs(GC) > 0).int() 
        else:
            return GC

def prox_update(network, lam, lr):
    W = network.gru.weight_ih_l0
    norm = torch.norm(W, dim=0, keepdim=True)
    W.data = ((W / torch.clamp(norm, min=(lam * lr)))
              * torch.clamp(norm - (lr * lam), min=0.0))
    network.gru.flatten_parameters()

def regularize(network, lam):
    W = network.gru.weight_ih_l0
    return lam * torch.sum(torch.norm(W, dim=0))

def ridge_regularize(network, lam):
    return lam * (
        torch.sum(network.linear.weight ** 2) +
        torch.sum(network.gru.weight_hh_l0 ** 2))

def restore_parameters(model, best_model):
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params

def arrange_input(data, context):
    #Arrange a single time series into overlapping short sequences.data: time series of shape (T, dim).context: length of short sequences.
    assert context >= 1 and isinstance(context, int)
    input = torch.zeros(len(data) - context, context, data.shape[1],
                        dtype=torch.float32, device=data.device)
    target = torch.zeros(len(data) - context, context, data.shape[1],
                         dtype=torch.float32, device=data.device)
    for i in range(context):
        start = i
        end = len(data) - context + i
        input[:, i, :] = data[start:end]
        target[:, i, :] = data[start+1:end+1]
    return input.detach(), target.detach()

def MinMaxScaler(data):
  min_val = np.min(np.min(data, axis = 0), axis = 0)
  data = data - min_val
    
  max_val = np.max(np.max(data, axis = 0), axis = 0)
  norm_data = data / (max_val + 1e-7)
    
  return norm_data
def train_phase1(granger, X, context, lr, max_iter, lam=0, lam_ridge=0,
                     lookback=5, check_every=10, verbose=1,sparsity = 100, batch_size = 256):
    #Train model with Adam
    des=1        #hyper-paremeters of NB
    p = X.shape[-1]
    device = granger.networks[0].gru.weight_ih_l0.device
    loss_fn = nn.CrossEntropyLoss()
    train_loss_list = []  #Store the loss value during the training process
    batch_size = batch_size
    # Set up data.4
    X, Y = zip(*[arrange_input(x, context) for x in X]) 
    X_all = torch.cat(X, dim=0)
    Y_all = torch.cat(Y, dim=0)
    idx = np.random.randint(len(X_all), size=(batch_size,))
    X = X_all[idx]
    Y = Y_all[idx]
    start_point = 0
    beta = 0.1
    # For early stopping.
    best_it = None
    best_loss = np.inf
    best_model = None
    # Calculate granger error.
    pred,mu,log_var,_mean,_disp = granger(X)
    loss_nb = granger.nb_loss(X,_mean, _disp)
    loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])
    mmd = (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = -1).sum(dim = 0)).mean(dim =0) 
    ridge = sum([ridge_regularize(net, lam_ridge) for net in granger.networks])
    smooth = loss + ridge + beta*mmd+des*loss_nb
    best_mmd = np.inf               
        
    for it in tqdm(range(max_iter), desc="训练进度", total=max_iter):
        smooth.backward()
        for param in granger.parameters():
            param.data -= lr * param.grad

        if lam > 0:
            for net in granger.networks:
                prox_update(net, lam, lr)
        
        granger.zero_grad()
        pred,mu,log_var,_mean,_disp = granger(X)
        loss = sum([loss_fn(pred[i][:, :, 0], X[:, 10:, i]) for i in range(p)])
        loss_nb = granger.nb_loss(X,_mean, _disp)
        mmd = (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = -1).sum(dim = 0)).mean(dim =0)   
        ridge = sum([ridge_regularize(net, lam_ridge)
                     for net in granger.networks])
        smooth = loss + ridge + beta*mmd+des*loss_nb
        
        # Check progress.
        if (it) % check_every == 0:     
            X_t = X
            Y_t = Y
            pred_t,mu_t ,log_var_t,_mean_t,_disp_t= granger(X_t)
            loss_nb_t = granger.nb_loss(X,_mean_t, _disp_t)
            sum_loss_nb=sum(loss_nb_t for i in range (p))
            loss_t = sum([loss_fn(pred_t[i][:, :, 0], X_t[:, 10:, i]) for i in range(p)])
            mmd_t = (-0.5*(1+log_var_t - mu_t**2- torch.exp(log_var_t)).sum(dim = -1).sum(dim = 0)).mean(dim =0)   
            ridge_t = sum([ridge_regularize(net, lam_ridge)
                     for net in granger.networks])
            smooth_t = loss_t + ridge_t+des*loss_nb_t  
            nonsmooth = sum([regularize(net, lam) for net in granger.networks])
            mean_loss = (smooth_t) / p
            
            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it ))
                if lam>0:
                   print('Variable usage = %.2f%%'
                        % (100 * torch.mean(granger.GC().float())))

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_it = it
                best_model = deepcopy(granger)
            start_point = 0
            predicted_data = granger(X_t,mode = 'test')
            syn = predicted_data[:,:-1,:].cpu().detach().numpy()
            ori= X_t[:,start_point:,:].cpu().detach().numpy()
            syn = MinMaxScaler(syn)
            ori = MinMaxScaler(ori)
    
    # Restore best model.
    restore_parameters(granger, best_model)
    return train_loss_list