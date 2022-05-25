import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m
import os
import torch
import seaborn as sns
import random

class DL:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class LinCalico(torch.nn.Module):
    def __init__(self, latent_dim, ae_layer, branch_layer, start_dim, end_dim):
        super().__init__()
 
        self.enc1 = torch.nn.Linear(in_features=start_dim, out_features=ae_layer)
        self.enc2 = torch.nn.Linear(in_features=ae_layer, out_features=latent_dim*2)
 
        self.dec1 = torch.nn.Linear(in_features=latent_dim, out_features=ae_layer)
        self.dec2 = torch.nn.Linear(in_features=ae_layer, out_features=start_dim)
        self.latent_dim = latent_dim

        self.branch1 = torch.nn.Linear(in_features=latent_dim, out_features=branch_layer)
        self.branch2 = torch.nn.Linear(in_features=branch_layer, out_features=end_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        epsilon = torch.randn_like(std)
        sample = mu + (epsilon * std) 
        return sample
 
    def forward(self, x):
        #encoder
        x = torch.nn.functional.relu(self.enc1(x))
        x = self.enc2(x)
        x = x.view(-1, 2, self.latent_dim)

        #stats
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        ls = self.reparameterize(mu, log_var)
 
        # decoding
        x = torch.nn.functional.relu(self.dec1(ls))
        reconstruction = torch.sigmoid(self.dec2(x))
        
        #branch
        preds = torch.nn.functional.relu(self.branch1(ls))
        preds = torch.sigmoid(self.branch2(preds))

        return ls, reconstruction, preds
      
class ConvCalico(torch.nn.Module):
    def __init__(self, latent_dim, ae_layer, branch_layer1, branch_layer2, start_dim, end_dim, branch='conv'):
        super().__init__()
        
        
        self.enc_conv1 = torch.nn.Conv1d(in_channels=1, out_channels=ae_layer1, kernel_size=start_dim)
        self.enc_conv2 = torch.nn.Conv1d(in_channels=ae_layer1, out_channels=ae_layer2, kernel_size=1)
        self.enc_lin1 = torch.nn.Linear(ae_layer2, latent_dim)
        
        self.dec_conv1 = torch.nn.Conv1d(in_channels=1, out_channels=ae_layer2, kernel_size=latent_dim, padding=0)
        self.dec_conv2 = torch.nn.Conv1d(in_channels=ae_layer2, out_channels=ae_layer1, kernel_size=1)
        self.dec_lin1 = torch.nn.Linear(ae_layer1, start_dim)
        
        self.latent_dim = latent_dim
        
        if branch=='conv':
          self.bran_conv1 = torch.nn.Conv1d(in_channels=1, out_channels=branch_layer1, kernel_size=latent_dim, padding=0)
          self.bran_conv2 = torch.nn.Conv1d(in_channels=branch_layer1, out_channels=branch_layer2, kernel_size=1)
          self.bran_lin1 = torch.nn.Linear(branch_layer2, end_dim)
        elif branch=='linear':
          self.branch1 = torch.nn.Linear(in_features=latent_dim, out_features=branch_layer)
          self.branch2 = torch.nn.Linear(in_features=branch_layer, out_features=end_dim)
          

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        epsilon = torch.randn_like(std)
        sample = mu + (epsilon * std) 
        return sample
 
    def forward(self, x):
        x = torch.nn.functional.relu(self.enc_conv1(x))
        x = torch.nn.functional.relu(self.enc_conv2(x))
        x = self.enc_lin1(x)
        x = x.view(-1, 2, self.latent_dim)

        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance

        ls = self.reparameterize(mu, log_var)
 
        # decoding
        recon = torch.nn.functional.relu(self.dec_conv1(ls))
        recon = torch.nn.functional.relu(self.dec_conv2(recon))
        recon = torch.sigmoid(self.dec_lin1(x))
        
        #branch
        preds = torch.nn.functional.relu(self.bran_conv1(ls))
        preds = torch.nn.functional.relu(self.bran_conv2(preds))
        preds = torch.sigmoid(self.bran_lin1(preds))

        return ls, recon, preds

class Abyssinian:
  def __init__(self, dicty):
    self.dicty = dicty

  def train(self, x, a, var, modeltype='conv'):
    if modeltype=='conv':
      self.model = ConvCalico(x.shape[1], dicty['latent_dim'], dicty['ae_layer'], dicty['branch_layer'], dicty['num_vars'])
    elif modeltype=='linear':
      self.model = LinCalico(x.shape[1], dicty['latent_dim'], dicty['ae_layer'], dicty['branch_layer'], dicty['num_vars'])
    self.var = var
    wandb.log({'vars': var})
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.dicty['learning_rate'])
    train_batches = DL(x, a, batch_size=self.dicty['batch_size'], shuffle=False)
    wandb.log(self.dicty)
    for epoch in range(self.dicty['epochs']):
      for idx, batch in enumerate(train_batches):
        optimizer.zero_grad()
        ls, recon, pred = self.model(batch[0])
        recon_loss = torch.nn.MSELoss()(recon, batch[0])
        branch_loss = torch.nn.MSELoss()(pred.squeeze(), batch[1].squeeze())
        loss = recon_loss + self.dicty['weight']*branch_loss
        loss.backward(retain_graph=True)
        optimizer.step()
        wandb.log({'loss': loss, 'recon_loss': recon_loss, 'branch_loss': branch_loss})

  def save(self):
    self.file_path = 'model_' + self.var + '_' + self.dicty['latent_dim']
    torch.save(self.model.state_dict(), self.file_path)

  def test_model(self, x, a):
    weights = torch.load(self.file_path)
    self.model.load_state_dict(weights)
    self.model.eval()
    self.ls, self.recon, self.pred = self.model(x)
    self.x_test = x
    loss = torch.nn.MSELoss()(self.recon, x) + torch.nn.MSELoss()(self.pred.squeeze(), a.squeeze())
  
  def loss_hist(self):
    losses = []
    dataloader = DL(self.recon, self.x_test, batch_size=1, shuffle=False)
    for idx, row in enumerate(dataloader):
      losses.append(self.loss_func(row))
    fig, ax = plt.subplots()
    sns.histplot(data=losses, ax=ax, binwidth=.000003)
    self.losses = losses
    #ax.set_xlim(0, .0001)
    plt.show()

  def graph_correlation(self, t, var, width=.03):
    plt.style.use('seaborn-whitegrid')
    if var == 'a':
      dt = t['A']
      pred = self.pred[:, :1]
    elif var == 'b':
      dt = t['B']
      pred = self.pred[:, 1:2]
    elif var == 'h':
      dt = t['H']
      pred = self.pred[:, 2:3]
    elif var == 'k':
      dt = t['K']
      pred = self.pred[:, 3:4]
    plt.plot(dt.detach().numpy().reshape(-1,), dt.detach().numpy().reshape(-1,), color='gray', linewidth=0.1)
    plt.scatter(pred.detach().numpy().reshape(-1,), dt.detach().numpy().reshape(-1,), color='purple', s=width)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.title(var + '-value')

  def graph_recon(self, sam):
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(1, 1)
    x = [x for x in np.arange(-euler, euler, euler/50)]
    print(x, self.recon)
    colors = ['mistyrose', '#ccffb3', '#ADD8E6']
    a = 0
    for i in sam:
      plt.plot(x, self.x_test.detach().numpy().tolist()[i], color=colors[a], linewidth=5)
      a+=1
    # plt.xlabel('time')
    # plt.ylabel('sample remaining')
    colors = ['salmon', 'darkolivegreen', 'darkcyan']
    a = 0
    for i in sam:
      plt.plot(x, self.recon.detach().numpy().tolist()[i], '--', color=colors[a], linewidth=1)
      a+=1
    fig.set_size_inches(12, 8)
    plt.ylim(-1, 1)
    plt.xlim(-euler, euler-euler/50)
    plt.show()

  def graph_latent_space(self, t, latent_dim, num_branches):
    dims = []
    for dim in range(latent_dim):
      dims.append([i[dim] for i in self.ls.detach().numpy()])

    if num_branches == 1:
      if latent_dim==1:
        fig, axs = plt.subplots(1, 1)
        axs.scatter(t['A'].detach().numpy(), dims[0], s=0.1, color='blue')
        axs.set(xlabel='a-value', ylabel='latent_space')
      else:
        fig, axs = plt.subplots(1, latent_dim)
        for dim in range(latent_dim):
          axs[dim].scatter(t['A'].detach().numpy(), dims[dim], s=0.1, color='blue')
          axs[dim].set(xlabel='a-value', ylabel='latent_space')
    else:
      fig, axs = plt.subplots(num_branches, latent_dim)
      for dim in range(latent_dim):
        for bra in range(num_branches):
          axs[dim, bra].scatter(t['A'].detach().numpy(), dims[dim], s=0.1, color='purple')
          axs[dim, bra].set(xlabel=str(dim) + 'value', ylabel='latent_space')
    fig.set_size_inches(latent_dim*5, num_branches*5)
    plt.style.use('seaborn-whitegrid')
