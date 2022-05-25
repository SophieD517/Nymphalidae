import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m
import os
import torch
import seaborn as sns
import random


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
=        recon = torch.sigmoid(self.dec_lin1(x))
        
        #branch
        preds = torch.nn.functional.relu(self.bran_conv1(ls))
        preds = torch.nn.functional.relu(self.bran_conv2(preds))
        preds = torch.sigmoid(self.bran_lin1(preds))

        return ls, recon, preds

class Abyssinian:
  def __init__(self, dicty):
    self.dicty = dicty
    self.model = Calico(dicty['latent_dim'], dicty['ae_layer'], dicty['branch_layer'], dicty['num_vars'])

  def train(self, x, a, var, modeltype='conv'):
    if modeltype=='conv':
      self.model = ConvCalico(x.shape[1], dicty['latent_dim'], dicty['ae_layer'], dicty['branch_layer'], dicty['num_vars'])
    elif modeltype=='linear'
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

  def test_model(self):
    weights = torch.load(self.file_path)
    self.model.load_state_dict(weights)
    self.model.eval()
