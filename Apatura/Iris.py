import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import seaborn as sns
import plotly.express as px
from Danaus import Plexippus
from tools import FastTensorDataLoader


class WorkUnit:
  def __init__(self, model, learning_rate, latent_dim, ae_layer1, branch_layer, ae_layer2=10):
    self.loss_func = torch.nn.MSELoss()
    if model=='FC':
      self.model = Plexippus.FCMulti(latent_dim, ae_layer, branch_layer, 1)
    elif model=='Conv':
      self.model = Plexippus.ConvMulti(latent_dim, ae_layer1, ae_layer2, branch_layer, 1)
    self.learning_rate, self.num_branches = learning_rate, num_branches
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    self.losses = []

  def train(self, x, branches, epochs, batch_size, weights='hi', graph_loss=True,save_ls=False): 
    self.batch_size = batch_size
    if weights=='hi':
      self.weights = np.full(shape=branches.shape[1], fill_value=1,dtype=int)
    assert branches.shape[1]=weights.shape[0], 'weights of wrong size'
    train_batches = FastTensorDataLoader(x, branches, batch_size=batch_size, shuffle=False)
    for epoch in range(epochs):
      for idx, batch in enumerate(train_batches):
        self.optimizer.zero_grad()
        try:
          latent_space, reconstructed, preds = self.model(batch[0])
          loss = self.loss_func(reconstructed.squeeze(), batch[0].squeeze())
          for i in range(self.num_branches):
            loss.append(weights[i]*self.loss_func(preds[i].squeeze(), batch[i].squeeze())
        except Indexerror:
          self.model.num_branches = branches.shape[1]
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.losses.append(loss.detach().numpy())
      self.epochs = epoch + 1
      if graph_loss==True:
        self.graph_loss(y_lim=.01, xlim=(epoch-10))
  
  def graph_loss(self, y_lim=1, xlim=0, linewidth=0.5):
    plt.style.use('seaborn-whitegrid')
    plt.xlabel('Iterations')
    y1 = self.losses
    x = np.arange(0, self.epochs, 1/(len(y1)/self.epochs))
    plt.plot(x, y1, color='green', linewidth=linewidth)
    plt.ylim(0, y_lim)
    plt.xlim(xlim)
    plt.show()


  def save(self, file_path):
    torch.save(self.model.state_dict(), file_path)
