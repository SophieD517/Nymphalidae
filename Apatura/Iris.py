# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import torch
# import seaborn as sns
# import plotly.express as px
# from Nymphalidae.Danaus import Plexippus
# from Nymphalidae.tools import FastTensorDataLoader
# !pip install wandb
# import wandb


# class WorkUnit:
#   def __init__(self, model, learning_rate, latent_dim, ae_layer1, branch_layer, ae_layer2=10, wandb=False):
#     self.loss_func = torch.nn.MSELoss()
#     if model=='FC':
#       self.model = Plexippus.FCMulti(latent_dim, ae_layer1, branch_layer, 1)
#     elif model=='Conv':
#       self.model = Plexippus.ConvMulti(latent_dim, ae_layer1, ae_layer2, branch_layer, 1)
#     else:
#       raise ValueError('only \'Conv\' and \'FC\' are supported at this time.')
#     self.learning_rate, self.latent_dim, self.wandb_ = learning_rate, latent_dim, wandb
#     self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=learning_rate)
#     self.losses = []
#     if self.wandb_ == True:
#       import wandb
#       !wandb login
#       self.proj = input('project name: ')
#       self.ent = input('entity: ')
#       wandb.init(project=self.proj, entity=self.ent)

#   def new_wandb(self):
#     wandb.init(project=self.proj, entity=self.ent)

#   def train(self, x, branches, epochs, batch_size, weights='hi', graph_loss=False,save_ls=False): 
#     self.batch_size = batch_size
#     self.x, self.branches, = x, branches
#     self.num_branches = branches[0].shape[1]
#     wandb.log({'batch_size':batch_size, 'epochs': epochs, 'num_branches': self.num_branches, 'weight1': 1, 'weight2': 1, 'weight3': 1, 'weight4': 1})
#     if weights=='hi':
#       self.weights = np.full(shape=branches[0].shape[1], fill_value=1,dtype=int)
#     else:
#       self.weights=weights
#     assert branches[0].shape[1]==len(self.weights), 'weights of wrong size'
#     train_batches = FastTensorDataLoader.DL(x, *branches, batch_size=batch_size, shuffle=False)
#     for epoch in range(epochs):
#       for idx, batch in enumerate(train_batches):
#         self.optimizer.zero_grad()
#         try:
#           latent_space, reconstructed, preds = self.model(batch[0])
#           loss = ae_loss = self.loss_func(reconstructed.squeeze(), batch[0].squeeze())
#           wandb.log({'ae_loss': ae_loss})
#           for i in range(self.num_branches):
#             loss = loss + self.loss_func(preds[i].squeeze(), batch[i+1].squeeze()).mul(self.weights[i])
#         except IndexError:
#           self.model.num_branches = self.num_branches
#         loss.backward(retain_graph=True)
#         self.optimizer.step()
#         self.losses.append(loss.detach().numpy())
#         if self.wandb_==True:
#           wandb.log({'loss': loss, 'branch_losses': int(loss-ae_loss)})
#       self.epochs = epoch + 1
#       if graph_loss==True:
#         self.graph_loss(y_lim=.01, xlim=(epoch-10))

  
#   def graph_loss(self, y_lim=1, xlim=0, linewidth=0.5):
#     plt.style.use('seaborn-whitegrid')
#     plt.xlabel('Iterations')
#     y1 = self.losses
#     x = np.arange(0, self.epochs, 1/(len(y1)/self.epochs))
#     plt.plot(x, y1, color='green', linewidth=linewidth)
#     #plt.ylim(0, y_lim)
#     plt.xlim(xlim)
#     plt.show()

#   def save(self, file_path):
#     torch.save(self.model.state_dict(), file_path)

                        
#   def test(self, x, branches, file_path):
#     self.x_test, self.test_branches = x, branches
#     weights = torch.load(file_path)
#     self.model.load_state_dict(weights)
#     self.model.eval()
#     latent_space, recon, preds = self.model(self.x_test)
#     self.test_ls, self.test_recon, self.preds = latent_space, recon, preds
#     self.loss = self.loss_func(recon.squeeze(), self.x_test.squeeze())
#     for i in range(self.num_branches):
#       self.loss = self.loss + self.loss_func(preds[i].squeeze(), branches[i].squeeze()).mul(self.weights[i])
    
#   def graph_correlation(self, var, name='hi', width=0.3):
#     plt.style.use('seaborn-whitegrid')
#     fig, ax = plt.subplots()
#     dt = self.test_branches[var]
#     pred = self.preds[var]
#     ax.plot(dt.mul(5).detach().numpy().reshape(-1,), dt.mul(5).detach().numpy().reshape(-1,), color='gray', linewidth=0.1)
#     ax.scatter(pred.mul(5).detach().numpy().reshape(-1,), dt.mul(5).detach().numpy().reshape(-1,), color='purple', s=width)
#     ax.set_xlabel('predicted')
#     ax.set_ylabel('actual')
#     if name=='hi':
#       ax.set_title(str(var) + '-value')
#     else:
#       ax.set_title(name + '-value')


#   def graph_latent_space(self):
#     dims = []
#     for dim in range(self.latent_dim):
#       dims.append([i[dim] for i in self.test_ls.detach().numpy()])

#     if self.num_branches == 1:
#       if self.latent_dim==1:
#         fig, axs = plt.subplots(1, 1)
#         axs.scatter(self.test_branches[0].detach().numpy(), dims[0], s=0.1, color='blue')
#         axs.set(xlabel='a-value', ylabel='latent_space')
#       else:
#         fig, axs = plt.subplots(1, self.latent_dim)
#         for dim in range(self.latent_dim):
#           axs[dim].scatter(self.test_branches[0].detach().numpy(), dims[dim], s=0.1, color='blue')
#           axs[dim].set(xlabel='a-value', ylabel='latent_space')
#     else:
#       fig, axs = plt.subplots(self.num_branches, self.latent_dim)
#       for dim in range(self.latent_dim):
#         for bra in range(self.num_branches):
#           axs[dim, bra].scatter(self.test_branches[0].detach().numpy(), dims[dim], s=0.1, color='purple')
#           axs[dim, bra].set(xlabel=str(dim) + 'value', ylabel='latent_space')
#     fig.set_size_inches(self.latent_dim*5, self.num_branches*5)
#     plt.style.use('seaborn-whitegrid')
