import torch

class Cressida(torch.nn.Module):
  def __init__(self, latent_dim, ae_layer1, ae_layer2, branch_layer, num_branches):
    super().__init__()
    self.enc_conv1 = torch.nn.Conv1d(in_channels=1, out_channels=ae_layer1, kernel_size=200)
    self.enc_conv2 = torch.nn.Conv1d(in_channels=ae_layer1, out_channels=ae_layer2, kernel_size=1)
    self.enc_lin1 = torch.nn.Linear(ae_layer2, latent_dim)
    self.dec_conv1 = torch.nn.Conv1d(in_channels=1, out_channels=ae_layer2, kernel_size=latent_dim, padding=0)
    self.dec_conv2 = torch.nn.Conv1d(in_channels=ae_layer2, out_channels=ae_layer1, kernel_size=1)
    self.dec_lin1 = torch.nn.Linear(ae_layer1, 200)
    
    self.num_branches = num_branches
    self.branches = []
    
    for i in self.num_branches:
      self.branches.append(torch.nn.Sequential(
        torch.nn.ReLU(),
        torch.nn.Linear(latent_dim, branch_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(branch_layer, 1)
      ))

  def encoder(self, x):
    x = self.enc_conv1(x.unsqueeze(dim=1))
    x = torch.nn.functional.leaky_relu(x)
    x = self.enc_conv2(x)
    x = torch.nn.functional.leaky_relu(x)
    x = self.enc_lin1(x.squeeze(dim=2))
    return x

  def decoder(self, x):
    x = self.dec_conv1(x.unsqueeze(dim=1))
    x = torch.nn.functional.leaky_relu(x)
    x = self.dec_conv2(x)
    x = torch.nn.functional.leaky_relu(x)
    x = self.dec_lin1(x.squeeze(dim=2))
    return x

  def forward(self, x): 
    ls = self.encoder(x)
    preds = []
    for branch in self.branches:
      preds.append(branch(ls))
    return ls, self.decoder(ls), preds
