import torch

class Cleophile(torch.nn.Module):
  def __init__(self, latent_dim, ae_layer, branch_layer, num_branches):
    super().__init__()
    self.latent_dim, self.ae_layer, self.branch_layer = latent_dim, ae_layer, branch_layer
    self.num_branches = num_branches
    self.branches = []
    
    for i in range(self.num_branches):
      self.branches.append(torch.nn.Sequential(
        torch.nn.ReLU(),
        torch.nn.Linear(latent_dim, branch_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(branch_layer, 1)
      ))

    self.encoder = torch.nn.Sequential(
      torch.nn.Linear(100, ae_layer),
      torch.nn.ReLU(),
      torch.nn.Linear(ae_layer, latent_dim)
    )
    self.decoder = torch.nn.Sequential( 
      torch.nn.ReLU(),
      torch.nn.Linear(latent_dim, ae_layer),
      torch.nn.ReLU(),
      torch.nn.Linear(ae_layer, 100)
    )
      
  def forward(self, x):
    try: 
      ls = self.encoder(x)
    except:
      self.encoder = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], self.ae_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(self.ae_layer, self.latent_dim)
      )
      self.decoder = torch.nn.Sequential( 
        torch.nn.ReLU(),
        torch.nn.Linear(self.latent_dim, self.ae_layer),
        torch.nn.ReLU(),
        torch.nn.Linear(self.ae_layer, x.shape[1])
      )
    preds = []
    for branch in self.branches:
      preds.append(branch(ls))
    return ls, self.decoder(ls), preds
