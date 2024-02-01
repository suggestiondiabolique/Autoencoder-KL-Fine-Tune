import torch
from torch.nn.functional import mse_loss,binary_cross_entropy
import numpy as np
from torch.nn import BCELoss

def vanilla_mse_loss(x, xhat):
    return mse_loss(x, xhat)

def vanilla_bce_loss(x, xhat):
    return binary_cross_entropy(x, xhat)

def vanilla_mae(x, xhat):
    return torch.mean(torch.abs(x - xhat))

def KL_Loss(x,xhat, mean, logvar):
     KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
     return KL_divergence

def reconstruction_loss(x,x_hat,logvar,mean):
  '''
    Note: Original implementation; large loss values for RGB values.
    Consider possibly implementing annealing / use MSE
  '''
  BCE_loss = BCELoss(reduction = "sum")
  reconstruction_loss = BCE_loss(x_hat, x)
  kl_loss = KL_Loss(x, x_hat, mean, logvar)
  return reconstruction_loss + kl_loss