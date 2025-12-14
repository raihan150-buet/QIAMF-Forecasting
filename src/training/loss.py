import torch
import torch.nn as nn
import math

class NovelLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambda_u = config['lambda_uncertainty']
        self.lambda_d = config['lambda_decomp']
        
    def forward(self, output_dict, target):
        pred = output_dict['prediction']
        
        # 1. MSE
        mse = nn.MSELoss()(pred, target)
        
        # 2. NLL (Uncertainty)
        # derived from intervals: width = 2 * 1.96 * std
        width = output_dict['upper_bound'] - output_dict['lower_bound']
        std = width / (2 * 1.96)
        nll = torch.mean(0.5 * torch.log(std**2 + 1e-6) + 0.5 * (pred - target)**2 / (std**2 + 1e-6))
        
        # 3. Regularization (Entropy of decomp weights)
        w_dec = output_dict['weights']['decomp']
        entropy = -torch.sum(w_dec * torch.log(w_dec + 1e-10), dim=-1).mean()
        
        total = mse + self.lambda_u * nll - self.lambda_d * entropy
        
        return total, {'mse': mse.item(), 'nll': nll.item(), 'loss': total.item()}