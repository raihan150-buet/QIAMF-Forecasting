import torch
import torch.nn as nn
from .layers import (QuantumInspiredAttention, AdaptiveHybridDecomposition, 
                     CrossModalFusion, MetaLearningAdapter, HierarchicalMultiHorizon)

class QIAMF(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.d_model = config['d_model']
        
        self.embed = nn.Linear(input_dim, self.d_model)
        
        # Initialize Components
        self.decomp = AdaptiveHybridDecomposition(config['seq_length'], self.d_model)
        self.fusion = CrossModalFusion(input_dim, self.d_model, config['dropout'])
        self.meta = MetaLearningAdapter(self.d_model)
        self.attn = QuantumInspiredAttention(self.d_model)
        self.head = HierarchicalMultiHorizon(self.d_model, config['output_dim'])
        
        # Uncertainty std dev head
        self.std_head = nn.Sequential(nn.Linear(self.d_model, config['output_dim']), nn.Softplus())

    def forward(self, x):
        # 1. Embedding & Decomp
        emb = self.embed(x)
        dec, w_dec = self.decomp(x)
        
        # 2. Fusion
        fus, aleatoric, epistemic = self.fusion(x)
        
        # 3. Combine & Meta-Adapt
        combined = emb + dec + fus
        adapted, w_patt = self.meta(combined)
        
        # 4. Attention & Prediction
        ctx, w_attn = self.attn(adapted)
        pred, _, w_route = self.head(ctx)
        
        # 5. Uncertainty Intervals
        std = self.std_head(ctx)
        total_unc = torch.sqrt(aleatoric**2 + epistemic**2 + 1e-6)
        
        return {
            'prediction': pred,
            'lower_bound': pred - 1.96 * std,
            'upper_bound': pred + 1.96 * std,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'weights': {
                'decomp': w_dec,
                'pattern': w_patt,
                'attn': w_attn,
                'router': w_route
            }
        }