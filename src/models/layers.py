import torch
import torch.nn as nn
import torch.nn.functional as F

# --- NOVELTY 1: Quantum-Inspired Attention (VECTORIZED) ---
class QuantumInspiredAttention(nn.Module):
    def __init__(self, d_model, num_quantum_states=8):
        super().__init__()
        self.d_model = d_model
        
        # State preparation
        self.state_preparation = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_quantum_states)
        ])
        
        # Entanglement
        self.entanglement_gates = nn.ModuleList([
            nn.Linear(d_model * 2, d_model) for _ in range(num_quantum_states)
        ])
        
        # Measurement
        self.measurement = nn.Sequential(
            nn.Linear(d_model * num_quantum_states, d_model),
            nn.LayerNorm(d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        self.phase_rotation = nn.Parameter(torch.randn(num_quantum_states, d_model))

    def forward(self, x):
        B, T, D = x.shape
        
        # 1. State Preparation (Superposition)
        # We process all states at once
        quantum_states = []
        for i, gate in enumerate(self.state_preparation):
            state = gate(x)
            phase = torch.exp(1j * self.phase_rotation[i].view(1, 1, -1))
            quantum_states.append(state * phase.real)

        # 2. Entanglement (Vectorized - No more loop!)
        # Create all-to-all pairs efficiently
        # x shape: (B, T, D)
        # We want a tensor of shape (B, T, T, 2*D) representing every pair
        
        x_i = x.unsqueeze(2).expand(-1, -1, T, -1) # (B, T, T, D) -> repeats along dim 2
        x_j = x.unsqueeze(1).expand(-1, T, -1, -1) # (B, T, T, D) -> repeats along dim 1
        
        # Concatenate to get all pairs (Time_i, Time_j)
        pairs = torch.cat([x_i, x_j], dim=-1) # (B, T, T, 2D)
        
        entangled_states = []
        for gate in self.entanglement_gates:
            # Apply linear layer to the massive (B, T, T, 2D) tensor
            out = gate(pairs) # (B, T, T, D)
            # Average over the second time dimension to get "entanglement with all others"
            entangled_states.append(out.mean(dim=2)) # (B, T, D)

        entangled_states = torch.stack(entangled_states, dim=1) # (B, num_states, T, D)
        
        # 3. Measurement
        # Reshape: (B, T, D * num_states)
        # Note: We need to permute to align dimensions correctly
        quantum_vector = entangled_states.permute(0, 2, 3, 1).reshape(B, T, -1)
        
        attn_logits = self.measurement(quantum_vector)
        attn_weights = F.softmax(attn_logits, dim=1)
        
        context = torch.sum(attn_weights * x, dim=1)
        return context, attn_weights

# --- NOVELTY 2: Adaptive Decomposition (FIXED CHANNELS + TIME) ---
class AdaptiveHybridDecomposition(nn.Module):
    def __init__(self, seq_length, d_model):
        super().__init__()
        self.channel_per_filter = d_model // 3
        
        self.trend_filters = nn.ModuleList([
            nn.Conv1d(1, self.channel_per_filter, k, padding=k//2) for k in [25, 49, 97]
        ])
        self.seasonal_filters = nn.ModuleList([
            nn.Conv1d(1, self.channel_per_filter, k, padding=k//2) for k in [7, 24, 168]
        ])
        self.residual_filters = nn.ModuleList([
            nn.Conv1d(1, self.channel_per_filter, k, padding=k//2) for k in [3, 5, 7]
        ])
        
        self.attention = nn.Sequential(
            nn.Linear(seq_length, 128),
            nn.ReLU(),
            nn.Linear(128, 9),
            nn.Softmax(dim=-1)
        )
        
        total_channels = 9 * self.channel_per_filter
        self.fusion = nn.Linear(total_channels, d_model)

    def forward(self, x):
        B, T, F = x.shape
        x_target = x[:, :, 0:1].permute(0, 2, 1) # (B, 1, T)
        weights = self.attention(x_target.squeeze(1))
        
        components = []
        all_filters = self.trend_filters + self.seasonal_filters + self.residual_filters
        
        for i, flt in enumerate(all_filters):
            out = flt(x_target)
            if out.shape[-1] > T: out = out[..., :T]
            components.append(out * weights[:, i].view(B, 1, 1))
            
        decomposed = torch.cat(components, dim=1).permute(0, 2, 1)
        return self.fusion(decomposed), weights

# --- NOVELTY 3: Cross Modal Fusion ---
class CrossModalFusion(nn.Module):
    def __init__(self, input_dim, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.time_enc = nn.Linear(input_dim, d_model)
        self.freq_enc = nn.Linear(input_dim * 2, d_model)
        self.wavelet_enc = nn.Linear(input_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, 8, dropout=dropout, batch_first=True)
        self.aleatoric = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
        self.epistemic = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())

    def forward(self, x):
        B, T, _ = x.shape
        feat_t = self.time_enc(x)
        fft = torch.fft.rfft(x, dim=1)
        fft_feat = torch.cat([fft.real, fft.imag], dim=-1)
        if fft_feat.size(1) < T:
            fft_feat = F.pad(fft_feat, (0, 0, 0, T - fft_feat.size(1)))
        else:
            fft_feat = fft_feat[:, :T, :]
        feat_f = self.freq_enc(fft_feat)
        feat_w = self.wavelet_enc(x)
        all_modes = torch.stack([feat_t, feat_f, feat_w], dim=1).view(B, 3*T, -1)
        fused, _ = self.cross_attn(all_modes, all_modes, all_modes)
        fused = fused.view(B, 3, T, -1).mean(dim=1)
        pooled = fused.mean(dim=1)
        return fused, self.aleatoric(pooled), self.epistemic(pooled)

# --- NOVELTY 4: Meta Adapter ---
class MetaLearningAdapter(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.meta_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(2)])
        self.adapt_rate = nn.Parameter(torch.tensor(0.01))
        self.detector = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 8), nn.Softmax(dim=-1)
        )
        self.adapters = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(8)])

    def forward(self, x):
        B, T, D = x.shape
        probs = self.detector(x.mean(dim=1))
        adapted = sum(adapter(x) * probs[:, i].view(B, 1, 1) for i, adapter in enumerate(self.adapters))
        out = adapted
        for layer in self.meta_layers:
            out = layer(out) + self.adapt_rate * out
        return out, probs

# --- NOVELTY 5: Multi-Horizon ---
class HierarchicalMultiHorizon(nn.Module):
    def __init__(self, d_model, output_dim=24):
        super().__init__()
        self.horizons = [6, 12, 24]
        self.shared = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        self.router = nn.Sequential(nn.Linear(d_model, len(self.horizons)), nn.Softmax(dim=-1))
        self.experts = nn.ModuleList([
            nn.Linear(d_model, h) for h in self.horizons
        ])

    def forward(self, x):
        shared = self.shared(x)
        weights = self.router(shared)
        preds = []
        for i, expert in enumerate(self.experts):
            p = expert(shared)
            if p.shape[1] < 24:
                p = F.pad(p, (0, 24 - p.shape[1]))
            preds.append(p * weights[:, i:i+1])
        final = torch.stack(preds).sum(dim=0)
        return final, preds, weights