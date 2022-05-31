from torch_geometric.nn import GCNConv
from torch.nn import BatchNorm1d
import torch.nn.functional as F
import torch.nn as nn
import torch

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, class_embed_dim, modulation_hidden_dim, hidden_dim):
        super(GCN, self).__init__()
        self.class_embeds = nn.Embedding(92, class_embed_dim)
        self.modulation = MLP(class_embed_dim, modulation_hidden_dim, hidden_dim, 2)
        
    
    def forward(self, probs):        
        val, idx = torch.max(probs, dim=-1)
        # Cutoff threshold?
        # Weight idx by val
        # Zero out no class! Only consider real predictions.
        val[idx==91] = 0
        lin_comb = val.unsqueeze(-1)*self.class_embeds(idx)
        ret = self.modulation(lin_comb)
        # Take all the 91s and change them
        ret[(val>0.5) & (idx != 91)] = 0
        return ret

def build_gnn(args):
    model = GCN(args.class_embed_dim, args.modulation_hidden_dim, args.hidden_dim)
    return model