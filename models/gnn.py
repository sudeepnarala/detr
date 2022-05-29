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

# Some code from CS224w Colab 2
# 2 layer GCN
class GCN(torch.nn.Module):
    def __init__(self, mlp_hidden_dim, top_k, num_queries, gnn_query_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList([GCNConv(in_channels=gnn_query_dim, out_channels=hidden_dim), GCNConv(in_channels=hidden_dim, out_channels=output_dim)])
        self.bns = BatchNorm1d(num_features=hidden_dim)
        self.dropout = dropout
        self.num_queries = num_queries
        self.query_embeddings = nn.Embedding(num_queries, gnn_query_dim)
        self.node_embeddings = nn.Embedding(num_queries, gnn_query_dim)
        # self.class_embeddings = nn.Embedding(1001, class_dim)
        # 91 classes
        self.edge_weights = MLP(gnn_query_dim+4+91+1, mlp_hidden_dim, num_queries, 2)
        self.top_k = top_k

        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
    
    def forward(self, probs, bbox_coords):
        """
        :param probs - (B, N, C+1)
        :param bbox_coords - (B, N, 4)
        """
        # Can't just use transformer output because means less without context.
        # It is used to derive the other quantities
        # First, we need to compute the adjacency matrix

        # Just pull the query embeds
        query_embeds = self.query_embeddings.weight
        
        # Concatenate query and class embeddings to get features of each node
        edge_weight_input = torch.concat([query_embeds.unsqueeze(0).repeat(probs.shape[0], 1, 1), probs, bbox_coords], dim=-1)

        # Generate adjacency matrix with varied weights
        # (B, N, N)
        adjacency_mat = self.edge_weights(edge_weight_input)
        # (B, N, self.top_k)
        vals, indices = torch.topk(adjacency_mat, self.top_k, dim=-1, largest=True)
        # Between the index and the indices (B, 2, N*self.top_k)
        # 0 self.top_k, then 1 self.top_k times etc.
        edge_indices = torch.stack([torch.arange(self.num_queries).unsqueeze(-1).repeat(1, self.top_k).flatten().unsqueeze(0).repeat(probs.shape[0], 1).to("cuda"), indices.flatten(1)], dim=1)
        # Just take this from the vals (B, N*self.top_k)
        edge_weights = vals.flatten(1)

        # node_embeds = (self.node_embeddings.weight).unsqueeze(0).repeat(probs.shape[0], 1, 1)
        node_embeds = self.node_embeddings.weight
        # Hmm, can't batch like that sadly, only 1 graph type per forward for GCNConv
        # out = self.convs[0](out, edge_indices, edge_weights)
        outs = []
        for i in range(probs.shape[0]):
            out = self.convs[0](node_embeds, edge_indices[i], edge_weights[i])
            out = self.bns(out)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            out = self.convs[1](out, edge_indices[i], edge_weights[i])
            outs.append(out.unsqueeze(0))
        return torch.stack(outs, dim=0)

def build_gnn(args):
    model = GCN(args.edge_mlp_hidden_dim, args.top_k, args.num_queries, args.gnn_query_dim, args.gnn_hidden_dim, args.hidden_dim, 0.2)
    return model