from torch_geometric.nn import GCNConv
from torch.nn import BatchNorm1d
import torch.functional as F
import torch.nn as nn

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
        self.convs = ModuleList([GCNConv(in_channels=gnn_query_dim, out_channels=hidden_dim), GCNConv(in_channels=hidden_dim, out_channels=output_dim)])
        self.bns = BatchNorm1d(num_features=hidden_dim)
        self.dropout = dropout
        self.query_embeddings = nn.Embedding(num_queries, gnn_query_dim)
        self.node_embeddings = nn.Embedding(num_queries, gnn_query_dim)
        # self.class_embeddings = nn.Embedding(1001, class_dim)
        self.edge_weights = MLP(num_queries+4+1001, mlp_hidden_dim, num_queries, 2)
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
        edge_weight_input = torch.concat([query_embeds, probs, bbox_coords], dim=-1)

        # Generate adjacency matrix with the probabilities
        # (B, N, N)
        adj_t = self.edge_weights(edge_weight_input)
        adj_t = torch.topk(adj_t, self.top_k, dim=-1, largest=True)

        out = self.node_embeddings.weight
        out = self.convs[0](out, adj_t)
        out = self.bns(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.convs[1](out, adj_t)
        return out

def build_gnn(args):
    model = GCN(args.edge_mlp_hidden_dim, args.top_k, args.num_queries, args.gnn_query_dim, args.gnn_hidden_dim, args.hidden_dim, 0.2)
    return model