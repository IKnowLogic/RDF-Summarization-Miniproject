from functools import partial
from dgl.contrib.data import load_data
from dgl.nn.pytorch import RelGraphConv
from dgl import DGLGraph
import torch
from torch import nn


class SummarizationModel(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases):
        super(SummarizationModel, self).__init__()

        # create rgcn layers
        softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.input_to_hidden = RelGraphConv(num_nodes, h_dim, num_rels, regularizer='basis', num_bases=num_bases)
        self.hidden_to_hidden = RelGraphConv(h_dim, h_dim, num_rels, regularizer='basis', num_bases=num_bases)
        self.hidden_to_output = RelGraphConv(h_dim, out_dim, num_rels, regularizer='basis', num_bases=num_bases)

        self.output_activation = partial(softmax, axis=1)

    def forward(self, g, feats, edge_type, edge_norm):
        feats = self.relu(self.input_to_hidden(g, feats, edge_type, edge_norm))
        feats = self.relu(self.hidden_to_hidden(g, feats, edge_type, edge_norm))
        feats = self.relu(self.hidden_to_output(g, feats, edge_type, edge_norm))
        return feats


data = load_data('aifb', bfs_level=3, relabel=False)

num_nodes = data.num_nodes
num_rels = data.num_rels
num_classes = data.num_classes
labels = data.labels
train_idx = data.train_idx
test_idx = data.test_idx

feats = torch.range(0, num_nodes - 1, dtype=torch.int64)
edge_type = torch.tensor(data.edge_type)
edge_norm = torch.unsqueeze(torch.tensor(data.edge_norm), 1)
labels = torch.reshape(torch.tensor(labels), (-1, ))

g = DGLGraph()
g.add_nodes(data.num_nodes)
g.add_edges(data.edge_src, data.edge_dst)

with torch.no_grad():
    model = SummarizationModel(g.number_of_nodes(), h_dim=16, out_dim=20, num_rels=num_rels, num_bases=25)
    logits = model(g, feats, edge_type, edge_norm)

print(logits)
print("done")

exit()
