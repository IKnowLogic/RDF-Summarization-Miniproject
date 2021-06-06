from functools import partial
from dgl.contrib.data import load_data
from dgl.nn.pytorch import RelGraphConv
from dgl import DGLGraph
import tensorflow as tf
import os

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['DGLBACKEND'] = 'keras'
#DGLBACKEND = tf


class SummarizationModel(tf.keras.Model):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases):
        super(SummarizationModel, self).__init__()

        # create rgcn layers
        self.input_to_hidden = RelGraphConv(num_nodes, h_dim, num_rels, regularizer='basis', num_bases=num_bases)
        self.hidden_to_hidden = RelGraphConv(h_dim, h_dim, num_rels, regularizer='basis', num_bases=num_bases)
        self.hidden_to_output = RelGraphConv(h_dim, out_dim, num_rels, regularizer='basis', num_bases= num_bases)

        self.output_activation = partial(tf.nn.softmax, axis=1)

    def call(self, g, feats, edge_type, edge_norm):
        feats = tf.nn.relu(self.input_to_hidden(g, feats, edge_type, edge_norm))
        feats = tf.nn.relu(self.hidden_to_hidden(g, feats, edge_type, edge_norm))
        feats = self.output_activation(self.hidden_to_output(g, feats, edge_type, edge_norm))
        return feats


data = load_data('aifb', bfs_level=3, relabel=False)

num_nodes = data.num_nodes
num_rels = data.num_rels
num_classes = data.num_classes
labels = data.labels
train_idx = data.train_idx
test_idx = data.test_idx

feats = tf.range(num_nodes, dtype=tf.int64)
edge_type = tf.convert_to_tensor(data.edge_type)
edge_norm = tf.expand_dims(tf.convert_to_tensor(data.edge_norm), 1)
labels = tf.reshape(tf.convert_to_tensor(labels), (-1, ))

g = DGLGraph()
g.add_nodes(data.num_nodes)
g.add_edges(data.edge_src, data.edge_dst)

import torch
g = g.to(torch.device('cpu'))

model = SummarizationModel(len(g), h_dim=16, out_dim=num_classes, num_rels=num_rels, num_bases=25)
logits = model(g, feats, edge_type, edge_norm)

print(logits)
print("done")

exit()
