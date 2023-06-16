import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import *

def make_models(layers:list,output_size):
    models = []
    for layer in layers:
        models.append(RecurrentGCN(layer,output_size))
    return models

def make_layers(num_feats,output,filter_size=2):
    layers = [DCRNN(num_feats, output, filter_size),\
              AGCRN(5570,num_feats, output, filter_size, embedding_dimensions = 32),\
              MPNNLSTM(num_feats, output, 5570, 1, 0.5),\
              A3TGCN(num_feats, output, periods = 7),\
              TGCN(num_feats, output),\
              GConvGRU(num_feats, output, filter_size),\
              GConvLSTM(num_feats, output, filter_size),\
              GCLSTM(num_feats, output, filter_size)]
    return layers

model_names = ["DCRNN","AGCRN","MPNNLSTM","A3TGCN","TGCN","GConvGRU","GConvLSTM","GCLSTM"]

class MLP(nn.Module):
    def __init__(self,num_input=35,hidden_output=100,num_output=5):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_input, hidden_output),
            nn.PReLU(),
            nn.Linear(hidden_output, num_output),
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x

class RecurrentGCN(torch.nn.Module):
    def __init__(self, layer, out):
        super(RecurrentGCN, self).__init__()
        self.recurrent = layer
        self.linear = torch.nn.Linear(out, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h