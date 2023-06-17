import torch
from torch import nn
import torch.nn.functional as F
from torchMY_Geometric_temporal.nn.recurrent import *

# def make_layers(num_feats,output,filter_size=2):
#     layers = [DCRNN(num_feats, output, filter_size),\
#               ## to correct embedding dim
#             #   AGCRN(5570,num_feats, output, filter_size, embedding_dimensions = 32),\
#             #   MPNNLSTM(num_feats, output, 5570, 1, 0.5),\
#               A3TGCN(num_feats, output, periods = 1),\
#               TGCN(num_feats, output),\
#               GConvGRU(num_feats, output, filter_size),\
#               GConvLSTM(num_feats, output, filter_size),\
#               GCLSTM(num_feats, output, filter_size)]
#     return layers

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
    
class MY_A3TGCN_(torch.nn.Module):
    def __init__(self, node_features, output_size=32, periods = 4):
        super(MY_A3TGCN_, self).__init__()
        self.recurrent = A3TGCN(node_features, output_size, periods)
        self.linear = torch.nn.Linear(output_size, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x.view(x.shape[0], 1, x.shape[1]), edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
        
class MY_AGCNRN_(torch.nn.Module):
    def __init__(self, node_features = 35, output_size = 2, filter_size=2, num_nodes=5570):
        super(MY_AGCNRN_, self).__init__()
        self.recurrent = AGCRN(number_of_nodes = num_nodes,
                              in_channels = node_features,
                              out_channels = output_size,
                              K = filter_size,
                              embedding_dimensions = 4)
        self.h = None
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x, e, w):
        h_0 = self.recurrent(x, e, self.h)
        y = F.relu(h_0)
        y = self.linear(y)
        self.h = h_0
        return y

class MY_DCRNN_(torch.nn.Module):
    def __init__(self, node_features, output_size=32, filter_size=2):
        super(MY_DCRNN_, self).__init__()
        self.recurrent = DCRNN(node_features, output_size, K=filter_size)
        self.linear = torch.nn.Linear(output_size, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
        
class  MY_GConvGRU_(torch.nn.Module):
    def __init__(self, node_features, output_size=32, filter_size=2):
        super(MY_GConvGRU_, self).__init__()
        self.recurrent = GConvGRU(node_features, output_size, K = filter_size)
        self.linear = torch.nn.Linear(output_size, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

class MY_GCLSTM_(torch.nn.Module):
    def __init__(self, node_features,output_size, filter_size=2):
        super(MY_GCLSTM_, self).__init__()
        self.recurrent = GCLSTM(node_features, output_size, filter_size)
        self.linear = torch.nn.Linear(output_size, 1)
        self.h = self.c = None

    def forward(self, x, edge_index, edge_weight):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, self.h, self.c)
        h = F.relu(h_0)
        h = self.linear(h)
        self.h = h_0
        self.c = c_0
        return h
        

def make_models(layers:list,num_feats,output_size=32,filter_size=2):
    # models = []
    # for layer in layers:
    #     models.append(RecurrentGCN(layer,output_size))
    models = [
            ##MY_A3TGCN_(node_features = num_feats, output_size = output_size),
            #   MY_AGCNRN_(node_features = num_feats, output_size = output_size, filter_size=filter_size, num_nodes=5570),
              MY_DCRNN_(node_features= num_feats, output_size=output_size, filter_size=filter_size),
              MY_GConvGRU_(node_features= num_feats, output_size=output_size, filter_size=filter_size),
              MY_GCLSTM_(node_features= num_feats, output_size=output_size, filter_size=filter_size)
              ]

    return models