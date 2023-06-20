import torch
import gc
from torch import nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import *
from model.covid_model import *
from torch_geometric.nn.models import GIN

class MY_ADCRNN_(torch.nn.Module):
    def __init__(self,num_features=35,out_channels = 5,num_filters=3):
        super(MY_ADCRNN_, self).__init__()
        # self.MLP = MLP(num_input = num_features,hidden_output=100, num_output=out_channels)
        # self.gin = GIN(in_channels = num_features, hidden_channels = 100, num_layers = 1, out_channels = num_features)
        self.recurrent = ADCRNN(in_channels = num_features, out_channels = out_channels,\
                  K = num_filters, bias = True)
        self.linear = torch.nn.Linear(out_channels, 1)

    def forward(self, x, edge_index, edge_weight,h,c):
        h_1=None
        # h_1 = self.MLP(x)
        # x = self.gin(x,edge_index)
        h_2,A = self.recurrent(x, edge_index, edge_weight, H=h_1)
        # h_3,_ = self.recurrent(x, edge_index, edge_weight, H=h_2, residual_matrix=A)
        h_4 = F.relu(h_2)
        y = self.linear(h_4)
        return y,None,None

class MY_TGCN_(torch.nn.Module):
    def __init__(self, node_features,output_size=32):
        super(MY_TGCN_, self).__init__()
        self.recurrent = TGCN(node_features, output_size)
        self.linear = torch.nn.Linear(output_size, 1)

    def forward(self, x, edge_index, edge_weight, prev_hidden_state, c):
        h = self.recurrent(x, edge_index, edge_weight, prev_hidden_state)
        y = F.relu(h)
        y = self.linear(y)
        return y, h, None

class MY_A3TGCN_(torch.nn.Module):
    def __init__(self, node_features, output_size=32, periods = 1):
        super(MY_A3TGCN_, self).__init__()
        self.recurrent = A3TGCN(node_features, output_size, periods)
        self.linear = torch.nn.Linear(output_size, 1)

    def forward(self, x, edge_index, edge_weight, h, c):
        h = self.recurrent(x.view(x.shape[0], 1, x.shape[1]), edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h, None, None

class MY_AGCRN_(torch.nn.Module):
    def __init__(self, node_features, output_size = 2, filter_size=2, num_nodes=5570,emb_dims=35):
        super(MY_AGCRN_, self).__init__()
        self.recurrent = AGCRN(number_of_nodes = num_nodes,
                              in_channels = node_features,
                              out_channels = output_size,
                              K = filter_size,
                              embedding_dimensions = emb_dims)
        self.linear = torch.nn.Linear(output_size, 1)
        self.num_nodes = num_nodes
        self.num_feats = node_features

    def forward(self, x, ignore, ignore_, h, e):
        x = x.view(1, self.num_nodes , self.num_feats)
        # print(h,e.shape)
        h_0 = self.recurrent(x, e, h)
        y = F.relu(h_0)
        y = self.linear(y)
        return y, h_0, e

class MY_DCRNN_(torch.nn.Module):
    def __init__(self, node_features, output_size=32, filter_size=2):
        super(MY_DCRNN_, self).__init__()
        self.recurrent = DCRNN(in_channels=node_features, out_channels=output_size, K=filter_size)
        self.linear = torch.nn.Linear(output_size, 1)

    def forward(self, x, edge_index, edge_weight, h_, c):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h, None, None

class  MY_GConvGRU_(torch.nn.Module):
    def __init__(self, node_features, output_size=32, filter_size=2):
        super(MY_GConvGRU_, self).__init__()
        self.recurrent = GConvGRU(node_features, output_size, K = filter_size)
        self.linear = torch.nn.Linear(output_size, 1)

    def forward(self, x, edge_index, edge_weight, h_, c):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h, None, None

class MY_GCLSTM_(torch.nn.Module):
    def __init__(self, node_features,output_size, filter_size=2):
        super(MY_GCLSTM_, self).__init__()
        self.recurrent = GCLSTM(node_features, output_size, filter_size)
        self.linear = torch.nn.Linear(output_size, 1)
        # self.h = self.c = None

    def forward(self, x, edge_index, edge_weight, h, c):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0

def make_models(num_feats,output_size=32,filter_size=2,num_nodes=5570):
    # models = []
    # for layer in layers:
    #     models.append(RecurrentGCN(layer,output_size))
    models = [
              MY_ADCRNN_(num_feats,output_size,filter_size),
              MY_TGCN_(node_features=num_feats, output_size = output_size),
              # MY_A3TGCN_(node_features = num_feats, output_size = output_size),
              MY_AGCRN_(node_features = num_feats, output_size = output_size, filter_size=filter_size,\
                        num_nodes=num_nodes,emb_dims=num_feats),
              MY_DCRNN_(node_features= num_feats),
              MY_GConvGRU_(node_features= num_feats, output_size=output_size, filter_size=filter_size)
              # MY_GCLSTM_(node_features= num_feats, output_size=output_size, filter_size=filter_size)
              ]
    model_names = ["ADCRNN","TGCN","AGCRN","DCRNN","GConvGRU"]
    return models,model_names