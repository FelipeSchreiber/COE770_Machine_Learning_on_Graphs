import torch
import torch.nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from .ADCRNN import ADCRNN
import gdown
import os

class RecurrentGCN(torch.nn.Module):
    def __init__(self, num_features=35,num_filters=3):
        super(RecurrentGCN, self).__init__()
        self.recurrent = ADCRNN(in_channels = num_features, out_channels = num_features,\
                  K = num_filters, bias = True)

        self.linear = torch.nn.Linear(num_filters, 1)

    def forward(self, x, edge_index, edge_weight):
        h,A = self.recurrent(x, edge_index, edge_weight)
        h,_ = self.recurrent(x, edge_index, edge_weight, h, A)
        h = self.linear(h)
        h = F.relu(h)
        return h,A
    
def get_model(get_whole_model=True,num_features=35,num_filters=3):
    if get_whole_model:
        url = "https://drive.google.com/uc?export=download&id=1XYICNGVXOEqazVJmfEfxlLiy7QoNeQzs"
        output = "covid_model"
        gdown.download(url, output)
        cwd = os.getcwd()+"/"
        return torch.load(cwd+output)
    else:
        url = "https://drive.google.com/uc?export=download&id=1skGodpbZZhuDyvxJbicnIe8ed9d0w5iN"
        output = "covid_model_weights"
        gdown.download(url, output)
        cwd = os.getcwd()+"/"
        covid_model = RecurrentGCN(num_features=num_features,num_filters=num_filters)
        covid_model.load_state_dict(torch.load(cwd+output))
        return covid_model