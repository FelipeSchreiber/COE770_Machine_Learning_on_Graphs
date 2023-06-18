import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN
from .ADCRNN import ADCRNN
import gdown
import os

class MLP(nn.Module):
    def __init__(self,num_input=35,hidden_output=100,num_output=5):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_input, hidden_output),
            nn.ReLU(),
            nn.Linear(hidden_output, num_output),
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x


class RecurrentGCN(torch.nn.Module):
    def __init__(self,num_features=35,out_channels = 5,num_filters=3):
        super(RecurrentGCN, self).__init__()
        self.MLP = MLP(num_input = num_features,hidden_output=100, num_output=out_channels)
        self.recurrent = ADCRNN(in_channels = num_features, out_channels = out_channels,\
                  K = num_filters, bias = True)
        self.linear = torch.nn.Linear(out_channels, 1)

    def forward(self, x, edge_index, edge_weight):
        h_1=None
        h_1 = self.MLP(x)
        h_2,A = self.recurrent(x, edge_index, edge_weight, H=h_1)
        # h_3,_ = self.recurrent(x, edge_index, edge_weight, H=h_2, residual_matrix=A)
        h_4 = F.relu(h_2)
        y = self.linear(h_4)
        return y,A
    
def get_model(get_whole_model=True,num_features=35,num_filters=3,gamma=1):
    if get_whole_model:
        url = "https://drive.google.com/uc?export=download&id=1XYICNGVXOEqazVJmfEfxlLiy7QoNeQzs"
        output = "covid_model"
        gdown.download(url, output)
        cwd = os.getcwd()+"/"
        return torch.load(cwd+output)
    else:
        url = "https://drive.google.com/drive/folders/1V8CaUUS3gPQAcRE2YebNkqWOWKRA7vmt?usp=sharing"
        output = "COE770_GNN/"
        model_name = f"model_weights_ADCRNN_{num_filters}_{gamma}"
        # url = "https://drive.google.com/uc?export=download&id=1skGodpbZZhuDyvxJbicnIe8ed9d0w5iN"
        # output = "covid_model_weights"
        # cwd+output+model_name
        gdrive_path = "/content/drive/MyDrive/COE770_GNN/"
        # gdown.download(url, output)
        cwd = os.getcwd()+"/"
        covid_model = RecurrentGCN(num_features=num_features,num_filters=num_filters)
        covid_model.load_state_dict(torch.load(gdrive_path+model_name))
        return covid_model