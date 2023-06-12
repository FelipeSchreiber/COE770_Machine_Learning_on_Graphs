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
        self.recurrent = ADCRNN(num_features, num_filters, 1)
        self.linear = torch.nn.Linear(num_filters, 1)

    def forward(self, x, edge_index, edge_weight):
        #,A_h,A_z,A_r 
        h,A_h,A_z,A_r = self.recurrent(x, edge_index, edge_weight)
        h = F.prelu(h)
        h = self.linear(h)
        return h,A_h,A_z,A_r 
    
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
"""    
class GeomRNN(torch.nn.Module):
    
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_classes = num_classes # output size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.input_size = input_size # input size
        self.hidden_size = hidden_size # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2) # lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) # fully connected 
        self.fc_2 = nn.Linear(128, num_classes) # fully connected last layer
        self.relu = nn.ReLU()
        
    def forward(self,x):
        # hidden state
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # cell state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # (input, hidden, and internal state)
        hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) # first dense
        out = self.relu(out) # relu
        out = self.fc_2(out) # final output
        return out

"""