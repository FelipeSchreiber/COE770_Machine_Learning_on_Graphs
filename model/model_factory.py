# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch_geometric_temporal.nn.recurrent import DCRNN

# class MLP(nn.Module):
#     def __init__(self,num_input=35,hidden_output=100,num_output=5):
#         super(MLP, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(num_input, hidden_output),
#             nn.PReLU(),
#             nn.Linear(hidden_output, num_output),
#         )
        
#     def forward(self, x):
#         x = self.layers(x)
#         return x

# class RecurrentGCN(torch.nn.Module):
#     def __init__(self, node_features):
#         super(RecurrentGCN, self).__init__()
#         self.recurrent = DCRNN(node_features, 32, 1)
#         self.linear = torch.nn.Linear(32, 1)

#     def forward(self, x, edge_index, edge_weight):
#         h = self.recurrent(x, edge_index, edge_weight)
#         h = F.relu(h)
#         h = self.linear(h)
#         return h


# class RecurrentGCN(torch.nn.Module):
#     def __init__(self,num_features=35,out_channels = 5,num_filters=3):
#         super(RecurrentGCN, self).__init__()
#         self.MLP = MLP(num_input = num_features,hidden_output=100, num_output=out_channels)
#         self.recurrent = DCRNN(in_channels = num_features, out_channels = out_channels,\
#                   K = num_filters, bias = True)
#         # self.recurrent_second = ADCRNN(in_channels = out_channels, out_channels = 1,\
#         #           K = num_filters, bias = True)
#         self.linear = torch.nn.Linear(out_channels, 1)

#     def forward(self, x, edge_index, edge_weight):
#         h_1 = self.MLP(x)
#         h_2,A = self.recurrent(x, edge_index, edge_weight, H=h_1)
#         h_3,_ = self.recurrent(x, edge_index, edge_weight, H=h_2, residual_matrix=A)
#         h_4 = F.relu(h_3)
#         y = self.linear(h_4)
#         return y,A