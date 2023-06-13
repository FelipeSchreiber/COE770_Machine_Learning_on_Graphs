import torch
import gc
from dataset.covidBR_dataset import *
from model.covid_model import *
from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm

device = "cpu"
dtype = torch.FloatTensor
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    device = "cuda"

class CovidBenchmark():
    def __init__(self):
        self.n = None

    def check_mem(self):
        #Additional Info when using cuda
        if device == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    def free_cache(self):
        gc.collect()
        torch.cuda.empty_cache()

    def run_test(self,lags=4,filter_size=3,train_model=True,gamma=1,num_epochs=100):
        loader = CovidDatasetLoader(method="other")
        dataset = loader.get_dataset(lags=lags)
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
        num_feats = dataset[0].x.shape[1]

        if train_model:
            model = RecurrentGCN(num_features = num_feats,out_channels = 5,num_filters = filter_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            model.train()

            for epoch in tqdm(range(num_epochs)):
                for time, snapshot in enumerate(train_dataset):
                    snapshot.to(device)
                    y_hat,A  = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                    cost = torch.mean((y_hat-snapshot.y)**2)\
                    + gamma*torch.norm(A,p=1)
                    cost.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    del snapshot
                    self.free_cache()
                    # self.check_mem()
                if epoch % 10 == 0:
                    torch.save(model.state_dict(), "./model_weights_ADCRNN")
            torch.save(model, "./the_whole_model_ADCRNN")
            self.free_cache()
        else:
            model = get_model(False)
            model.to(device)

        model.eval()
        cost = 0

        for time, snapshot in enumerate(test_dataset):
            snapshot.to(device)
            y_hat,_ = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost = cost + torch.mean((y_hat-snapshot.y)**2).item()
            cost = cost / (time+1)
            del snapshot
            self.free_cache()