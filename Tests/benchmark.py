import torch
from itertools import product
import gc
from dataset.covidBR_dataset import *
from model.covid_model import *
from model.model_factory import *
from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm
# import shutil
import os

gdrive_path = "/content/drive/MyDrive/COE770_GNN/"
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

    def run_test(self,lags=4,filter_sizes=[16,8,4],\
                 train_model=True,gammas=[1,1e4,1e6,1e8],\
                 num_epochs=100,warm_start=True):
        loader = CovidDatasetLoader(method="other")
        dataset = loader.get_dataset(lags=lags)
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
        num_feats = dataset[0].x.shape[1]
        stats = {"MSE":[],"gamma":[],"filter_size":[]}
        
        for filter_size, gamma in tqdm(product(filter_sizes,gammas)):
            stats["gamma"].append(gamma)
            stats["filter_size"].append(filter_size)
            model = None
            if train_model:
                if not warm_start:
                    model = RecurrentGCN(num_features = num_feats,\
                                         out_channels = 5,\
                                         um_filters = filter_size).to(device)
                else:
                    model = get_model(False,num_features=35,num_filters=filter_size,gamma=gamma)
                    model.to(device)
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
                        # self.free_cache()
                        # self.check_mem()
                    if epoch % 10 == 0:
                        filepath = f"./model_weights_ADCRNN_{filter_size}_{gamma}"
                        if (os.path.isfile(filepath)):
                            os.remove(filepath)
                        torch.save(model.state_dict(), filepath)
                        torch.save(model.state_dict(), gdrive_path+filepath)
                        #url = "https://drive.google.com/drive/folders/1V8CaUUS3gPQAcRE2YebNkqWOWKRA7vmt?usp=sharing"
                        #output = "COE770_GNN/"
                        # shutil.copy(f"./model_weights_ADCRNN_{filter_size}_{gamma}",\
                        #             "/content/drive/MyDrive/COE770_GNN/")
                torch.save(model, f"./the_whole_model_ADCRNN_{filter_size}_{gamma}")
                self.free_cache()
            else:
                model = get_model(False,num_features=35,num_filters=filter_size,gamma=gamma)
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
            stats["MSE"].append(cost)
        return stats

    def run_test_other_models(self,lags=4,train_model=True,filter_size=2,\
                              num_epochs=100,output_size=32,warm_start=False):
        loader = CovidDatasetLoader(method="other")
        dataset = loader.get_dataset(lags=lags)
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
        num_feats = dataset[0].x.shape[1]
        stats = {"MSE":[],"model":[]}
        
        layers = make_layers(num_feats,output_size,filter_size)
        models = make_models(layers,output_size)
        gdrive_path = "/content/drive/MyDrive/COE770_GNN/"
        ##model_names is defined in model_factory
        for model,model_name in zip(models,model_names):
            if train_model:
                if warm_start:
                    model.load_state_dict(torch.load(gdrive_path+model_name))
                    model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                model.train()

                for epoch in tqdm(range(num_epochs)):
                    for time, snapshot in enumerate(train_dataset):
                        snapshot.to(device)
                        y_hat  = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                        cost = torch.mean((y_hat-snapshot.y)**2)
                        cost.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        del snapshot
                        # self.free_cache()
                        # self.check_mem()
                    if epoch % 10 == 0:
                        filepath = f"./{model_name}"
                        if (os.path.isfile(filepath)):
                            os.remove(filepath)
                        torch.save(model.state_dict(), filepath)
                        torch.save(model.state_dict(), gdrive_path+model_name)
                        #url = "https://drive.google.com/drive/folders/1V8CaUUS3gPQAcRE2YebNkqWOWKRA7vmt?usp=sharing"
                        #output = "COE770_GNN/"
                        # shutil.copy(f"./model_weights_ADCRNN_{filter_size}_{gamma}",\
                        #             "/content/drive/MyDrive/COE770_GNN/")
                torch.save(model, f"./the_whole_model_{model_name}")
                torch.save(model, gdrive_path+f"the_whole_model_{model_name}")
                # self.free_cache()
            else:
                model.load_state_dict(torch.load(gdrive_path+model_name))
                model.to(device)

            model.eval()
            cost = 0

            for time, snapshot in enumerate(test_dataset):
                snapshot.to(device)
                y_hat,_ = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                cost = cost + torch.mean((y_hat-snapshot.y)**2).item()
                cost = cost / (time+1)
                del snapshot
                # self.free_cache()
            stats["MSE"].append(cost)
            stats["model"].append(model_name)
        return stats