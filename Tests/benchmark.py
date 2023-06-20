import torch
from itertools import product
import gc
from dataset.covidBR_dataset import *
from model.covid_model import *
from model.model_factory import *
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from dataset.covidBR_dataset import *
from model.covid_model import *
import matplotlib.pyplot as plt
import numpy as np
import os
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

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
                                         num_filters = filter_size).to(device)
                else:
                    model = get_model(False,num_features=num_feats,num_filters=filter_size,gamma=gamma)
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
                torch.save(model, f"./the_whole_model_ADCRNN_{filter_size}_{gamma}")
                self.free_cache()
            else:
                model = get_model(False,num_features=num_feats,num_filters=filter_size,gamma=gamma)
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
                              num_epochs=100,output_size=32,make_plot=True):

        loader = ChickenpoxDatasetLoader()

        dataset = loader.get_dataset(lags)

        # loader = CovidDatasetLoader(method="other")
        # dataset = loader.get_dataset(lags=4)
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

        num_nodes = dataset[0].x.shape[0]
        num_feats = dataset[0].x.shape[1]

        stats_test = {"MSE":[],"model":[]}
        stats_all = {"MSE":[],"model":[]}
        models,model_names = make_models(num_feats,\
                                         output_size,\
                                         filter_size=filter_size,\
                                        num_nodes=num_nodes)
        for model,model_name in zip(models,model_names):
            if train_model:
                model.to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

                h, c = None, None
                model.train()

                if model_name == "AGCRN":
                    c = torch.empty(num_nodes, num_feats).to(device)
                    torch.nn.init.xavier_uniform_(c)

                for epoch in tqdm(range(num_epochs)):
                    cost = 0
                    for time, snapshot in enumerate(train_dataset):
                        snapshot.to(device)
                        y_hat,h,c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr,h,c)
                        cost = cost + torch.mean((y_hat-snapshot.y)**2)
                    cost = cost / (time+1)
                    cost.backward(retain_graph=True)
                    optimizer.step()
                    optimizer.zero_grad()
                    if epoch % 10 == 0:
                        filepath = f"./{model_name}"
                        if (os.path.isfile(filepath)):
                            os.remove(filepath)
                        torch.save(model.state_dict(), filepath)
                        torch.save(model.state_dict(), gdrive_path+model_name)
                torch.save(model, f"./the_whole_model_{model_name}")
                torch.save(model, gdrive_path+f"the_whole_model_{model_name}")
            model.load_state_dict(torch.load(gdrive_path+model_name))
            model.to(device)
            model.eval()
            cost = 0
            for time, snapshot in enumerate(test_dataset):
                snapshot.to(device)
                y_hat,h,c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr,h,c)
                cost = cost + torch.mean((y_hat-snapshot.y)**2).item()
                del snapshot
            cost = cost / (time+1)
            print(cost)
            print(model_name)
            print(model)
            stats_test["MSE"].append(cost)
            stats_test["model"].append(model_name)
            if make_plot:
                preds = []
                y = []
                cost = 0
                for time, snapshot in enumerate(dataset):
                    snapshot.to(device)
                    y_hat,h,c  = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr,h,c )
                    preds.append(y_hat.mean().cpu().detach().numpy())
                    y.append(snapshot.y.mean().cpu().detach().numpy())
                    cost+=torch.mean((y_hat-snapshot.y)**2).item()
                    del snapshot
                plt.plot(preds,label=f"{model_name} - MSE: {cost/(time+1):.4f}")
                stats_all['MSE'].append(cost/(time+1))
                stats_all["model"].append(model_name)
                if model_name == "TGCN":
                    plt.plot(y,label=f"Y")
                # print("MSE: {:.4f}".format(cost))
                plt.legend(bbox_to_anchor=(1, 1))
                plt.ylabel("Total de casos agregados")
                plt.xticks(rotation=45)
        plt.savefig("casos_agregados.png")
        plt.show()
        return stats_all,stats_test
    
    def run_DCRNN_covid(self,lags=4,train_model=True,filter_sizes = [2,4,8,16],\
                              num_epochs=100,output_size=32):

        loader = CovidDatasetLoader(method="other")
        dataset = loader.get_dataset(lags=lags)
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
        num_feats = dataset[0].x.shape[1]
        stats = {"MSE":[],"model":[], "filter_size":[]}
        
        for filter_size in filter_sizes:
            model = MY_DCRNN_(node_features = num_feats, output_size=output_size,\
                               filter_size=filter_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            model.train()
            h = c = None
            model_name= f"model_weights_DCRNN_Covid_{filter_size}"
            filepath = "./"+model_name
            if train_model:
                for epoch in tqdm(range(num_epochs)):
                    for time, snapshot in enumerate(train_dataset):
                        snapshot.to(device)
                        y_hat,h,c  = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr,h,c)
                        cost = torch.mean((y_hat-snapshot.y)**2)
                        cost.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        del snapshot
                        if epoch % 10 == 0:
                            if (os.path.isfile(filepath)):
                                os.remove(filepath)
                            torch.save(model.state_dict(), filepath)
                            torch.save(model.state_dict(), gdrive_path+model_name)
                            print(gdrive_path+model_name)
                torch.save(model, f"./the_whole_model_DCRNN_Covid_{filter_size}")
            model.load_state_dict(torch.load(gdrive_path+model_name))
            model.to(device)
            model.eval()
            cost = 0
            for time, snapshot in enumerate(test_dataset):
                snapshot.to(device)
                y_hat,h,c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr,h,c)
                cost = cost + torch.mean((y_hat-snapshot.y)**2).item()
                del snapshot
            cost = cost / (time+1)
            stats["MSE"].append(cost)
            stats["model"].append("DCRNN")
            stats["filter_size"].append(filter_size)
        return stats