import os
import pandas as pd
import networkx as nx
import zipfile
from libpysal import weights
from libpysal.weights import lat2W, Rook, KNN, attach_islands
import geopandas as gpd
import numpy as np
import gdown
from tqdm import tqdm
import torch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, StaticGraphTemporalSignalBatch

class CovidDatasetLoader(object):
    """A dataset of COVID 19 epidemic in Brazil (we included 4 lags).
    """
    def __init__(self,method="knn",k=8):
        self.method = method
        self.knn = k
        self.static_feat = None
        if not os.path.exists("caso_full.csv"):
            self._read_web_data()
    
    def _read_web_data(self):
        url = "https://drive.google.com/uc?export=download&id=1qLrPQ0IReIzBy26GQDd5MaqQ4IU89s_2"
        output = "caso_full.csv"
        gdown.download(url, output)
        ##get cities graph
        url =  "https://drive.google.com/uc?export=download&id=1cHVcrjJxDMoLdI_h6YuCdeoz_PE71XqN"
        output = "BR_Municipios_2020" 
        gdown.download(url, output+".zip")
        cwd = os.getcwd()+"/"
        with zipfile.ZipFile(cwd+output+".zip", 'r') as zip_ref:
            zip_ref.extractall(cwd)

    def  _get_edges(self):
        municipios = gpd.read_file('BR_Municipios_2020/BR_Municipios_2020.shp')
        one_hot = pd.get_dummies(municipios['SIGLA_UF'],prefix='UF_')
        print("onehot: ",one_hot.shape)
        m = municipios.geometry.to_crs(epsg=5641)
        centroids = np.column_stack((m.centroid.x, m.centroid.y))
        lenghts = m.length.values.reshape(-1,1)
        areas = municipios.AREA_KM2.values.reshape(-1,1)
        queen = None
        if self.method == "knn":
            queen = weights.distance.Kernel.from_dataframe(
                            municipios, fixed=False, k=self.knn)
        else:
            w = weights.Queen.from_dataframe(municipios)
            w_knn1 = KNN.from_dataframe(municipios,k=1)
            queen = attach_islands(w, w_knn1)
            print(queen.islands)

        G = queen.to_networkx()
        G = nx.relabel_nodes(G,lambda x: int(x))
        for src,dest,edge_data in G.edges.data():
            edge_data['weight'] = 1/(np.linalg.norm(centroids[src] - centroids[dest]) + 1)
        G = G.to_directed()
        nx.stochastic_graph(G, copy=False)
        self._edges = np.array(G.edges).T
        self._edge_weights = np.array([w['weight'] for u, v, w in G.edges(data=True)])
        self.static_feat = np.hstack([centroids,one_hot,lenghts,areas])

    def _get_edge_weights(self):
        self._edge_weights = np.ones(self._edges.shape[1])

    def _get_targets_and_features(self):
        data = pd.read_csv('caso_full.csv')
        data = data[(data['city'].isnull() == False) & (data['is_repeated'] == False)]
        a = ['city_ibge_code','order_for_place','date','epidemiological_week',
        'last_available_confirmed_per_100k_inhabitants','last_available_confirmed',
        'last_available_death_rate']
        columns_to_drop = ['state','city','place_type','last_available_confirmed',
                        'last_available_date','estimated_population',
                        'is_last','is_repeated']
        data.drop(columns=columns_to_drop,inplace=True)
        data['date'] = pd.to_datetime(data['date'],format='%Y-%m-%d')
        data.epidemiological_week -= data.epidemiological_week.min()
        print(data.columns)
        grouped = data[["last_available_confirmed_per_100k_inhabitants",\
                        "city_ibge_code","date"]]\
                            .groupby('city_ibge_code')
        min_date = data["date"].min()
        max_date = data["date"].max()
        delta = max_date - min_date
        dates = pd.date_range(start=min_date, end=max_date, freq='D')
        i = 0
        stacked_target = np.zeros((len(grouped),delta.days + 1))
        for group, data in grouped:
            series = data[["last_available_confirmed_per_100k_inhabitants","date"]]
            series = series.set_index("date").reindex(dates).interpolate().ffill().fillna(0)
            #shape = series.shape
            stacked_target[i,:] = series.values.reshape(len(series))
            i += 1
        # print("Stack_targ.shape: ",stacked_target.shape)
        ## Stack_targ.shape:  (5570, 761)
        ## Features are first indexed by time, i.e.,
        ## features[i] gives the [ith time_index] node features
        self.features = [
            np.hstack([self.static_feat,stacked_target[:,i : i + self.lags]])
            for i in range(stacked_target.shape[1] - self.lags)
        ]
        print("feat: ",len(self.features),self.features[0].shape)
        self.targets = [
            stacked_target[:,i + self.lags]
            for i in range(stacked_target.shape[1] - self.lags)
        ]

    def get_dataset(self, lags: int = 4, from_drive = False) -> StaticGraphTemporalSignal:
        """
        Args types:
            * **lags** *(int)* - The number of time lags.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The PedalMe dataset.
        """
        dataset = None
        if from_drive:
            url = "https://drive.google.com/uc?export=download&id=141pexJ02xzG9spi-QsepavTmsjDdLg6D"
            output = "covid.data"
            gdown.download(url, output)
            
        if (not os.path.isfile('covid.data')):
            self.lags = lags
            self._get_edges()
            #self._get_edge_weights()
            self._get_targets_and_features()
            dataset = StaticGraphTemporalSignal(
                self._edges,  self._edge_weights, self.features, self.targets
            )
            torch.save(dataset, 'covid.data')
        else:
            dataset = torch.load('covid.data')
        return dataset
    
    def get_dataset_batch(self, lags: int = 4, batch_size: int = 16) -> StaticGraphTemporalSignalBatch:
        self.lags = lags
        self._get_edges()
        #self._get_edge_weights()
        self._get_targets_and_features()
        dataset = StaticGraphTemporalSignalBatch(
            self._edges,  self._edge_weights, self.features, self.targets, batch_size
        )
        return dataset