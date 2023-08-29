from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor
import torch
import numpy as np

def get_data_pyg(name, split=0):
  path = '../data/' +name
  if name in ['chameleon','squirrel']:
    dataset = WikipediaNetwork(root=path, name=name)
  if name in ['cornell', 'texas', 'wisconsin']:
    dataset = WebKB(path ,name=name)
  if name == 'film':
    dataset = Actor(root=path)

  data = dataset[0]
  if name in ['chameleon', 'squirrel']:
    splits_file = np.load(f'{path}/{name}/geom_gcn/raw/{name}_split_0.6_0.2_{split}.npz')
  if name in ['cornell', 'texas', 'wisconsin']:
    splits_file = np.load(f'{path}/{name}/raw/{name}_split_0.6_0.2_{split}.npz')
  if name == 'film':
    splits_file = np.load(f'{path}/raw/{name}_split_0.6_0.2_{split}.npz')
  if name in ['Cora', 'Citeseer', 'Pubmed']:
      splits_file = np.load(f'{path}/{name}/raw/{name}_split_0.6_0.2_{split}.npz')
  train_mask = splits_file['train_mask']
  val_mask = splits_file['val_mask']
  test_mask = splits_file['test_mask']

  data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
  data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
  data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

  return data


import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import  CoraFullDataset, AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset,CoauthorCSDataset,CoauthorPhysicsDataset


def get_dataset_dgl(dataset):
    if dataset in {"pubmed", "corafull", "computer", "photo", "cs", "physics","cora", "citeseer"}:



        file_path = "dataset/"+dataset+".pt"

        data_list = torch.load(file_path)

        # data_list = [adj, features, labels, idx_train, idx_val, idx_test]
        adj = data_list[0]
        features = data_list[1]
        labels = data_list[2]

        idx_train = data_list[3]
        idx_val = data_list[4]
        idx_test = data_list[5]

        if dataset == "pubmed":
            graph = PubmedGraphDataset()[0]
        elif dataset == "corafull":
            graph = CoraFullDataset()[0]
        elif dataset == "computer":
            graph = AmazonCoBuyComputerDataset()[0]
        elif dataset == "photo":
            graph = AmazonCoBuyPhotoDataset()[0]
        elif dataset == "cs":
            graph = CoauthorCSDataset()[0]
        elif dataset == "physics":
            graph = CoauthorPhysicsDataset()[0]
        elif dataset == "cora":
            graph = CoraGraphDataset()[0]
        elif dataset == "citeseer":
            graph = CiteseerGraphDataset()[0]

        graph = dgl.to_bidirected(graph)



    elif dataset in {"aminer", "reddit", "Amazon2M"}:

 
        file_path = './dataset/'+dataset+'.pt'

        data_list = torch.load(file_path)

        #adj, features, labels, idx_train, idx_val, idx_test

        adj = data_list[0]
        
        #print(type(adj))
        features = torch.tensor(data_list[1], dtype=torch.float32)
        labels = torch.tensor(data_list[2])
        idx_train = torch.tensor(data_list[3])
        idx_val = torch.tensor(data_list[4])
        idx_test = torch.tensor(data_list[5])

        graph = dgl.from_scipy(adj)

        labels = torch.argmax(labels, -1)
        

    return adj, features, labels, idx_train, idx_val, idx_test