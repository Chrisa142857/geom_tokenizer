from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, ZINC, AQSOL, WikiCS, GNNBenchmarkDataset, Planetoid, HeterophilousGraphDataset
import torch
import numpy as np
import torch_geometric.transforms as T

def get_data_pyg(name, split=0):
  path = '../data/' +name
  transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
  if name in ['chameleon','squirrel']:
    dataset = WikipediaNetwork(root=path, name=name, transform=transform)
  if name in ['cornell', 'texas', 'wisconsin']:
    dataset = WebKB(path ,name=name, transform=transform)
  if name == 'film':
    dataset = Actor(root=path, transform=transform)
  if name == 'zinc':
    dataset = [ZINC(root=path, split='train', subset=True, transform=transform), ZINC(root=path, split='val', subset=True, transform=transform), ZINC(root=path, split='test', subset=True, transform=transform)]
  if name in ['pubmed', 'cora', 'citeseer']:
    dataset = Planetoid(root=path, name=name, split='geom-gcn', transform=transform)
  if name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
    dataset = HeterophilousGraphDataset(root=path, name=name, transform=transform)

  if name in ['pubmed', 'cora', 'citeseer', 'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
    data = dataset
    data.train_mask = data.train_mask[:, split]
    data.val_mask = data.val_mask[:, split]
    data.test_mask = data.test_mask[:, split]
  elif name in ['zinc']:
    data = dataset
  else:  
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
