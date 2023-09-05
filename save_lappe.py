from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, ZINC, AQSOL, WikiCS, GNNBenchmarkDataset, Planetoid, HeterophilousGraphDataset, Amazon, Coauthor, CoraFull
import torch_geometric.transforms as T
import torch_geometric
import torch

import scipy.sparse as sp
import numpy as np

def LapPE(edge_index, pos_enc_dim, num_nodes):
    """
        Graph positional encoding v/ Laplacian eigenvectors
        https://github.com/XiaoxinHe/Graph-ViT-MLPMixer/blob/3025337b60ab6d1c156a321368d0889d0e6fab0f/core/data_utils/pe.py
    """

    # Laplacian
    degree = torch_geometric.utils.degree(edge_index[0], num_nodes)
    A = torch_geometric.utils.to_scipy_sparse_matrix(
        edge_index, num_nodes=num_nodes)
    N = sp.diags(np.array(degree.clip(1) ** -0.5, dtype=float))
    L = sp.eye(num_nodes) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    PE = torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float()
    if PE.size(1) < pos_enc_dim:
        zeros = torch.zeros(num_nodes, pos_enc_dim)
        zeros[:, :PE.size(1)] = PE
        PE = zeros
    return PE

pe_dim = 15
dataset = 'pubmed'
path = f'../data/{dataset}'
transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
# graph = HeterophilousGraphDataset(root=path, name=dataset, transform=transform)
graph = Planetoid(root=path, split='geom-gcn', name=dataset, transform=transform)

lpe = LapPE(graph.edge_index, pos_enc_dim=pe_dim, num_nodes=len(graph.x))

torch.save(lpe, f'{path}/lap_pe_dim={pe_dim}.pth')