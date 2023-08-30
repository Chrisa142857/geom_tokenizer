from geom_tokenizer import geom_tokenizer_onenode, token_zeropad
import torch
from torch.utils.data import Dataset, DataLoader


def main():
    from data_handling import get_data_pyg
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default='pubmed')
    parser.add_argument("--device", type=str, default='cuda:0')
    args = parser.parse_args()
    batch_size = 4
    if args.dataname in ['zinc']: # graph level task
        datasets = get_data_pyg(args.dataname, split=0)
        train_set, val_set, test_set = datasets
        train_set = DataBatchSet([d.x for d in train_set], [d.edge_index for d in train_set], [d.y for d in train_set], node_feat2bin=True)
        trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_set = DataBatchSet([d.x for d in val_set], [d.edge_index for d in val_set], [d.y for d in val_set], node_feat2bin=True)
        valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_set = DataBatchSet([d.x for d in test_set], [d.edge_index for d in test_set], [d.y for d in test_set], node_feat2bin=True)
        testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        loop_data(trainloader)
        
    elif args.dataname in ['pubmed', 'cora', 'citeseer']: # node level task
        for i in range(10):    
            data = get_data_pyg(args.dataname, split=i)
            train_set = DataBatchSet(data.x, data.edge_index, data.y, mask=data.train_mask)
            trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            val_set = DataBatchSet(data.x, data.edge_index, data.y, mask=data.val_mask)
            valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
            test_set = DataBatchSet(data.x, data.edge_index, data.y, mask=data.test_mask)
            testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
            loop_data(trainloader)

def loop_data(loader):
    for data in loader:
        node_feat, geom_d3, view_d3, pad_mask_d3, label, did = data
        print([d.shape for d in data])

def binary(x, bits):
    mask = 2**torch.arange(bits-1,-1,-1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

class DataBatchSet(Dataset):

    def __init__(self, node_feat, edge_index, label, node_idx=None, mask=None, node_feat2bin=False, N=10, geom_dim=3, seq_len=512) -> None:
        self.node_feat = node_feat
        self.edge_index = edge_index
        self.label = label
        assert len(node_feat) == len(label)
        if isinstance(node_feat, list): 
            ## if task is graph level, sequence will include all nodes of a graph
            self.graph_level = True
            self.node_idx = []
            for gi in range(len(node_feat)):
                for ni in range(len(node_feat[gi])):
                    self.node_idx.append([gi, ni])
            self.node_idx = torch.LongTensor(self.node_idx)
        else:
            ## if task is node level, sequence will include tokens of one node
            self.graph_level = False
            if node_idx is not None:
                self.node_idx = node_idx
            elif mask is not None:
                self.node_idx = torch.where(mask)[0]
            else:
                self.node_idx = torch.arange(len(node_feat))
        self.N = N
        self.dim = geom_dim
        self.seq_len = seq_len
        self.node_feat2bin = node_feat2bin
        if node_feat2bin:
            self.node_feat_ch = len(bin(max([f.max() for f in node_feat]))) - 2

    def __getitem__(self, i):
        if self.graph_level:
            gi, ni = self.node_idx[i]
            node_feat = self.node_feat[gi]
            if self.node_feat2bin:
                node_feat = binary(node_feat, self.node_feat_ch)
            edge_index = self.edge_index[gi]
            datay = self.label[gi]
        else:
            ni = self.node_idx[i]
            node_feat = self.node_feat
            edge_index = self.edge_index
            datay = self.label[ni:ni+1]

        geom_tokens, view_dirs, node_embeds, token_count, distance_sorts = geom_tokenizer_onenode(ni, node_feat, edge_index, self.N, self.dim)
        geom_tokens, masks, labels = token_zeropad(geom_tokens, token_count, self.seq_len, datay, distance_sorts)
        # geom_batches_d4, masks_d4, _ = token_padder(geom_tokens_d4, token_d4_count, seq_len, data.y, distance_sorts)
        view_dirs, _, _ = token_zeropad(view_dirs, token_count, self.seq_len, datay, distance_sorts)
        # view_batches_d4, _, _ = token_padder(view_dirs_d4, token_d4_count, seq_len, data.y, distance_sorts)
        node_embeds, _, _ = token_zeropad(node_embeds, token_count, self.seq_len, datay, distance_sorts)
        return node_embeds[0], geom_tokens[0], view_dirs[0], masks[0], labels[0], gi if self.graph_level else ni
  
    def __len__(self):
        return len(self.node_idx)

if __name__ == '__main__':
    main()