from geom_tokenizer import geom_tokenizer_onenode, token_zeropad
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
from positional_encoder import PositionalEncodingTransform
import multiprocessing as mp

def main():
    from data_handling import get_data_pyg
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default='zinc')
    parser.add_argument("--device", type=str, default='cuda:0')
    args = parser.parse_args()
    batch_size = 4
    num_worker = 4
    if args.dataname in ['zinc']: # graph level task
        datasets = get_data_pyg(args.dataname, split=0)
        train_set, val_set, test_set = datasets
        train_set = DataBatchSetGraphLevel([d.x for d in train_set], [d.edge_index for d in train_set], [d.y for d in train_set], node_feat2bin=True)
        trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_worker)
        val_set = DataBatchSetGraphLevel([d.x for d in val_set], [d.edge_index for d in val_set], [d.y for d in val_set], node_feat2bin=True)
        valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_worker)
        test_set = DataBatchSetGraphLevel([d.x for d in test_set], [d.edge_index for d in test_set], [d.y for d in test_set], node_feat2bin=True)
        testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_worker)
        loop_data(trainloader)
        
    elif args.dataname in ['pubmed', 'cora', 'citeseer']: # node level task
        for i in range(10):    
            data = get_data_pyg(args.dataname, split=i)
            train_set = DataBatchSetGraphLevel(data.x, data.edge_index, data.y, mask=data.train_mask)
            trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_worker)
            val_set = DataBatchSetGraphLevel(data.x, data.edge_index, data.y, mask=data.val_mask)
            valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_worker)
            test_set = DataBatchSetGraphLevel(data.x, data.edge_index, data.y, mask=data.test_mask)
            testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_worker)
            loop_data(trainloader)

def loop_data(loader):
    for data in loader:
        node_feat, geom_d3, view_d3, pad_mask_d3, label, did = data
        print([d.shape for d in data])

def binary(x, bits):
    mask = 2**torch.arange(bits-1,-1,-1).to(x.device, x.dtype)
    return x.bitwise_and(mask).ne(0).byte()

def mp_f(args):
    return geom_tokenizer_onenode(*args)

class DataBatchSetGraphLevel(Dataset):

    def __init__(self, node_feat, edge_index, label, node_idx=None, mask=None, node_feat2bin=False, N=10, geom_dim=3, seq_len=512, rw_pe_dim=8, lap_pe_dim=0, do_cache=True, node_feat_ch=None) -> None:
        self.pe_trans = PositionalEncodingTransform(lap_dim=lap_pe_dim, rw_dim=rw_pe_dim)
        self.pe_dim = max(lap_pe_dim, rw_pe_dim)
        self.do_cache = do_cache
        self.node_feat = node_feat
        self.edge_index = edge_index
        self.label = label
        self.N = N
        self.dim = geom_dim
        self.seq_len = seq_len
        # self.node_feat2bin = node_feat2bin
        self.token_padder = token_zeropad
        if node_feat2bin:
            self.node_feat_ch = int(math.log(max([f.max() for f in node_feat]),2))+1 if node_feat_ch is None else node_feat_ch
        else:
            self.node_feat_ch = node_feat[0].shape[-1]
        print("Data node feat channel", self.node_feat_ch)
        assert len(node_feat) == len(label)
        
        ## if task is graph level, sequence will include all nodes of a graph
        self.graph_level = True
        self.node_idx = []
        for gi in range(len(node_feat)):
            if node_feat2bin:
                node_feat[gi] = binary(node_feat[gi], self.node_feat_ch).float()
            for ni in range(len(node_feat[gi])):
                self.node_idx.append([gi, ni])
        self.node_idx = torch.LongTensor(self.node_idx)

        self.geom_tokens, self.view_dirs, self.node_embeds, self.token_count, self.distance_sorts = [], [], [], [], []
        graph_geom_tokens = [[] for gi in range(len(node_feat))]
        graph_view_dirs = [[] for gi in range(len(node_feat))]
        graph_node_embeds = [[] for gi in range(len(node_feat))]
        # graph_token_count = [[] for gi in range(len(node_feat))]
        # graph_pe = [[] for gi in range(len(node_feat))]
        self.caches = []
        # with mp.Pool(8) as pool:
        #     ins = [(ni, node_feat[gi], edge_index[gi], N, geom_dim) for gi, ni in self.node_idx]
        #     outs = tqdm(list(pool.imap(mp_f, ins)))
        # for geom_tokens, view_dirs, node_embeds, token_count, distance_sorts in outs:
        for gi, ni in tqdm(self.node_idx, desc='Init tokens'):
            geom_tokens, view_dirs, node_embeds, token_count, distance_sorts = geom_tokenizer_onenode(ni, node_feat[gi], edge_index[gi], N, geom_dim)
            graph_geom_tokens[gi].append(geom_tokens)
            graph_view_dirs[gi].append(view_dirs)
            graph_node_embeds[gi].append(node_embeds)
            # graph_token_count[gi].append(token_count)
            # graph_distance_sorts.append(distance_sorts)
        for gi in range(len(node_feat)):
            geom_tokens = torch.cat(graph_geom_tokens[gi])
            view_dirs = torch.cat(graph_view_dirs[gi])
            node_embeds = torch.cat(graph_node_embeds[gi])
            # token_count = torch.cat(graph_token_count[gi])
            if self.pe_dim > 0: 
                pe = self.pe_trans(edge_index[gi], len(node_feat[gi]))
                pe = torch.cat([torch.stack([pe[ni] for _ in range(len(graph_node_embeds[gi][ni]))], 0) for ni in range(len(graph_node_embeds[gi]))])
                node_embeds = torch.cat([node_embeds, pe], -1)
            if self.do_cache:
                datay = [self.label[gi]]
                out = pad_tokens(self, geom_tokens, view_dirs, node_embeds, torch.FloatTensor([len(node_embeds)]), None, datay)
                self.caches.append([o[0] for o in out])
            else:
                self.geom_tokens.append(geom_tokens)
                self.view_dirs.append(view_dirs)
                self.node_embeds.append(node_embeds)

    def __getitem__(self, i):
        if not self.do_cache:
            gi, ni = self.node_idx[i]
            datay = self.label[gi]
            geom_tokens = self.geom_tokens[i]
            view_dirs = self.view_dirs[i]
            node_embeds = self.node_embeds[i]
            # node_embeds, geom_tokens, view_dirs, masks, labels = pad_tokens(self, geom_tokens, view_dirs, node_embeds, token_count, distance_sorts, datay)
            out = pad_tokens(self, geom_tokens, view_dirs, node_embeds, torch.FloatTensor([len(node_embeds)]), None, datay)
            out = [o[0] for o in out]
            node_embeds, geom_tokens, view_dirs, masks, labels = out
        else:
            node_embeds, geom_tokens, view_dirs, masks, labels = self.caches[i]
        return node_embeds, geom_tokens, view_dirs, masks, labels, 0 #gi if self.graph_level else ni
  
    def __len__(self):
        return len(self.geom_tokens) if not self.do_cache else len(self.caches)

class DataBatchSetNodeLevel(Dataset):

    def __init__(self, node_feat, edge_index, label, node_idx=None, mask=None, node_feat2bin=False, N=10, geom_dim=3, seq_len=512, lap_pe_dim=0, do_cache=True) -> None:
        self.pe_trans = PositionalEncodingTransform(lap_dim=lap_pe_dim)
        self.pe_dim = lap_pe_dim
        self.do_cache = do_cache
        self.node_feat = node_feat
        self.edge_index = edge_index
        self.label = label
        self.N = N
        self.dim = geom_dim
        self.seq_len = seq_len
        self.token_padder = token_zeropad
        if node_feat2bin:
            self.node_feat_ch = int(math.log(max([f.max() for f in node_feat]),2))+1
        else:
            self.node_feat_ch = node_feat[0].shape[-1]
        print("Data node feat channel", self.node_feat_ch)

        ## sequence will include tokens of one node
        if node_idx is not None:
            self.node_idx = node_idx
        elif mask is not None:
            self.node_idx = torch.where(mask)[0]
        else:
            self.node_idx = torch.arange(len(node_feat))
        self.geom_tokens, self.view_dirs, self.node_embeds, self.token_count, self.distance_sorts = [], [], [], [], []
        self.caches = []
        if self.pe_dim > 0: pe = self.pe_trans(edge_index, len(node_feat))
        for ni in tqdm(self.node_idx, desc='Init tokens'):
            geom_tokens, view_dirs, node_embeds, token_count, distance_sorts = geom_tokenizer_onenode(ni, node_feat, edge_index, N, geom_dim)
            if self.pe_dim > 0: node_embeds = torch.cat([node_embeds, torch.stack([pe[ni] for _ in range(len(node_embeds))], 0)], -1)
            if not self.do_cache:
                self.geom_tokens.append(geom_tokens)
                self.view_dirs.append(view_dirs)
                self.node_embeds.append(node_embeds)
                self.token_count.append(token_count)
                self.distance_sorts.append(distance_sorts)
            else:
                datay = self.label[ni:ni+1]
                outs = pad_tokens(self, geom_tokens, view_dirs, node_embeds, token_count, distance_sorts, datay)
                self.caches.append([o[0] for o in outs])
            

    def __getitem__(self, i):
        if not self.do_cache:
            ni = self.node_idx[i]
            datay = self.label[ni:ni+1]
            geom_tokens = self.geom_tokens[i]
            view_dirs = self.view_dirs[i]
            node_embeds = self.node_embeds[i]
            token_count = self.token_count[i]
            distance_sorts = self.distance_sorts[i]
            node_embeds, geom_tokens, view_dirs, masks, labels = pad_tokens(self, geom_tokens, view_dirs, node_embeds, token_count, distance_sorts, datay)
        else:
            node_embeds, geom_tokens, view_dirs, masks, labels = self.caches[i]
        return node_embeds, geom_tokens, view_dirs, masks, labels, 0 #gi if self.graph_level else ni
  

    def __len__(self):
        return len(self.node_idx)


def pad_tokens(cls, geom_tokens, view_dirs, node_embeds, token_count, distance_sorts, datay):
    geom_tokens, masks, labels = cls.token_padder(geom_tokens, token_count, cls.seq_len, datay, distance_sorts)
    # geom_batches_d4, masks_d4, _ = self.token_padder(geom_tokens_d4, token_d4_count, seq_len, data.y, distance_sorts)
    view_dirs, _, _ = cls.token_padder(view_dirs, token_count, cls.seq_len, datay, distance_sorts)
    # view_batches_d4, _, _ = self.token_padder(view_dirs_d4, token_d4_count, seq_len, data.y, distance_sorts)
    node_embeds, _, _ = cls.token_padder(node_embeds, token_count, cls.seq_len, datay, distance_sorts)
    return node_embeds, geom_tokens, view_dirs, masks, labels

if __name__ == '__main__':
    main()