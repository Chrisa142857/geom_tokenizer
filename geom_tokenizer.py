from torch_geometric.data import Data
import torch
import torch.optim as optim
from tqdm import trange, tqdm
import transformers
import random
from datetime import datetime
from itertools import combinations


def geom_tokenizer(node_feat: torch.Tensor, edge_index: torch.Tensor, N: int, dim: int=3):
    nids = torch.arange(len(node_feat))
    geom_tokens = [] 
    token_count = []
    dis_sorts = []
    view_embeds = []
    node_embeds = []
    for ni in nids:
        distances1 = ((node_feat[ni] - node_feat[ni+1:]) **2 ).sum(1)
        if len(distances1) > 0:
            mind = distances1.min()
            maxd = distances1.max()
        else:
            mind = 1e10
            maxd = 0
        distances2 = ((node_feat[ni] - node_feat[:ni]) **2 ).sum(1)
        if len(distances2) > 0:
            mind = min(mind, distances2.min())
            maxd = max(maxd, distances2.max())
        elif len(distances1) == 0:
            print(f'Data error, data shape {node_feat.shape}, this is the {ni}-th node')
            exit()

        distances = torch.cat([distances1, torch.FloatTensor([maxd+1]), distances2]) # X
        ## spatially close nodes are neighborhood
        dis_sort = distances.argsort()
        dis_sorts.append(dis_sort)
        nei_nid = dis_sort[:N]
        ## connected nodes are neighborhood
        connected_node = edge_index[1, edge_index[0] == ni]
        nei_conn = (connected_node==nei_nid[..., None])
        connected_node = connected_node[~(nei_conn.any(0))]
        connected_node = connected_node[distances[connected_node].argsort()[:N]] # also sort connected nodes
        nei_nid = torch.cat([nei_nid, connected_node]) # concat in the sorted rank
        ## geom level 3, triangle, it has dim-1 = 2 view tokens
        ## view token, max = N, the neighbor num
        # view_id = torch.stack(torch.meshgrid(torch.arange(len(nei_nid)), torch.arange(len(nei_nid))), -1) # N x N x 2
        # indices = torch.triu_indices(len(nei_nid), len(nei_nid), offset=1) # M = N x (N-1) / 2
        # view_id = view_id[indices[0], indices[1]].T # 2 x M
        view_id = torch.LongTensor(list(combinations(torch.arange(len(nei_nid)), dim-1))).T # dim-1 x M
        # ## view nodes are sorted by distance
        view_node = [nei_nid[view_id[di]] for di in range(dim-1)] # dim-1 x M, each is a node id
        ## pos token, max = node num
        # pos_token = torch.LongTensor([ni for _ in range(len(view_node[0]))]) # M
        ## geom token, max = 2**3 = 8
        nei_pair = torch.LongTensor(list(combinations(torch.arange(dim-1), 2))) # dim-1 * (dim-2) / 2 x 2, if neighbors connected
        geom_token = torch.stack([torch.zeros_like(view_node[0]) for _ in range(dim-1+len(nei_pair))]) # dim x M
        where_edges = torch.cat([nei_conn.any(1), torch.ones_like(connected_node, dtype=bool)])
        # view_id[0] < view_id[1]
        for di in range(dim-1):
            where_edge = where_edges[view_id[di]]
            geom_token[di, where_edge] = 1
        ## if view points are connected
        for di in range(len(nei_pair)):
            where_edge = []
            views = torch.stack([view_node[nei_pair[di, 0]], view_node[nei_pair[di, 1]]], -1) # pair of views
            for view in views: # for 2 in M x 2
                where_edge.append((view == edge_index.T).any())
            where_edge = torch.where(torch.stack(where_edge))[0]
            geom_token[dim-1+di, where_edge] = 1
        ## convert token to 10-base number
        geom_token = bin2dec(geom_token) # M
        geom_tokens.append(geom_token)
        token_count.append(len(view_node[0]))
        ## embed of geom direction from view points to cur node
        view_embed = torch.stack([node_feat[view_node[di]] - node_feat[ni] for di in range(dim-1)]).sum(0) # M x C
        view_embeds.append(view_embed)
        node_embed = torch.stack([node_feat[ni] for _ in range(len(view_node[0]))]) # M x C
        node_embeds.append(node_embed)
    
    geom_tokens = torch.cat(geom_tokens)
    dis_sorts = torch.stack(dis_sorts)
    token_count = torch.FloatTensor(token_count)
    view_dirs = torch.cat(view_embeds) #
    node_embeds = torch.cat(node_embeds) #
    # return pos_tokens, geom_tokens, view_tokens, node_embeds, token_count, dis_sorts
    return geom_tokens, view_dirs, node_embeds, token_count, dis_sorts

def geom_tokenizer_onenode(ni: int, node_feat: torch.Tensor, edge_index: torch.Tensor, N: int, dim: int=3):
    distances1 = ((node_feat[ni] - node_feat[ni+1:]) **2 ).sum(1)
    if len(distances1) > 0:
        mind = distances1.min()
        maxd = distances1.max()
    else:
        mind = 1e10
        maxd = 0
    distances2 = ((node_feat[ni] - node_feat[:ni]) **2 ).sum(1)
    if len(distances2) > 0:
        mind = min(mind, distances2.min())
        maxd = max(maxd, distances2.max())
    elif len(distances1) == 0:
        print(f'Data error, data shape {node_feat.shape}, this is the {ni}-th node')
        exit()


    distances = torch.cat([distances1, torch.FloatTensor([maxd+1]), distances2]) # X
    ## spatially close nodes are neighborhood
    dis_sort = distances.argsort()
    # dis_sorts.append(dis_sort)
    nei_nid = dis_sort[:N]
    ## connected nodes are neighborhood
    connected_node = edge_index[1, edge_index[0] == ni]
    nei_conn = (connected_node==nei_nid[..., None])
    connected_node = connected_node[~(nei_conn.any(0))]
    connected_node = connected_node[distances[connected_node].argsort()[:N]] # also sort connected nodes
    nei_nid = torch.cat([nei_nid, connected_node]) # concat in the sorted rank
    ## view token, max = N, the neighbor num
    view_id = torch.LongTensor(list(combinations(torch.arange(len(nei_nid)), dim-1))).T # dim-1 x M
    # ## view nodes are sorted by distance
    view_node = [nei_nid[view_id[di]] for di in range(dim-1)] # dim-1 x M, each is a node id
    ## geom token, max = 2**3 = 8
    nei_pair = torch.LongTensor(list(combinations(torch.arange(dim-1), 2))) # dim-1 * (dim-2) / 2 x 2, if neighbors connected
    geom_token = torch.stack([torch.zeros_like(view_node[0]) for _ in range(dim-1+len(nei_pair))]) # dim x M
    where_edges = torch.cat([nei_conn.any(1), torch.ones_like(connected_node, dtype=bool)])
    # view_id[0] < view_id[1] 
    for di in range(dim-1):
        where_edge = where_edges[view_id[di]]
        geom_token[di, where_edge] = 1
    ## if view points are connected
    for di in range(len(nei_pair)):
        where_edge = []
        views = torch.stack([view_node[nei_pair[di, 0]], view_node[nei_pair[di, 1]]], -1) # pair of views
        for view in views: # for 2 in M x 2
            where_edge.append((view == edge_index.T).any())
        where_edge = torch.where(torch.stack(where_edge))[0]
        geom_token[dim-1+di, where_edge] = 1
    ## convert token to 10-base number
    geom_token = bin2dec(geom_token) # M
    token_count = torch.FloatTensor([len(view_node[0])])
    ## embed of geom direction from view points to cur node
    view_dir = torch.stack([node_feat[view_node[di]] - node_feat[ni] for di in range(dim-1)]).sum(0) # M x C
    node_embed = torch.stack([node_feat[ni] for _ in range(len(view_node[0]))]) # M x C

    return geom_token, view_dir, node_embed, token_count, dis_sort

def bin2dec(b):
    bits, batch = b.shape
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    mask = torch.stack([mask for _ in range(batch)], 1)
    return torch.sum(mask * b, 0)

def token_zeropad(tokens: torch.Tensor, token_count: torch.Tensor, seq_len: int, datay: torch.Tensor, *args):
    '''
    tokens: M
    M = token number
    # n = token type number
    '''
    batches = []
    labels = []
    masks = [] # mask for zero padding
    token_cumsum = torch.cumsum(token_count, 0)
    token_cumsum = token_cumsum.long()
    for i in range(len(token_cumsum)):
        prev = 0 if i == 0 else token_cumsum[i-1]
        seq = tokens[prev:token_cumsum[i].item()]
        seq = seq[:seq_len]
        mask = torch.ones(seq_len, dtype=bool, device=seq.device)
        if len(seq) < seq_len:
            if len(seq.shape) > 1:
              one = torch.cat([seq, torch.zeros((seq_len-len(seq), seq.shape[1]), device=seq.device, dtype=seq.dtype)])
            else:
              one = torch.cat([seq, torch.zeros(seq_len-len(seq), device=seq.device, dtype=seq.dtype)])
            mask[len(seq):] = False
        else:
            one = seq
        labels.append(datay[i])
        batches.append(one)
        masks.append(mask)
    batches = torch.stack(batches)
    masks = torch.stack(masks)
    labels = torch.stack(labels)
    return batches, masks, labels

def token_neighborpad(tokens, token_count, seq_len, datay, dis_sorts):
    '''
    tokens: M
    M = token number
    # n = token type number
    '''
    batches = []
    labels = []
    masks = [] # mask for class token
    token_count = token_count.long()
    token_cumsum = torch.cumsum(token_count, 0)
    for i in range(len(token_count)):
        prev = 0 if i == 0 else token_count[i-1]
        seq = tokens[prev:prev+token_count[i]]
        mask = torch.ones(seq_len, dtype=bool, device=seq.device)
        if len(seq) < seq_len:
            pid = 0
            pcumsum = torch.cumsum(token_count[dis_sorts[i]], 0)
            while pcumsum[pid] < seq_len-len(seq): pid += 1
            pad_nid = dis_sorts[i, :pid]
            for ni in pad_nid:
                prev = token_cumsum[ni-1] if ni != 0 else 0
                assert token_cumsum[ni] == prev + token_count[ni]
                seq = torch.cat([seq, tokens[prev:token_cumsum[ni]]])
        if len(seq) < seq_len:
            if len(seq.shape) > 1:
              seq = torch.cat([seq, torch.zeros((seq_len-len(seq), seq.shape[1]), device=seq.device, dtype=seq.dtype)])
            else:
              seq = torch.cat([seq, torch.zeros(seq_len-len(seq), device=seq.device, dtype=seq.dtype)])
            mask[len(seq):] = False
        one = seq[:seq_len]
        labels.append(datay[i])
        batches.append(one)
        masks.append(mask)
    batches = torch.stack(batches)
    masks = torch.stack(masks)
    labels = torch.LongTensor(labels)
    return batches, masks, labels

class ToyModel(torch.nn.Module):
    def __init__(self, node_num, node_channel, geom_dim, cls_num, nhead=8) -> None:
        '''
        Toy transformer for node classification
        '''
        super().__init__()
        ## Embed all tokens
        # self.encoder = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.encoder = transformers.BertModel(transformers.BertConfig())
        # self.encoder.config.output_attentions = True
        hdim = self.encoder.config.hidden_size
        tokens_num = 1
        # token_embed1 = torch.nn.Embedding(node_num, hdim//tokens_num)
        token_embed2 = torch.nn.Embedding(2**geom_dim, hdim//tokens_num)
        token_embed3 = torch.nn.Linear(node_channel, hdim//tokens_num)
        # token_embed3 = torch.nn.Embedding(node_num**2, hdim//tokens_num)
        # token_embed4 = torch.nn.Embedding(node_num, hdim//4)
        self.token_embeds = torch.nn.ModuleList([token_embed2, token_embed3]) #, token_embed4
        self.node_embed =  torch.nn.Linear(node_channel, hdim) #, token_embed4
        ## Transformer Encoder
        # encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hdim, nhead=nhead)
        # self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.classifier = torch.nn.Linear(hdim, cls_num)

    def forward(self, inputs, masks=None):
        # x, pos_tokens, geom_tokens, view_tokens = inputs
        ## geom_token + view_token
        embeds = []
        for f, token in zip(self.token_embeds, inputs[1:]):
           embeds.append(f(token))
        embeds = torch.stack(embeds, 0).sum(0)
        ## geom_token + view_token + node feat
        embeds = embeds + self.node_embed(inputs[0])
        ## node feat
        # embeds = self.node_embed(inputs[0])
        outputs = self.encoder(inputs_embeds=embeds, attention_mask=masks)
        ## last_hidden_state, pooler_output, attentions = outputs
        out = self.classifier(outputs[1])
        return out

def toy_trainval(batches_list, data_idx, train=True, use_mask=True):
    if train:
        model.train()
    else:
        model.eval()
    losses = []
    preds = []
    idx_shuffle = list(range(0, len(data_idx), batch_size))
    id_list = []
    random.shuffle(idx_shuffle)
    for bi, i in enumerate(idx_shuffle):
        idx = data_idx[i:i+batch_size]
        id_list.append(idx)
        batch = [batches[idx].to(device) for batches in batches_list]
        mask = masks[idx].to(device)
        label = labels[idx].to(device)
        if train:
            if use_mask:
              out = model(batch, mask)
            else:
              out = model(batch)
            optimizer.zero_grad()
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                if use_mask:
                  out = model(batch, mask)
                else:
                  out = model(batch)
            loss = loss_fn(out, label)
        pred = out.max(1)[1].detach().cpu()
        preds.append(pred)
        losses.append(loss.detach().cpu())
    preds = torch.cat(preds)
    losses = torch.stack(losses)
    id_list = torch.cat(id_list)
    acc = preds.eq(labels[id_list]).sum().item() / len(data_idx)
    return losses.mean().item(), acc


if __name__=='__main__':
    from data_handling import get_data_pyg
    torch.manual_seed(142857)
    device = 'cuda:1'
    seq_len = 512
    batch_size = 16
    epoch = 100
    lr = 1e-6
    token_padder = token_zeropad
    # token_padder = token_neighborpad
    use_mask = True
    for i in range(10):
        data = get_data_pyg('wisconsin', split=i)
        geom_tokens, view_dirs, node_embeds, token_count, distance_sorts = geom_tokenizer(data.x, data.edge_index, 5, dim=3)
        # geom_tokens_d4, view_dirs_d4, _, token_d4_count, _ = geom_tokenizer(data.x, data.edge_index, 10, dim=4)
        geom_batches, masks, labels = token_padder(geom_tokens, token_count, seq_len, data.y, distance_sorts)
        # geom_batches_d4, masks_d4, _ = token_padder(geom_tokens_d4, token_d4_count, seq_len, data.y, distance_sorts)
        view_batches, _, _ = token_padder(view_dirs, token_count, seq_len, data.y, distance_sorts)
        # view_batches_d4, _, _ = token_padder(view_dirs_d4, token_d4_count, seq_len, data.y, distance_sorts)
        x_batches, _, _ = token_padder(node_embeds, token_count, seq_len, data.y, distance_sorts)
        train_idx = torch.where(data.train_mask)[0]
        val_idx = torch.where(data.val_mask)[0]
        test_idx = torch.where(data.test_mask)[0]
        loss_fn = torch.nn.CrossEntropyLoss()
        model = ToyModel(len(data.x), data.x.shape[1], 3, data.y.max().item()+1).to(device)
        optimizer = optim.Adam(model.parameters(),lr=lr) # 1e-4,weight_decay=0.01
        # optimizer = optim.SGD(model.parameters(),lr=lr) # 1e-3
        inputs = [x_batches, geom_batches, view_batches]
        # inputs = [data.x.unsqueeze(1)]
        print([i.shape for i in inputs])
        for e in range(epoch):
            train_loss, train_acc = toy_trainval(inputs, train_idx, train=True, use_mask=use_mask)
            val_loss, val_acc = toy_trainval(inputs, val_idx, train=False, use_mask=use_mask)
            test_loss, test_acc = toy_trainval(inputs, test_idx, train=False, use_mask=use_mask)
            log = f'Epoch [{e+1}\t/{epoch}] Train Loss: {train_loss:.03f} \t Train Acc: {train_acc:.06f} \t Val Loss: {val_loss:.03f} \t Val Acc: {val_acc:.06f} \t Test Loss: {test_loss:.03f} \t Test Acc: {test_acc:.06f}'
            print(datetime.now(), log)