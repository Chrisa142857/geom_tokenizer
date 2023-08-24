from torch_geometric.data import Data
import torch
import torch.optim as optim
from tqdm import trange, tqdm
import transformers
import random
from datetime import datetime

def geom_token_level3(data: Data, N: int, dim: int=3):
    node_list = data.x # X x C
    nids = torch.arange(len(data.x))
    edge_list = data.edge_index # 2 x E
    tokens = [] # pos, geom, view1, view2, ...
    node_mask = []
    for ni in tqdm(nids, desc='Prepare tokens...'):
        distances1 = ((node_list[ni] - node_list[ni+1:]) **2 ).sum(1)
        if len(distances1) > 0:
            mind = distances1.min()
            maxd = distances1.max()
        distances2 = ((node_list[ni] - node_list[:ni]) **2 ).sum(1)
        if len(distances2) > 0:
            mind = min(mind, distances2.min())
            maxd = max(maxd, distances2.max())
        
        distances = torch.cat([distances1, torch.FloatTensor([maxd+1]), distances2]) # X
        ## spatially close nodes are neighborhood
        nei_nid = distances.argsort()[:N]
        ## connected nodes are neighborhood
        connected_node = edge_list[1, edge_list[0] == ni]
        nei_conn = (connected_node==nei_nid[..., None])
        connected_node = connected_node[~(nei_conn.any(0))]
        connected_node = connected_node[distances[connected_node].argsort()] # also sort connected nodes
        nei_nid = torch.cat([nei_nid, connected_node]) # concat in the sorted rank
        ## geom level 3, triangle, it has dim-1 = 2 view tokens
        ## view token, max = N, the neighbor num
        view_id = torch.stack(torch.meshgrid(torch.arange(len(nei_nid)), torch.arange(len(nei_nid))), -1) # N x N x 2
        indices = torch.triu_indices(len(nei_nid), len(nei_nid), offset=1) # M = N x (N-1) / 2
        view_id = view_id[indices[0], indices[1]].T # 2 x M
        ## view nodes are sorted by distance
        view_token = [nei_nid[view_id[di]] for di in range(dim-1)] # 2 x M, each is a node id
        ## pos token, max = node num
        pos_token = torch.LongTensor([ni for _ in range(len(view_token[0]))]) # M
        ## geom token, max = 2**3 = 8
        geom_token = torch.stack([torch.zeros_like(view_token[0]) for _ in range(dim)]) # 3 x M
        where_edges = torch.cat([nei_conn.any(1), torch.ones_like(connected_node, dtype=bool)])
        # view_id[0] < view_id[1]
        for di in range(dim-1):
            where_edge = where_edges[view_id[di]]
            geom_token[di, where_edge] = 1
        ## if view points are connected
        where_edge = []
        for view in torch.stack(view_token, -1): # for 2 in M x 2
            where_edge.append((view == edge_list.T).any())
        where_edge = torch.where(torch.stack(where_edge))[0]
        geom_token[2, where_edge] = 1   
        ## convert token to 10-base number
        geom_token = bin2dec(geom_token) # M
        tokens.append(torch.stack([pos_token, geom_token] + view_token, 0))
        node_mask.append(len(view_token[0]))
        
    tokens = torch.cat(tokens, 1)
    node_mask = torch.FloatTensor(node_mask)
    pos_tokens, geom_tokens = tokens[:2]
    view_tokens = tokens[2:]
    return pos_tokens, geom_tokens, view_tokens, torch.cumsum(node_mask, 0)

def bin2dec(b):
    bits, batch = b.shape
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    mask = torch.stack([mask for _ in range(batch)], 1)
    return torch.sum(mask * b, 0)

def batcher_token(tokens, node_cumsum, seq_len, datay):
    '''
    tokens: M
    M = token number
    # n = token type number
    '''
    batches = []
    labels = []
    masks = []
    node_cumsum = node_cumsum.long()
    for i in range(len(node_cumsum)):
        prev = 0 if i == 0 else node_cumsum[i-1]
        seq = tokens[prev:node_cumsum[i].item()]
        seq = seq[:seq_len]
        mask = torch.ones(seq_len, dtype=bool, device=seq.device)
        if len(seq) < seq_len:
            one = torch.cat([seq, torch.zeros(seq_len-len(seq), device=seq.device, dtype=seq.dtype)])
            mask[len(seq):] = False
        else:
            one = seq
        labels.append(datay[i])
        batches.append(one)
        masks.append(mask)
    batches = torch.stack(batches)
    masks = torch.stack(masks)
    labels = torch.LongTensor(labels)
    return batches, masks, labels

if __name__ == "__main__":
    from data_handling import get_data
    seq_len = 512
    batch_size = 8
    epoch = 100
    lr = 1e-5
    data = get_data('wisconsin', split=0)
    pos_tokens, geom_tokens, view_tokens, node_cumsum = geom_token_level3(data, 15)
    geom_batches, masks, labels = batcher_token(geom_tokens, node_cumsum, seq_len, data.y)
    pos_batches, _, _ = batcher_token(pos_tokens, node_cumsum, seq_len, data.y)
    view1_batches, _, _ = batcher_token(view_tokens[0], node_cumsum, seq_len, data.y)
    view2_batches, _, _ = batcher_token(view_tokens[1], node_cumsum, seq_len, data.y)
    train_idx = torch.where(data.train_mask)[0]
    val_idx = torch.where(data.val_mask)[0]
    test_idx = torch.where(data.test_mask)[0]
    loss_fn = torch.nn.CrossEntropyLoss()
    # node_feature = torch.cat([data.x[pos_tokens]] + [data.x[view] for view in view_tokens], -1)
    # print(node_feature.shape)

    class ToyModel(torch.nn.Module):
        def __init__(self, node_num, geom_dim, cls_num, nhead=8) -> None:
            '''
            Toy transformer for node classification
            '''
            super().__init__()
            ## Embed all tokens
            self.encoder = transformers.BertModel.from_pretrained('bert-base-uncased')
            # self.encoder.config.output_attentions = True
            hdim = self.encoder.config.hidden_size
            token_embed1 = torch.nn.Embedding(node_num, hdim//4)
            token_embed2 = torch.nn.Embedding(2**geom_dim, hdim//4)
            token_embed3 = torch.nn.Embedding(node_num, hdim//4)
            token_embed4 = torch.nn.Embedding(node_num, hdim//4)
            self.token_embeds = torch.nn.ModuleList([token_embed1, token_embed2, token_embed3, token_embed4])
            ## Transformer Encoder 
            # encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hdim, nhead=nhead)
            # self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)
            self.classifier = torch.nn.Linear(hdim, cls_num)
        
        def forward(self, tokens, masks):
            # pos_tokens, geom_tokens, view_tokens = tokens
            embeds = []
            for f, token in zip(self.token_embeds, tokens):
                embeds.append(f(token))
            embeds = torch.cat(embeds, -1)
            outputs = self.encoder(inputs_embeds=embeds, attention_mask=masks)
            ## last_hidden_state, pooler_output, attentions = outputs
            out = self.classifier(outputs[1])
            return out
    
    def toy_trainval(batches_list, data_idx, train=True):
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
            batch = [batches[idx].cuda() for batches in batches_list]
            mask = masks[idx].cuda()
            label = labels[idx].cuda()
            if train:
                out = model(batch, mask)
                optimizer.zero_grad()
                loss = loss_fn(out, label)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    out = model(batch, mask)
                loss = loss_fn(out, label)
            pred = out.max(1)[1].detach().cpu()
            preds.append(pred)
            losses.append(loss.detach().cpu())
        preds = torch.cat(preds)
        losses = torch.stack(losses)
        id_list = torch.cat(id_list)
        acc = preds.eq(labels[id_list]).sum().item() / len(data_idx)
        return losses.mean().item(), acc

    model = ToyModel(len(data.x), 3, data.y.max().item()+1).cuda()
    optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=0.01)
    inputs = [pos_batches, geom_batches, view1_batches, view2_batches]
    for e in range(epoch):
        train_loss, train_acc = toy_trainval(inputs, train_idx, train=True)
        val_loss, val_acc = toy_trainval(inputs, val_idx, train=False)
        test_loss, test_acc = toy_trainval(inputs, test_idx, train=False)
        log = f'Epoch [{e+1}/{epoch}] Train Loss: {train_loss:.03f} \t Train Acc: {train_acc:.06f} \t Val Loss: {val_loss:.03f} \t Val Acc: {val_acc:.06f} \t Test Loss: {test_loss:.03f} \t Test Acc: {test_acc:.06f}'
        print(datetime.now(), log)
    