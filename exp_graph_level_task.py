from tqdm import tqdm
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import DataBatchSetGraphLevel
from data_handling import get_data_pyg
from geom_tokenizer import ToyModel

def main():
    torch.manual_seed(142857)
    device = 'cuda:3'
    seq_len = 1024
    batch_size = 8
    epoch = 100
    lr = 1e-6
    num_worker = 8
    use_mask = True
    topn = 5
    regress = True
    data = get_data_pyg('zinc')
    train_set, val_set, test_set = data
    # train_set, val_set, test_set = train_set[:100], val_set[:10], test_set[:10]
    train_set = DataBatchSetGraphLevel([d.x for d in train_set], [d.edge_index for d in train_set], [d.y for d in train_set], seq_len=seq_len, node_feat2bin=True, N=topn)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    val_set = DataBatchSetGraphLevel([d.x for d in val_set], [d.edge_index for d in val_set], [d.y for d in val_set], seq_len=seq_len, node_feat2bin=True, N=topn, node_feat_ch=train_set.node_feat_ch)
    valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    test_set = DataBatchSetGraphLevel([d.x for d in test_set], [d.edge_index for d in test_set], [d.y for d in test_set], seq_len=seq_len, node_feat2bin=True, N=topn, node_feat_ch=train_set.node_feat_ch)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_worker)
    if not regress:
        loss_fn = torch.nn.CrossEntropyLoss()
        model = ToyModel(0, train_set.node_feat_ch, 3, data.y.max().item()+1, pe_dim=train_set.pe_dim).to(device)
    else:
        loss_fn = torch.nn.L1Loss()
        model = ToyModel(0, train_set.node_feat_ch, 3, 1, pe_dim=train_set.pe_dim, max_position_embeddings=seq_len).to(device)
    optimizer = optim.Adam(model.parameters(),lr=lr) # 1e-4,weight_decay=0.01
    print(f'[start train val]')
    for e in range(epoch):
        train_loss, train_metric = trainval(trainloader, model, optimizer, loss_fn, e, epoch, device, train=True, use_mask=use_mask, regress=regress)
        val_loss, val_metric = trainval(valloader, model, optimizer, loss_fn, e, epoch, device, train=False, use_mask=use_mask, regress=regress)
        test_loss, test_metric = trainval(testloader, model, optimizer, loss_fn, e, epoch, device, train=False, use_mask=use_mask, regress=regress)
        if regress:
            log = f'Epoch [{e+1}\t/{epoch}] Train Loss: {train_loss:.03f} \t Train MAE: {train_metric:.06f} \t Val Loss: {val_loss:.03f} \t Val MAE: {val_metric:.06f} \t Test Loss: {test_loss:.03f} \t Test MAE: {test_metric:.06f}'
        else:
            log = f'Epoch [{e+1}\t/{epoch}] Train Loss: {train_loss:.03f} \t Train Acc: {train_metric:.06f} \t Val Loss: {val_loss:.03f} \t Val Acc: {val_metric:.06f} \t Test Loss: {test_loss:.03f} \t Test Acc: {test_metric:.06f}'
        print(datetime.now(), log)


def trainval(loader, model, optimizer, loss_fn, e, epoch, device, train=True, use_mask=True, regress=False):
    if train:
        model.train()
        loop_type = 'train'
    else:
        model.eval()
        loop_type = 'val/test'
    losses = []
    preds = []
    labels = []
    for data in tqdm(loader, desc=f'Epoch [{e+1}\t/{epoch}] {loop_type}'):
        pad_mask_d3, label, did = data[3:]
        batch = [input.to(device) for input in data[:3]]
        mask = pad_mask_d3.to(device)
        label = label.to(device)
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
        if not regress:
            pred = out.max(1)[1].detach().cpu()
        else:
            pred = out.detach().cpu()
        preds.append(pred)
        labels.append(label.detach().cpu())
        losses.append(loss.detach().cpu())
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    losses = torch.stack(losses)
    if not regress:
        metric = preds.eq(labels).sum().item() / len(labels)
    else:
        metric = (preds - labels).abs().mean().item()
    return losses.mean().item(), metric

if __name__ == '__main__':
    main() 