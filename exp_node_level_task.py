from tqdm import tqdm
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchmetrics

from datasets import DataBatchSetNodeLevel
from data_handling import get_data_pyg
# from geom_tokenizer import ToyModel, ToyModelPE
from models import BERT

def main():
    torch.manual_seed(142857)
    device = 'cuda:1'
    seq_len = 190
    batch_size = 32
    epoch = 100
    lr = 1e-5
    num_worker = 8
    topN = 10
    use_mask = True
    geom_dim = 3
    lap_pe_dim = 15
    for i in range(10):
        data = get_data_pyg('pubmed', split=i)
        train_set = DataBatchSetNodeLevel(data.x, data.edge_index, data.y, mask=data.train_mask, seq_len=seq_len, N=topN, lap_pe_dim=lap_pe_dim)
        trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_worker)
        val_set = DataBatchSetNodeLevel(data.x, data.edge_index, data.y, mask=data.val_mask, seq_len=seq_len, N=topN, lap_pe_dim=lap_pe_dim)
        valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_worker)
        test_set = DataBatchSetNodeLevel(data.x, data.edge_index, data.y, mask=data.test_mask, seq_len=seq_len, N=topN, lap_pe_dim=lap_pe_dim)
        testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_worker)
        loss_fn = torch.nn.CrossEntropyLoss()
        model = BERT(0, data.x.shape[1], geom_dim, data.y.max().item()+1, pe_dim=train_set.pe_dim).to(device)
        optimizer = optim.Adam(model.parameters(),lr=lr) # 1e-4,weight_decay=0.01
        print(f'[start train val on split {i}]')
        for e in range(epoch):
            train_loss, train_acc = trainval(trainloader, model, optimizer, loss_fn, e, epoch, device, train=True, use_mask=use_mask)
            val_loss, val_acc = trainval(valloader, model, optimizer, loss_fn, e, epoch, device, train=False, use_mask=use_mask)
            test_loss, test_acc = trainval(testloader, model, optimizer, loss_fn, e, epoch, device, train=False, use_mask=use_mask)
            log = f'Epoch [{e+1}\t/{epoch}] Train Loss: {train_loss:.03f} \t Train Acc: {train_acc:.06f} \t Val Loss: {val_loss:.03f} \t Val Acc: {val_acc:.06f} \t Test Loss: {test_loss:.03f} \t Test Acc: {test_acc:.06f}'
            print(datetime.now(), log)


def trainval(loader, model, optimizer, loss_fn, e, epoch, device, train=True, use_mask=True):
    if train:
        model.train()
        loop_type = 'train'
    else:
        model.eval()
        loop_type = 'val/test'
    metricer = torchmetrics.AUROC(task='binary')
    # metricer = acc_metric
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
        if metricer == acc_metric:
            pred = out.max(1)[1].detach().cpu()
        else:
            pred = out.max(1)[0].detach().cpu()
        preds.append(pred)
        labels.append(label.detach().cpu())
        losses.append(loss.detach().cpu())
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    losses = torch.stack(losses)
    return losses.mean().item(), metricer(preds, labels)

def acc_metric(preds, labels):
   return preds.eq(labels).sum().item() / len(labels)


if __name__ == '__main__':
    main() 