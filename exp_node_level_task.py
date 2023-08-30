from tqdm import tqdm
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import DataBatchSet
from data_handling import get_data_pyg
from geom_tokenizer import ToyModel

def main():
    torch.manual_seed(142857)
    device = 'cuda:0'
    seq_len = 512
    batch_size = 16
    epoch = 100
    lr = 1e-6
    use_mask = True
    for i in range(10):
        data = get_data_pyg('cora', split=i)
        train_set = DataBatchSet(data.x, data.edge_index, data.y, mask=data.train_mask, seq_len=seq_len)
        trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_set = DataBatchSet(data.x, data.edge_index, data.y, mask=data.val_mask, seq_len=seq_len)
        valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_set = DataBatchSet(data.x, data.edge_index, data.y, mask=data.test_mask, seq_len=seq_len)
        testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        loss_fn = torch.nn.CrossEntropyLoss()
        model = ToyModel(0, data.x.shape[1], 3, data.y.max().item()+1).to(device)
        optimizer = optim.Adam(model.parameters(),lr=lr) # 1e-4,weight_decay=0.01
        print(f'[start train val on split {i}]')
        for e in range(epoch):
            train_loss, train_acc = trainval(trainloader, model, optimizer, loss_fn, e, epoch, device, train=True, use_mask=True)
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
        pred = out.max(1)[1].detach().cpu()
        preds.append(pred)
        labels.append(label.detach().cpu())
        losses.append(loss.detach().cpu())
    preds = torch.cat(preds)
    preds = torch.cat(labels)
    losses = torch.stack(losses)
    acc = preds.eq(labels).sum().item() / len(labels)
    return losses.mean().item(), acc

if __name__ == '__main__':
    main() 