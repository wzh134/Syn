#%%
import argparse
import torch
import numpy as np

from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from mil_bags import MILBags
from syn import Net
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

#%%
def train_epoch(network, optimiser, scheduler):
    network.train()
    for data, target in data_loader_train:
        data, target = data.to(device), target.to(device)

        loss, _ = network.calculate_objective(data, target)

        optimiser.zero_grad()
        loss.backward()
        clip_grad_norm_(parameters=network.parameters(), max_norm=1.0, norm_type=2)
        optimiser.step()

    if scheduler is not None:
        scheduler.step()

def eval_iter(network):
    network.eval()
    with torch.no_grad():
        targets = []
        preds = []
        for data, target in data_loader_eval:
            targets.append(target.item())
            data, target = data.to(device), target.to(device)

            pred, _ = network(data)
            preds.append(pred.item())

        targets = np.array(targets)
        preds = np.array(preds)
        return targets, preds

def operate(network, optimiser, scheduler, num_epochs):
    best_auc = 0.0
    for _ in range(num_epochs):
        train_epoch(network, optimiser, scheduler)

        targets, preds = eval_iter(network)
        fpr, tpr, _ = roc_curve(targets, preds, pos_label=1)
        result = auc(fpr, tpr)
        if result > best_auc:
            best_auc = result
    return best_auc

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--time", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", type=str, default='ucsb')
    args = parser.parse_args()
    device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')

    if args.set == 'web1':
        input_dim = 5864
    elif args.set == 'web2':
        input_dim = 6520
    elif args.set == 'web3':
        input_dim = 6307
    elif args.set == 'web4':
        input_dim = 6060
    elif args.set == 'web5':
        input_dim = 6408
    elif args.set == 'web6':
        input_dim = 6418
    elif args.set == 'web7':
        input_dim = 6451
    elif args.set == 'web8':
        input_dim = 6000
    elif args.set == 'web9':
        input_dim = 6280
    elif args.set == 'ucsb':
        input_dim = 709
    elif args.set[:4] == 'musk':
        input_dim = 166
    else:
        input_dim = 230

    lr = 1e-3
    embed_layer_num = 1
    hidden_dim = 256
    num_head = 4
    embed_dim = 32
    step = -20
    scale = 10

    data_loader_train = DataLoader(MILBags(train=True, set=args.set, seed=args.seed, fold=args.fold, max_fold=4), batch_size=1, shuffle=True)
    data_loader_eval = DataLoader(MILBags(train=False, set=args.set, seed=args.seed, fold=args.fold, max_fold=4), batch_size=1, shuffle=False)

    network = Net(embed_layer_num, input_dim, hidden_dim, embed_dim, step, num_head, scale).to(device=device)

    optimizer = AdamW(network.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = None
    result = operate(network, optimizer, scheduler, num_epochs=50)
    print(result)

# %%
