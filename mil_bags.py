#%%
import scipy.io as sio
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Dataset

#%%
path = './mil_datasets/'
class MILBags(Dataset):
    def __init__(self, set, seed=0, fold=0, max_fold=10, train=True, bag_shuffle=False, snr=None, text_noise=None):
        super().__init__()
        print(set, max_fold)
        self.train = train
        self.snr = snr
        self.text_noise = text_noise
        if set == 'ucsb' or set[:3] == 'web':
            if set == 'ucsb':
                dataset = sio.loadmat(path+'ucsb_breast.mat')['data']
            else:
                dataset = sio.loadmat(path+set+'.mat')['data']
            num_bags = dataset.shape[0]
            bags_list = []
            labels_list = []

            for i in range(num_bags):
                bags_list.append(torch.from_numpy(dataset[i,0].astype(np.float32)))
                labels_list.append(1.0 if dataset[i,1][0,0] == 1 else 0.0)
        
        else:
            if set[:4] == 'musk':
                dataset = sio.loadmat(path+set+'norm_matlab.mat')
            else:
                dataset = sio.loadmat(path+set+'_100x100_matlab.mat')
            features = torch.from_numpy(dataset['features'].todense().astype(np.float32))
            labels = torch.from_numpy(dataset['labels'].todense().astype(np.int8)).squeeze()
            bag_ids = torch.from_numpy(dataset['bag_ids'].astype(np.int64)).squeeze()

            bags_list = []
            labels_list = []
            num_bags = torch.max(bag_ids).item()
            for i in range(num_bags):
                pos = torch.where(bag_ids==i+1)[0]
                if bag_shuffle:
                    idx = torch.randperm(pos.shape[0])
                    pos = pos[idx]
                bags_list.append(features[pos,:])
                labels_list.append(1.0 if labels[pos[0]] == 1 else 0.0)

        assert num_bags == len(bags_list)
        data = list(zip(bags_list, labels_list))
        random.seed(seed)
        random.shuffle(data)

        if set[:3] == 'web':
            start_pos = 0
            end_pos = 38
        else:
            start_pos = round(fold*num_bags/max_fold)
            end_pos = round((fold+1)*num_bags/max_fold)
        assert start_pos >= 0 and start_pos < num_bags and end_pos >0 and end_pos <= num_bags
        self.test_data = data[start_pos:end_pos]
        self.train_data = data[:start_pos]+data[end_pos:]
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            data = self.train_data[index]
        else:
            data = self.test_data[index]

        if self.snr is not None:
            x = self.awgn(data[0])
            data = (x, data[1])
        
        if self.text_noise is not None:
            x = self.add_text_noise(data[0])
            data = (x, data[1])

        return data

# %%
