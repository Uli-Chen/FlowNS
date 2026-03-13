import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random

class MINDDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.user2id = {}
        self.item2id = {}
        self.id2user = []
        self.id2item = []
        
        self.train_interactions = defaultdict(list)
        self.test_interactions = defaultdict(list)
        self.exposures = defaultdict(list)
        
        self.n_users = 0
        self.n_items = 0
        self.all_items_set = set()
        
        self.load_data()
        
    def _add_user(self, user):
        if user not in self.user2id:
            self.user2id[user] = self.n_users
            self.id2user.append(user)
            self.n_users += 1
        return self.user2id[user]
        
    def _add_item(self, item):
        if item not in self.item2id:
            self.item2id[item] = self.n_items
            self.id2item.append(item)
            self.all_items_set.add(self.n_items)
            self.n_items += 1
        return self.item2id[item]

    def load_data(self):
        behavior_file = os.path.join(self.data_dir, 'behaviors.tsv')
        print(f"Loading data from {behavior_file}")
        
        # columns: Impression ID, User ID, Time, History, Impressions
        df = pd.read_csv(behavior_file, sep='\t', header=None, 
                         names=['imp_id', 'user_id', 'time', 'history', 'impressions'], nrows=5000)
        
        # We will split data for evaluation. Since this is a test pipeline, 
        # let's use the last clicked item in impressions as the test item for each user 
        # (or random 80/20 if preferable). 
        # Let's collect all positive interactions for each user.
        
        user_clicks = defaultdict(list)
        user_exposures = defaultdict(list)
        
        for row in df.itertuples():
            u = row.user_id
            uid = self._add_user(u)
            
            # History
            if pd.notna(row.history):
                hist_items = row.history.split()
                for item in hist_items:
                    iid = self._add_item(item)
                    user_clicks[uid].append(iid)
            
            # Impressions
            if pd.notna(row.impressions):
                imps = row.impressions.split()
                for imp in imps:
                    item, click = imp.split('-')
                    iid = self._add_item(item)
                    if click == '1':
                        user_clicks[uid].append(iid)
                    else:
                        user_exposures[uid].append(iid)
        
        # Remove duplicates and setup train/test split
        # For simplicity in recommendation task: leave-one-out testing
        for uid, clicks in user_clicks.items():
            # Keep unique items and preserve order roughly
            unique_clicks = list(dict.fromkeys(clicks))
            
            if len(unique_clicks) > 1:
                self.train_interactions[uid] = unique_clicks[:-1]
                self.test_interactions[uid] = [unique_clicks[-1]]
            else:
                self.train_interactions[uid] = unique_clicks
                self.test_interactions[uid] = []
                
        for uid, exposures in user_exposures.items():
            self.exposures[uid] = list(set(exposures))
            
        print(f"Dataset loaded: {self.n_users} users, {self.n_items} items.")
        
    def get_train_dataset(self, num_negatives=1):
        return MFDataset(self.train_interactions, self.n_items, num_negatives)

    def get_test_dataset(self):
        return self.test_interactions, self.train_interactions

class MFDataset(Dataset):
    def __init__(self, user_interactions, n_items, num_negatives=1):
        self.user_interactions = user_interactions
        self.n_items = n_items
        self.num_negatives = num_negatives
        
        self.users = []
        self.pos_items = []
        
        for u, items in user_interactions.items():
            for i in items:
                self.users.append(u)
                self.pos_items.append(i)
                
    def __len__(self):
        return len(self.users)
        
    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]
        
        # Sample negative
        neg_items = []
        for _ in range(self.num_negatives):
            while True:
                neg_item = random.randint(0, self.n_items - 1)
                if neg_item not in self.user_interactions[user]:
                    neg_items.append(neg_item)
                    break
                    
        return {
            'user': torch.tensor(user, dtype=torch.long),
            'pos_item': torch.tensor(pos_item, dtype=torch.long),
            'neg_item': torch.tensor(neg_items[0] if self.num_negatives > 0 else -1, dtype=torch.long)
        }

class ExposureDataset(Dataset):
    """
    Dataset for Flow Matching generator training.
    Provides (user, exposed_item) pairs to act as target points for the flow.
    """
    def __init__(self, exposures):
        self.exposures = exposures
        self.users = []
        self.exp_items = []
        
        for u, items in exposures.items():
            for i in items:
                self.users.append(u)
                self.exp_items.append(i)
                
    def __len__(self):
        return len(self.users)
        
    def __getitem__(self, idx):
        return {
            'user': torch.tensor(self.users[idx], dtype=torch.long),
            'item': torch.tensor(self.exp_items[idx], dtype=torch.long)
        }

if __name__ == '__main__':
    # Simple test
    dataset = MINDDataset('data/MINDsmall_train')
    train_dl = DataLoader(dataset.get_train_dataset(), batch_size=256, shuffle=True)
    for batch in train_dl:
        print("MF Train Batch:", batch['user'].shape, batch['pos_item'].shape, batch['neg_item'].shape)
        break
    
    exp_ds = ExposureDataset(dataset.exposures)
    exp_dl = DataLoader(exp_ds, batch_size=256, shuffle=True)
    for batch in exp_dl:
        print("Exposure Train Batch:", batch['user'].shape, batch['item'].shape)
        break
