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

    def load_validation_data(self, val_dir):
        """
        Load validation set from a separate MINDsmall_dev directory.
        Only maps users/items that already exist in the training vocabulary.
        """
        self.val_interactions = defaultdict(list)
        behavior_file = os.path.join(val_dir, 'behaviors.tsv')
        print(f"Loading validation data from {behavior_file}")
        
        df = pd.read_csv(behavior_file, sep='\t', header=None, 
                         names=['imp_id', 'user_id', 'time', 'history', 'impressions'])
        
        val_user_clicks = defaultdict(set)
        
        for row in df.itertuples():
            u = row.user_id
            if u not in self.user2id:
                continue  # Skip unknown users
            uid = self.user2id[u]
            
            # Impressions: collect clicked items as validation positives
            if pd.notna(row.impressions):
                imps = row.impressions.split()
                for imp in imps:
                    item, click = imp.split('-')
                    if click == '1' and item in self.item2id:
                        iid = self.item2id[item]
                        # Exclude items already in train to get true held-out positives
                        if iid not in self.train_interactions.get(uid, []):
                            val_user_clicks[uid].add(iid)
        
        for uid, items in val_user_clicks.items():
            self.val_interactions[uid] = list(items)
        
        n_val_users = len(self.val_interactions)
        n_val_pairs = sum(len(v) for v in self.val_interactions.values())
        print(f"Validation set loaded: {n_val_users} users, {n_val_pairs} interactions.")
    
    def get_val_dataset(self):
        """Returns (val_interactions, train_interactions) for evaluation."""
        if not hasattr(self, 'val_interactions'):
            raise RuntimeError("Validation data not loaded. Call load_validation_data() first.")
        return self.val_interactions, self.train_interactions


class ML100KDataset:
    """MovieLens 100K dataset loader using ua.base/ua.test split."""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_interactions = defaultdict(list)
        self.test_interactions = defaultdict(list)
        self.exposures = defaultdict(list)  # Empty, ml-100k has no exposure data
        
        self.n_users = 0
        self.n_items = 0
        
        self.load_data()
    
    def load_data(self):
        train_file = os.path.join(self.data_dir, 'ua.base')
        test_file = os.path.join(self.data_dir, 'ua.test')
        print(f"Loading ML-100K from {self.data_dir}")
        
        # Load train split
        train_df = pd.read_csv(train_file, sep='\t', header=None,
                               names=['user_id', 'item_id', 'rating', 'timestamp'])
        # Load test split
        test_df = pd.read_csv(test_file, sep='\t', header=None,
                              names=['user_id', 'item_id', 'rating', 'timestamp'])
        
        # Collect all unique user/item IDs (1-indexed in ml-100k, remap to 0-indexed)
        all_users = sorted(set(train_df['user_id'].tolist() + test_df['user_id'].tolist()))
        all_items = sorted(set(train_df['item_id'].tolist() + test_df['item_id'].tolist()))
        
        user2id = {u: i for i, u in enumerate(all_users)}
        item2id = {it: i for i, it in enumerate(all_items)}
        
        self.n_users = len(user2id)
        self.n_items = len(item2id)
        
        # Build train interactions (implicit: all rated items are positive)
        for row in train_df.itertuples():
            uid = user2id[row.user_id]
            iid = item2id[row.item_id]
            self.train_interactions[uid].append(iid)
        
        # Deduplicate
        for uid in self.train_interactions:
            self.train_interactions[uid] = list(dict.fromkeys(self.train_interactions[uid]))
        
        # Build test interactions
        for row in test_df.itertuples():
            uid = user2id[row.user_id]
            iid = item2id[row.item_id]
            self.test_interactions[uid].append(iid)
        
        for uid in self.test_interactions:
            self.test_interactions[uid] = list(dict.fromkeys(self.test_interactions[uid]))
        
        n_train = sum(len(v) for v in self.train_interactions.values())
        n_test = sum(len(v) for v in self.test_interactions.values())
        print(f"ML-100K loaded: {self.n_users} users, {self.n_items} items, "
              f"{n_train} train / {n_test} test interactions.")
    
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
