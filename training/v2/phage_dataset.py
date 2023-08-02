import numpy as np
import random
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import pdb
import math
from sklearn.model_selection import train_test_split

class PhageDataset(Dataset):
    def __init__(self):
        embedding_type = "prott5"
        rbp_embeddings = pd.read_csv(f'data/features_csv/rbp_embeddings_{embedding_type}.csv', 
                                     low_memory = False)
        rbp_embeddings['Modification Date'] = pd.to_datetime(rbp_embeddings['Modification Date'])
        # Get only the top 25% hosts
        # Get only the top 25% hosts
        all_counts = rbp_embeddings['Host'].value_counts()
        TOP_X_PERCENT = 0.25
        top_x = math.floor(all_counts.shape[0] * TOP_X_PERCENT)

        top_genus = set()
        genus_counts = all_counts.index
        for entry in genus_counts[:top_x]:
            top_genus.add(entry) 
        print(top_genus)

        self.hosts = [ x if x in top_genus else "others" for x in rbp_embeddings['Host'].tolist()]
        self.host_to_idx = { h: i for i, h in enumerate(list(set(self.hosts))) }
        
        embeddings_size = 1024
        feature_columns = [str(i) for i in range(1, embeddings_size + 1)]
        self.features = rbp_embeddings.loc[:, rbp_embeddings.columns.isin(feature_columns)].to_numpy()
       
        X = self.features.copy()
        Y = list(self.hosts)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
        self.y_target = [self.host_to_idx[h] for h in self.y_test]
        #print(X_train.shape, X_test.shape)
        print("Total hosts: ", len(self.host_to_idx))
        print("Total phages: ", len(self.hosts))
        
    
    def __len__(self):
        return len(self.y_train)
        
    def __getitem__(self, idx):   
        
        host_id = self.host_to_idx[self.y_train[idx]]
        rbp_embedding = self.X_train[idx]
        host_vector = np.zeros(len(self.host_to_idx), dtype=np.float32)
        host_vector[host_id] = 1.0
        sample = {"rbp_embedding": rbp_embedding, "host_vector": host_vector}
        return sample
        
def test_dataloader():
    from torch.utils.data import DataLoader
    dataset = PhageDataset()
    train_dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)
    train_dataloader = iter(train_dataloader)
    
    sample = next(train_dataloader)
    print(sample["rbp_embedding"].shape, sample["host_vector"].shape)
        
if __name__ == "__main__":
    test_dataloader()
    
    
