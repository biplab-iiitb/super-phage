""" Python module To perform DIS phage training (U2Net)
"""

__author__ = "Biplab Ch Das"
__authors__ = ["Biplab Ch Das"]
__contact__ = "biplab.das@samsung.com"
__copyright__ = "Copyright 2022, Samsung"
__credits__ = []
__date__ = "2022/08/30"
__deprecated__ = False
__email__ = "biplab.das@samsung.com"
__license__ = ""
__maintainer__ = "developer"
__status__ = "Prototype"
__version__ = "0.1.0"


import torch
import torch.nn as nn
import pdb, random
import pickle
import numpy as np
import torch.nn.functional as F
from rbp_net import RBPNet
import pandas as pd
from scipy.optimize import linear_sum_assignment

class PhageDesigner:
    def __init__(self):
        self.model = RBPNet().cuda()

    @torch.no_grad()
    def predict(self):
        
        self.model.load_state_dict(
            torch.load("checkpoints/rbp_net/rbp_net_v1.pth"),
           strict=False,  # "models/init/token_nca_v18.pth"
        )
        
        embedding_type = "prott5"
        rbp_embeddings = pd.read_csv(f'data/features_csv/rbp_embeddings_{embedding_type}.csv', 
                                     low_memory = False)
        rbp_embeddings['Modification Date'] = pd.to_datetime(rbp_embeddings['Modification Date'])
        # Get only the top 25% hosts
        self.hosts = rbp_embeddings['Host'].tolist()
        self.host_to_idx = { h: i for i, h in enumerate(list(set(self.hosts))) }
        
        embeddings_size = 1024
        feature_columns = [str(i) for i in range(1, embeddings_size + 1)]
        features = rbp_embeddings.loc[:, rbp_embeddings.columns.isin(feature_columns)].to_numpy()
        
        self.model.eval()
        output = self.model(torch.tensor(features).cuda().float())
        output = torch.sigmoid(output)  
        self.protein_ids = rbp_embeddings['Protein ID'].tolist() 
        self.phages = rbp_embeddings['Description'].tolist()
        self.host_list = list(set(self.hosts))
        return output.cpu().numpy()     
        


import ot

class BacteriophageCocktailOptimizer:
    def __init__(self,):
        designer = PhageDesigner()
        phage_host_interaction = designer.predict()
        #self.efficacy_scores = efficacy_scores
        #self.safety_scores = safety_scores
        #self.phage_phage_interactions = phage_phage_interactions
        self.cost_matrix = 1.0 - phage_host_interaction.T
        self.phages = designer.phages
        self.bacteria = designer.host_list
        self.protein_ids = designer.protein_ids

    def optimize_cocktail(self):
        M = self.cost_matrix
        a = np.ones(M.shape[0]) / M.shape[0]
        b = np.ones(M.shape[1]) / M.shape[1]
        result  = ot.sinkhorn(a, b, M, reg = 0.1)
        return result
        
        
    def optimize_cocktail_hungarian(self):
        M = self.cost_matrix
        row_ind, col_ind = linear_sum_assignment(self.cost_matrix)
        result = []
        for i in range(len(row_ind)):
            result += [(self.bacteria[row_ind[i]], self.protein_ids[col_ind[i]])]
        return result
        

    def print_optimized_cocktail(self):
        result = self.optimize_cocktail()
        cocktail = set()
        targets = set()
        for i, target_bacteria in enumerate(self.bacteria):
            protein_id = self.protein_ids[result[i].argmax()]
            cocktail.add(protein_id)
                                    
        print("Optimal Transport Cocktail length:", len(cocktail))
        #print("Targets length:", len(targets))
        #print("Bacteriophage cocktail:", cocktail)



           
def do_prediction():
    # Create an instance of the BacteriophageCocktailOptimizer class
    optimizer = BacteriophageCocktailOptimizer()

    # Print the optimized cocktail
    optimizer.print_optimized_cocktail()
    #np.save('data/phage_host_interaction.npy', phage_host_interaction)
    result = optimizer.optimize_cocktail_hungarian()
    print("Hungarian Cocktail length:", len(result))
    #print(result)
            
    


if __name__ == "__main__":
    #do_training()
    do_prediction()
