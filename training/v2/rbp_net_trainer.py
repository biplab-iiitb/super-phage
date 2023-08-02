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
from torch.utils.data import Dataset, DataLoader
from phage_dataset import PhageDataset
import pdb, random
import pickle
import numpy as np
import torch.nn.functional as F
from rbp_net import RBPNet
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
class PhageTrainer:
    def __init__(self):
        self.phage_dataset = PhageDataset()  # DISDataset()
        print("Total dataset size:", len(self.phage_dataset))
        self.phage_dataloader = DataLoader(
            self.phage_dataset, batch_size=128, shuffle=True, num_workers=1
        )

        NUM_CLASSES = 1
        self.model = RBPNet().cuda()
        #self.model.load_state_dict(
        #    torch.load("checkpoints/nca/recursive_nca_v1.pth"),
        #   strict=False,  # "models/init/token_nca_v18.pth"
        #)

        self._setup_optimizers()

        self.phage_iter = iter(self.phage_dataloader)

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="sum")

    def _clip_weights(self):
        """
        Performs clipping of weights.
        """
        for p in self.model.parameters():
            p.data.clamp_(-1.0 * self.clip_value, self.clip_value)

    def _setup_optimizers(self):
        self.iter_size = 1
        self.optimizer = torch.optim.Adam(
            [param for name, param in self.model.named_parameters()],
            lr=3e-4,
            weight_decay=0.00001)
            #momentum = 0.95)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[30, 80], gamma=0.1
        )

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path + ".pth")

    @torch.no_grad()
    def predict(self):
        
        self.model.load_state_dict(
            torch.load("checkpoints/rbp_net/rbp_net_v2.pth"),
           strict=False,  # "models/init/token_nca_v18.pth"
        )
        
        features = self.phage_dataset.X_test
        self.model.eval()
        output = self.model(torch.tensor(features).cuda().float())
        output = torch.sigmoid(output) 
        
        y_pred = output.cpu().numpy().argmax(-1)
        y_target = self.phage_dataset.y_target
        print(y_target[:10], y_pred[:10])
        acc = accuracy_score(y_target, y_pred)
        print("acc:", acc)
        f1 = f1_score(y_target, y_pred, average='weighted')
        print("f1:", f1)
        return output.cpu().numpy()     
       
    def step(self):
        self.optimizer.zero_grad()
        seg_loss = 0.0
        for _ in range(self.iter_size):
            # auxialiary classifier
            try:
                phage_sample = next(self.phage_iter)
            except StopIteration:
                print("bbox dataloader reset.")
                self.phage_iter = iter(self.phage_dataloader)
                phage_sample = next(self.phage_iter)

            #edge_band = phage_sample["edge_band"].cuda()
            labels = phage_sample["host_vector"].float().cuda()#.unsqueeze(1)
            output = self.model(phage_sample["rbp_embedding"].float().cuda())
            #print(output.shape, labels.shape)
            loss = self.bce_loss(output, labels) / output.shape[0]
            loss.backward()
         
        seg_loss = loss.detach().item()
        self.optimizer.step()
        #self._clip_weights()

        return [
            seg_loss / self.iter_size,
        ]


def do_training():
    trainer = PhageTrainer()
    max_iters = 20000
    save_iter = 100
    snap_iter = 1000
    for iter_no in range(max_iters):
        #batch_loss = trainer.step()
        try:
            #pass
            batch_loss = trainer.step()
        except KeyboardInterrupt:
            print("User Exit.")
            exit(1)
        except:
            batch_loss = 0.0, 0.0, 0.0, 0.0
        print(
            "[Iter %d/%d] seg_loss = %f"
            % (iter_no, max_iters, batch_loss[0])
        )
        if (iter_no + 1) % snap_iter == 0:
            trainer.save("checkpoints/rbp_net/rbp_net_v2_%d" % (iter_no + 1))
        elif (iter_no + 1) % save_iter == 0:
            trainer.save("checkpoints/rbp_net/rbp_net_v2")
            
            
def do_prediction():
    trainer = PhageTrainer()
    phage_host_interaction = trainer.predict()
    print(phage_host_interaction.shape)
            
    


if __name__ == "__main__":
    do_training()
    do_prediction()
