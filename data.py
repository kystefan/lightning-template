import pytorch_lightning as L
import torch

from torch.utils.data import random_split, DataLoader
from dataset import SHHQDataset

class SHHQDataModule(L.LightningDataModule):

    def __init__(self, data_dir,
                 num_workers, pin_memory,
                 batch_size, img_height, img_width,
                 n_train, n_val, n_test):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test

    def setup(self, stage=None):
        shhq_full = SHHQDataset(self.data_dir, self.img_height, self.img_width)
        self.shhq_train, self.shhq_val, self.sshq_test = random_split(shhq_full, [self.n_train, self.n_val, self.n_test], 
                                                                      generator=torch.Generator().manual_seed(17))
        
        self.shhq_predict = self.sshq_test

    def train_dataloader(self):
        return DataLoader(self.shhq_train, batch_size=self.batch_size, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.shhq_val, batch_size=self.batch_size, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.sshq_test, batch_size=self.batch_size, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory)
    
    def predict_dataloader(self):
        return DataLoader(self.shhq_predict, batch_size=self.batch_size, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory)