import pytorch_lightning as L
import torch

from torch.utils.data import random_split, DataLoader
from dataset import DummyDataset

class DummyDataModule(L.LightningDataModule):

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
        if stage == "predict":
            self.dummy_predict = DummyDataset(self.data_dir, self.img_height, self.img_width)
        else:
            dummy_full = DummyDataset(self.data_dir, self.img_height, self.img_width)
            self.dummy_train, self.dummy_val, self.dummy_test = random_split(dummy_full, [self.n_train, self.n_val, self.n_test], 
                                                                        generator=torch.Generator().manual_seed(17))

    def train_dataloader(self):
        return DataLoader(self.dummy_train, batch_size=self.batch_size, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.dummy_val, batch_size=self.batch_size, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.dummy_test, batch_size=self.batch_size, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory)
    
    def predict_dataloader(self):
        return DataLoader(self.dummy_predict, batch_size=self.batch_size, 
                          num_workers=self.num_workers, pin_memory=self.pin_memory)