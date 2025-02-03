import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import Accuracy, F1Score, ConfusionMatrix
from multiprocessing import cpu_count
from torchsampler import ImbalancedDatasetSampler
import pandas as pd
import pytorch_lightning as pl

from scipy.fft import fft
from scipy import signal

import os
from glob import glob
from multiprocessing import cpu_count

class HandFaceDataset(Dataset):
    # def __init__(self, path, features, label_encoder, normalize=False):
    #     sequences = []
    #     labels = []
    #     for subpath, subdir, files in os.walk(path):
    #         for file in glob(os.path.join(subpath, "*.csv")):
    #             df = pd.read_csv(file)
    #             # Normalize data:
    #             if normalize:
    #                 df[features] = (df[features] - df[features].mean()) / df[features].std()
    #             sequences.append(df[features])
    #             labels.append(df['label'][0])
                
    #     self.sequences = sequences
    #     self.labels = label_encoder.transform([label for label in labels])
    #     self.label_encoder = label_encoder

    def __init__(self, paths, features, label_encoder, normalize=False, add_fft = False):
        sequences = []
        labels = []
        for path in paths:
          for subpath, subdir, files in os.walk(path):
              for file in glob(os.path.join(subpath, "*.csv")):
                  df = pd.read_csv(file)
                  # Normalize data:
                  if normalize:
                      df[features] = (df[features] - df[features].mean()) / df[features].std()
                  
                  
                  if add_fft:
                      new_features = features.copy()
                      for feat in features:
                          fourier = fft(df[feat].values)
                          sos = signal.butter(1, 20, 'lowpass', fs=50, output='sos')
                          fourier = signal.sosfilt(sos, abs(fourier))
                          new_features.append(feat+'_fft')
                          df[feat+'_fft'] = abs(fourier)
                      sequences.append(df[new_features])
                  else:
                      sequences.append(df[features])

                  labels.append(df['label'][0])
                  
        self.sequences = sequences
        self.labels = label_encoder.transform([label for label in labels])
        self.label_encoder = label_encoder
    
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx].to_numpy()
        label = self.labels[idx]
        return torch.tensor(sequence, dtype=torch.float), torch.tensor(label).long()

    def get_labels(self):          
        return self.labels  


class HandFaceDataModule(pl.LightningDataModule):
    def __init__(self, train_path, test_path, features, label_encoder, batch_size, normalize, add_fft = False):
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.features = features
        self.label_encoder = label_encoder
        self.batch_size = batch_size
        self.normalize = normalize 
        self.add_fft = add_fft
        
    def setup(self, stage=None):
        self.train_dataset = HandFaceDataset(self.train_path, self.features, self.label_encoder, self.normalize, self.add_fft)
        self.test_dataset = HandFaceDataset(self.test_path, self.features, self.label_encoder, self.normalize, self.add_fft)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            sampler = ImbalancedDatasetSampler(self.train_dataset),
            #shuffle = True,
            num_workers = cpu_count()
        )
      
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = cpu_count()
        )