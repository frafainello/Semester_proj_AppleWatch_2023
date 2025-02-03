import numpy as np
import pickle
import torch
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import wandb

from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import Accuracy, F1Score, ConfusionMatrix
from multiprocessing import cpu_count
from torchsampler import ImbalancedDatasetSampler

import os
from glob import glob
from multiprocessing import cpu_count

from scheduler import CosineWarmupScheduler

### LOSS ###
class FocalLoss(nn.Module):
    '''
    Multi-class Focal Loss
    '''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C], float32
        target: [N, ], int64
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss
    
### BASE MODULE ###

class Model(pl.LightningModule):
    def __init__(self, n_features, n_classes, learning_rate, warmup, warmup_iters, datamodule):
        super().__init__()
        self.learning_rate = learning_rate
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.datamodule = datamodule
        #self.criterion = nn.CrossEntropyLoss()
        self.criterion = FocalLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.f1 = F1Score(task="multiclass", num_classes=n_classes)
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=n_classes)

        self.best_f1 = 0
        self.fig = None

    def training_step(self, batch, batch_idx):
        y_hat, y_pred, y_true, loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, logger=False, on_step=True)
        # self.log("lr", self.optimizer.param_groups[0]["lr"], prog_bar=True, logger=False, on_step=True)
        self.logger.experiment.add_scalar('train_loss', loss, self.current_epoch)

        # Learning rate scheduler step
        sch = self.lr_scheduler
        sch.step()

        return {"loss":loss, "y_pred": y_pred, "y_true": y_true}
    
    def validation_step(self, batch, batch_idx):
        y_hat, y_pred, y_true, loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, logger=False, on_step=True)
        self.logger.experiment.add_scalar('val_loss', loss, self.current_epoch)
        return {"loss":loss, "y_pred": y_pred, "y_true": y_true}
    
    def _shared_step(self, batch, batch_idx):
        x, y_true = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y_true)
        y_pred = self.predict_step(batch, batch_idx)
        return y_hat, y_pred, y_true, loss
    
    def training_epoch_end(self, outputs): # OLD: training_epoch_end 
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)
        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        acc = self.accuracy(y_pred, y_true)
        fig_ = self._plot_cm(y_pred, y_true)
        self.logger.experiment.add_figure("Confusion matrix train", fig_, self.current_epoch)
        self.log("train_acc", acc, prog_bar=True, logger=False)
        self.logger.experiment.add_scalar('train_acc', acc, self.current_epoch)

    def validation_epoch_end(self, outputs): # OLD: validation_epoch_end 
        y_true = torch.cat([x['y_true'] for x in outputs], dim=0)
        y_pred = torch.cat([x['y_pred'] for x in outputs], dim=0)
        acc = self.accuracy(y_pred, y_true)
        f1 = self.f1(y_pred, y_true)
        fig_ = self._plot_cm(y_pred, y_true)
        self.log("val_acc", acc, prog_bar=True, logger=False)
        self.log("f1", f1, prog_bar=True, logger=False)
        self.logger.experiment.add_scalar('val_acc', acc, self.current_epoch)
        self.logger.experiment.add_scalar('f1', f1, self.current_epoch)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

        if self.current_epoch == 1:
          self.best_f1 = f1
        if f1 > self.best_f1:
          try:
            wandb.log({'f1': f1, 'epoch' : self.current_epoch})
            self.best_f1 = f1
          except:
            self.best_f1 = f1
          self.fig = fig_
          

    
    def _plot_cm(self, y_pred, y_true):
        cm = self.confmat(y_pred, y_true)
        cm = cm / cm.sum(axis=1)[:, None]
        fig, ax = plt.subplots(figsize=(10,10))
        fig_ = sns.heatmap(cm.to('cpu').numpy(), vmin=0, vmax=1, annot=True, cmap='Blues', ax=ax).get_figure()
        return fig_
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_pred = torch.argmax(y_hat, dim=1)
        return y_pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-6)
        
        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup, max_iters=self.warmup_iters
        )
        # self.lr_scheduler = SimpleWarmupScheduler(
        #     optimizer, warmup=500
        # )
        return optimizer

### SELF ATTENTION ###

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / np.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


# Multihead Attention module
class MultiheadAttention(nn.Module):
    
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        # Embedding dim is sum over all heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, input_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

# Attention block employing multihead self attention + linear layer, norm, residual
class AttentionBlock(nn.Module):
    def __init__(self, model_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads to use in the attention block
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(model_dim, model_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(model_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, model_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


# Module to add positional encoding as a feature
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        """
        Args
            model_dim: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-np.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


# Final attention layer before classification network
# as used by "Human Activity Recognition from Wearable Sensor Data Using Self-Attention"
# Saif Mahmud and M. Tanjid Hasan Tonmoy et al.
class GlobalTemporalAttention(nn.Module):
    def __init__(self, model_dim, dropout=0.0):
        """
        Args:
            
        """
        super().__init__()

        self.linear_tanh = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Dropout(dropout),
            nn.Tanh(),
        )

        # self.g = nn.Parameter(torch.Tensor(model_dim,1))
        self.g_net = nn.Linear(model_dim, 1)

    def forward(self, x, mask=None):
        
        uit = self.linear_tanh(x)
        # ait = torch.matmul(uit, self.g)
        ait = self.g_net(uit)
        a = torch.exp(ait)
        
        if mask is not None:
            a *= mask.unsqueeze(2)
        
        a = a / ( torch.sum(a, dim=1, keepdim=True) + torch.finfo(torch.float32).eps )
        
        # a = a.unsqueeze(2)
        weighted_input = x * a
        result = torch.sum(weighted_input, dim=1)
        
        return result


#####################################################
# Full model with self attention
class AttentionModel(Model):
    def __init__(self, n_features, n_classes,learning_rate, datamodule,
                 seq_length,input_dim,model_dim,num_heads,num_layers,
                 warmup=0,warmup_iters=10000,dropout=0.2):
        super().__init__(n_features, n_classes,learning_rate, warmup, warmup_iters, datamodule)
        
        """
        Args:
            n_features: Number of features per sequence element
            n_classes: Number of classes to predict per sequence element
            
            input_dim: Hidden dimensionality of the input
            model_dim: Hidden dimensionality to use inside the Transformer
            num_heads: Number of heads to use in the Multi-Head Attention blocks
            num_layers: Number of encoder blocks to use
            
            lr: Learning rate in the optimizer
            warmup: Number of warmup steps. Usually between 50 and 500
            max_iters: Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout: Dropout to apply inside the model
        """
        self.kwargs = {'n_features': n_features, 'n_classes': n_classes, 'learning_rate': learning_rate, 
                       'datamodule': datamodule, 'seq_length': seq_length, 'input_dim': input_dim,
                       'model_dim': model_dim, 'num_heads': num_heads,
                       'num_layers': num_layers, 'warmup': warmup, 'warmup_iters': warmup_iters, 'dropout': dropout}
        # Input embedding with ffnn
        # self.input_net = nn.Sequential(
        #     nn.Dropout(dropout), 
        #     nn.Linear(input_dim, model_dim)
        # )

        # Input embedding with 1D convolution
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=model_dim, kernel_size=1)
        self.relu = nn.ReLU()

        # Positional encoding
        self.positional_encoding = PositionalEncoding(model_dim)


        # Self attention blocks
        self.num_layers = num_layers
        self.attn_layers = nn.ModuleList()
        for i in range(num_layers):
          self.attn_layers.append(AttentionBlock(model_dim, num_heads, 3*model_dim, dropout = dropout))

        # Final global attention
        self.global_attn = GlobalTemporalAttention(model_dim, dropout=dropout)
        
        # Classification nn
        self.classifier = nn.Linear(model_dim, n_classes)
        
    def forward(self, x):
        
        # Input dim -> Model dim WITH FFNN
        # x = self.input_net(x)
        
        # Input dim -> Model dim WITH CONV1D
        x = x.permute(0, 2, 1)  # Transpose the input to match PyTorch's expected shape
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # Transpose the output back to its original shape

        # Positional encoding
        x = self.positional_encoding(x)

        for layer in self.attn_layers:
            x = layer(x)

        # Linear network and classifier
        # lin_out = self.net(attn_out)
        lin_out = self.global_attn(x)  

        return self.classifier(lin_out)
    
    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """Function for extracting the attention matrices of the whole Transformer for a single batch.

        Input arguments same as the forward pass.
        """
        x = x.permute(0, 2, 1)  # Transpose the input to match PyTorch's expected shape
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # Transpose the output back to its original shape

        # Positional encoding
        x = self.positional_encoding(x)
        
        attention_maps = []
        for layer in self.attn_layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps
    
    
### LSTM + SELF ATTENTION ###

class LSTM_AttentionModel(Model):
    def __init__(self, n_features, n_classes,learning_rate, datamodule,
                 seq_length,input_dim,model_dim,num_heads,lstm_layers,num_layers,
                 warmup=0,warmup_iters=10000,dropout=0.2):
        super().__init__(n_features, n_classes,learning_rate, warmup, warmup_iters, datamodule)
        
        """
        Args:
            n_features: Number of features per sequence element
            n_classes: Number of classes to predict per sequence element
            
            input_dim: Hidden dimensionality of the input
            model_dim: Hidden dimensionality to use inside the Transformer
            num_heads: Number of heads to use in the Multi-Head Attention blocks
            lstm_layers: Number of LSTM layers
            num_layers: Number of encoder blocks to use.
            
            lr: Learning rate in the optimizer
            warmup: Number of warmup steps. Usually between 50 and 500
            max_iters: Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout: Dropout to apply inside the model
        """
        self.kwargs = {'n_features': n_features, 'n_classes': n_classes, 'learning_rate': learning_rate, 
                       'datamodule': datamodule, 'seq_length': seq_length, 'input_dim': input_dim,
                       'model_dim': model_dim, 'num_heads': num_heads, 'lstm_layers': lstm_layers,
                       'num_layers': num_layers, 'warmup': warmup, 'warmup_iters': warmup_iters, 'dropout': dropout}

        # Input embedding with 1D convolution
        # self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=model_dim, kernel_size=1)
        # self.relu = nn.ReLU()

        # LSTM
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=model_dim, num_layers=lstm_layers, batch_first=True, dropout=dropout)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(model_dim)

        # Self attention blocks
        self.num_layers = num_layers
        self.attn_layers = nn.ModuleList()
        for i in range(num_layers):
          self.attn_layers.append(AttentionBlock(model_dim, num_heads, 3*model_dim, dropout = dropout))

        # Final global attention
        self.global_attn = GlobalTemporalAttention(model_dim, dropout=dropout)
        
        # Classification nn
        self.classifier = nn.Sequential(nn.Linear(model_dim, model_dim),
                                        nn.Linear(model_dim, n_classes),
                                        )

        
    def forward(self, x):
        # LSTM
        x, (h_n,c_n) = self.lstm(x)

        # Positional encoding
        x = self.positional_encoding(x)

        for layer in self.attn_layers:
            x = layer(x)

        # Linear network and classifier
        lin_out = self.global_attn(x)  

        return self.classifier(lin_out)

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """Function for extracting the attention matrices of the whole Transformer for a single batch.

        Input arguments same as the forward pass.
        """
        # LSTM
        x, (h_n,c_n) = self.lstm(x)

        # Positional encoding
        x = self.positional_encoding(x)
        
        attention_maps = []
        for layer in self.attn_layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps


# BASELINE MODELS
class CNN(Model):
    def __init__(self, n_features, n_classes, learning_rate, warmup, warmup_iters, datamodule):
        super().__init__(n_features, n_classes, learning_rate, warmup, warmup_iters, datamodule)
        
        self.kwargs = {'n_features': n_features, 'n_classes': n_classes, 'learning_rate': learning_rate, 
                'datamodule': datamodule, 'warmup': warmup, 'warmup_iters': warmup_iters}

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=5),
            nn.Dropout1d(p=0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5),
            nn.Dropout1d(p=0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5),
            nn.Dropout1d(p=0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5),
            nn.Dropout1d(p=0.5),
            nn.ReLU(),
            nn.Flatten()
        )
        self.classifier = nn.Linear(2176, n_classes)
        
    def forward(self, x):
        out = self.conv(x.permute(0, 2, 1))
        return self.classifier(out)

class DeepConvLSTM(Model):
    def __init__(self, n_features, n_classes, learning_rate, warmup, warmup_iters, datamodule):
        super().__init__(n_features, n_classes, learning_rate, warmup, warmup_iters, datamodule)
        
        self.kwargs = {'n_features': n_features, 'n_classes': n_classes, 'learning_rate': learning_rate, 
                'datamodule': datamodule, 'warmup': warmup, 'warmup_iters': warmup_iters}

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=5),
            nn.Dropout1d(p=0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5),
            nn.Dropout1d(p=0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5),
            nn.Dropout1d(p=0.5),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5),
            nn.Dropout1d(p=0.5),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.75
        )
        self.classifier = nn.Linear(128, n_classes)
        
    def forward(self, x):
        # CNN
        out = self.conv(x.permute(0, 2, 1))

        # LSTM
        out = out.permute(0, 2, 1)
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(out)
        out = hidden[-1]

        return self.classifier(out)