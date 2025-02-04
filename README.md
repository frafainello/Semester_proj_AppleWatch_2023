# Applying self-attention to detect hand face interactions with Apple Watch
This repo contains key code used for the project. Data is missing since I don't own the rights to it.

## Train_attention.ipynb
Main notebook to train all models and generate all results. 
Included sections of the code are:

### Package installation and import
Installs correct version of required packages for use in Colab. Imports function from .py files.

### Load Data
Loading data for training and testing. Possibility to select between first dataset and most recent dataset

### Data analysis
Brief analysis on distribution of the data

### Train model
Cells to train all models and visualize results using f1 scores, confusion matrices and TensorBoard logs

### Tune hyperparameters
Hyperparameter tuning using online WandB sweep tool

### Transfer learning
Retraining of last layer for transfer learning. Generation of the necessary data and visualization of the results for selected user

## HFI_data
Folder containing all data for training, including processed data for transfer learning

## logs folders
Containing all checkpoints, arguments and tensorboard logs for best trained models.
For LSTM, files for 5 labels with and without FFT are saved

## plots
Plots produced by main notebook

## Helper files
.py files with helper functions
#### models.py
Torch modules for all neural networks, in Pytorch Lightning framework

#### datasets.py
Torch datasets and dataloaders

#### scheduler.py
Learning rate warmup scheduler

#### attention_maps.py
Code for attention map plots

#### helpers.py
Helper function for file management
