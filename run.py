import wandb
from NN import NeuralNet
import torch
from dataset import ContourDataset
from trainer import train
from torch.utils.data import DataLoader
from torch import autograd
import numpy as np
import random 
import wandb


# Seed
seed = 10 # 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from createdata import dim



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run():
    # Hyper-parameters
    cnn_layer_sizes = [32,32,32]
    layer_sizes = [128,128,128, dim*dim*dim*3] 
    num_epochs = 5
    batch_size = 12  
    learning_rate = 0.0025385948251533917
    dropout=0.0
    dict = {
        "learning_rate":learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "cnn_layer_sizes": cnn_layer_sizes,
        "layer_sizes": layer_sizes
    }

    use_existing_model = False
    # Dataset
    train_dataset = ContourDataset()
    test_dataset = ContourDataset(split="test")
    validation_dataset = ContourDataset(split="validation")
    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True, num_workers=3, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=3, pin_memory=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=3, pin_memory=True)
    
    model = NeuralNet(layer_sizes, cnn_layer_sizes, dropout).to(device)
    if use_existing_model:
        model.load_state_dict(torch.load("model.pth")["state_dict"])
    #with autograd.detect_anomaly():
    wandb.init(project="3d_conv3d_fully_connected", entity="master-thesis-ntnu", config=dict)

    model = train(model, num_epochs, batch_size, train_loader, test_loader, validation_loader, learning_rate=learning_rate, device=device)
    #train(model, num_epochs, batch_size, train_loader, test_loader, validation_loader, learning_rate=learning_rate, device=device)
    
    # Save model
    config = {
        "state_dict": model.state_dict()
    }

    torch.save(config, "model_end.pth")

if __name__ == "__main__":
    run()