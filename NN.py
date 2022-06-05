
import torch.nn as nn
import torch
import torch_geometric.nn as pyg_nn
from torch.nn.functional import conv2d
from createdata import dim, step, BB
import torch.nn.functional as F

PTS = torch.from_numpy(BB).float()

class NeuralNet(nn.Module):
    def __init__(self, layer_sizes, cnn_layer_sizes, dropout, device="cuda"):
        super(NeuralNet, self).__init__()
        self.device = device
        conv_layers = []
        layers = []

        input_features = 24 + 3
        for i, output_features in enumerate(cnn_layer_sizes):
            conv_layers.append(nn.Conv3d(input_features, output_features, kernel_size=(3,3,3)))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool3d(2,2))
            input_features = output_features
        self.conv_layers = nn.Sequential(*conv_layers)

        
        input_features = (input_features*8*8*8)
        for i, output_features in enumerate(layer_sizes):
            layers.append(nn.Linear(input_features, output_features))
            if i != len(layer_sizes) - 1:
                layers.append(nn.ReLU())
                #layers.append(nn.BatchNorm1d(output_features))
                layers.append(nn.Dropout(dropout))
            input_features = output_features
        self.layers = nn.Sequential(*layers)    

    # forward function: 
    def forward(self, Pc, sdf):                                                         # [B,P=8,3] [B,dim,dim,dim] 
        B = Pc.shape[0]
        pts = PTS.to(self.device).unsqueeze(0).expand(B,-1,-1).view(B,dim,dim,dim,3)    # [B,dim,dim,dim,3] 
        pts = pts.permute(0,4,1,2,3)                                                    # [B,3,dim,dim,dim]
        Pc = Pc.view(B,-1)                                                              # [B,8*3=24]   
        Pc = Pc.view(B,-1,1,1,1).repeat(1,1,dim,dim,dim)                                # [B,24,dim,dim,dim]
        x = torch.cat((Pc,pts), dim=1)                                                  # [B,f=24+3,dim,dim,dim]
    
        x = self.conv_layers(x)                     # [B,32,8,8,8]
        
        

        
        x = x.view(B,-1)                            # [B, 32*8*8*8]
        pred = self.layers(x)                       # [B,dim*dim*dim*3]
        pred = pred.view(B,dim,dim,dim,3)           # [B,dim,dim,dim,3]

        return pred