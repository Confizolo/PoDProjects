import torch.nn as nn
import torch

class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim, act_func = nn.ReLU, in_channels = [8,16,32], linear_size = 64, device = "cpu"):
        super().__init__()
        
        self.device = device

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(in_channels=1, out_channels=in_channels[0], kernel_size=3, 
                      stride=2, padding=1),
            act_func(True),
            # Second convolutional layer
            nn.Conv2d(in_channels=in_channels[0], out_channels=in_channels[1], kernel_size=3, 
                      stride=2, padding=1),
            act_func(True),
            # Third convolutional layer
            nn.Conv2d(in_channels=in_channels[1], out_channels=in_channels[2], kernel_size=3, 
                      stride=2, padding=0),
            act_func(True)
        ).to(device)
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1).to(device)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(in_features=in_channels[2]*3*3, out_features=linear_size),
            act_func(True),
            # Second linear layer
            nn.Linear(in_features=linear_size, out_features=encoded_space_dim)
        ).to(device)
        
    def forward(self, x):
        x = x.to(self.device)
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = self.flatten(x)
        # # Apply linear layers
        x = self.encoder_lin(x)
        return x