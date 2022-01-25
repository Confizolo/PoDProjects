import torch.nn as nn
import torch

class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim, act_func = nn.ReLU, in_channels = [32,16,8], linear_size = 64,device = "cpu"):
        super().__init__()
        self.device = device 

        ### Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(in_features=encoded_space_dim, out_features=linear_size),
            act_func(True),
            # Second linear layer
            nn.Linear(in_features=linear_size, out_features=3*3*in_channels[0]),
            act_func(True)
        ).to(device)

        ### Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(in_channels[0], 3, 3)).to(device)

        ### Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(in_channels=in_channels[0], out_channels=in_channels[1], kernel_size=3, 
                               stride=2,  output_padding=0),
            act_func(True),
            # Second transposed convolution
            nn.ConvTranspose2d(in_channels=in_channels[1], out_channels=in_channels[2], kernel_size=3, 
                               stride=2, padding=1, output_padding=1),
            act_func(True),
            # Third transposed convolution
            nn.ConvTranspose2d(in_channels=in_channels[2], out_channels=1, kernel_size=3, 
                               stride=2, padding=1, output_padding=1)
        ).to(device)
        
    def forward(self, x):
        x = x.to(self.device)
        # Apply linear layers
        x = self.decoder_lin(x)
        # Unflatten
        x = self.unflatten(x)
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        x = torch.sigmoid(x)
        return x