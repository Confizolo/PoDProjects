import torch
import numpy as np
import torch.nn as nn

#LIGHTNING
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms, datasets
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning import loggers as pl_loggers
import torchmetrics
from utilities import add_noise

class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim, act_func = nn.ReLU, in_channels = [8,16,32], linear_size = 64, device = "cpu"):
        super().__init__()
        
        self.device = device
        self.kl = 0
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
        x = x
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = self.flatten(x)
        # # Apply linear layers
        x = self.encoder_lin(x)
        return x

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
        
        x = x
        # Apply linear layers
        x = self.decoder_lin(x)
        # Unflatten
        x = self.unflatten(x)
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        x = torch.sigmoid(x)
        return x

class StandardAE(pl.LightningModule):


    def __init__(self,  hyper,  device = "cpu"):
        super().__init__()

        self.hyper = hyper 

        self.loss_fn = nn.MSELoss()

        self.encoder = Encoder(encoded_space_dim = self.hyper['encoded_space_dim'],act_func = self.hyper["act_func"],in_channels = self.hyper["in_channels"], linear_size = self.hyper["linear_size"], device = device)

        self.decoder = Decoder(encoded_space_dim = self.hyper['encoded_space_dim'],act_func = self.hyper["act_func"],in_channels = self.hyper["in_channels"][::-1], linear_size = self.hyper["linear_size"], device = device)

        
    def forward(self, x):
        # Apply encoder
        x = self.encoder(x)

        # Apply decoder
        x = self.decoder(x)

        return x

    def configure_optimizers(self):

        opt      = self.hyper['opt']
        lr       = self.hyper['lr']
        reg      = self.hyper['reg']

        if opt == 'SGD':
            return torch.optim.SGD(self.parameters(),  lr=lr, momentum=self.hyper["mom"], weight_decay=reg)
        elif opt == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=reg)
                

    def training_step(self, batch, batch_idx,  loss_name = 'train_loss'):
        x, y = batch
        z = self.forward(x)

        loss = self.loss_fn(z, x) 
        self.log(loss_name, loss)
        return loss
    
    def validation_step(self, batch, batch_idx, loss_name = 'val_loss'):
        x, y = batch
        z = self.forward(x)
        loss = self.loss_fn(z, x) 
        self.log(loss_name, loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch #now labels matter!
        z = self.forward(x)
        self.log('accuracy', self.accuracy(z, y), prog_bar=True)
        return self.accuracy(z, y)

class DenoisingAE(pl.LightningModule):


    def __init__(self,  hyper,  device = "cpu"):
        super().__init__()

        self.hyper = hyper 

        self.loss_fn = nn.MSELoss()

        self.encoder = Encoder(encoded_space_dim = self.hyper['encoded_space_dim'],act_func = self.hyper["act_func"],in_channels = self.hyper["in_channels"], linear_size = self.hyper["linear_size"], device = device)

        self.decoder = Decoder(encoded_space_dim = self.hyper['encoded_space_dim'],act_func = self.hyper["act_func"],in_channels = self.hyper["in_channels"][::-1], linear_size = self.hyper["linear_size"], device = device)


    def forward(self, x):
        # Apply encoder
        x = self.encoder(add_noise(x))

        # Apply decoder
        x = self.decoder(x)

        return x
    def configure_optimizers(self):

        opt      = self.hyper['opt']
        lr       = self.hyper['lr']
        reg      = self.hyper['reg']

        if opt == 'SGD':
            return torch.optim.SGD(self.parameters(),  lr=lr, momentum=self.hyper["mom"], weight_decay=reg)
        elif opt == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=reg)
                

    def training_step(self, batch, batch_idx,  loss_name = 'train_loss'):
        x, y = batch
        z = self.forward(x)

        loss = self.loss_fn(z, x) 
        self.log(loss_name, loss)
        return loss
    
    def validation_step(self, batch, batch_idx, loss_name = 'val_loss'):
        x, y = batch
        z = self.forward(x)
        loss = self.loss_fn(z, x) 
        self.log(loss_name, loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch #now labels matter!
        z = self.forward(x)
        self.log('accuracy', self.accuracy(z, y), prog_bar=True)
        return self.accuracy(z, y)

class VariationalAE(pl.LightningModule):


    def __init__(self, hyper,  device = "cpu"):
        super().__init__()

        self.hyper = hyper 

        self.loss_fn = nn.MSELoss()

        self.encoder = Encoder(encoded_space_dim = self.hyper["out_linear_size"],act_func = self.hyper["act_func"],in_channels = self.hyper["in_channels"], linear_size = self.hyper["linear_size"], device = device)

        self.linear2 = nn.Linear(self.hyper["out_linear_size"], self.hyper['encoded_space_dim']).to(device)
        self.linear3 = nn.Linear(self.hyper["out_linear_size"], self.hyper['encoded_space_dim']).to(device)
        
        self.N = torch.distributions.Normal(0, 1)

        self.decoder = Decoder(encoded_space_dim = self.hyper['encoded_space_dim'],act_func = self.hyper["act_func"],in_channels = self.hyper["in_channels"][::-1], linear_size = self.hyper["linear_size"], device = device)

    def forward(self, x):
        # Apply encoder
        x = self.encoder(x)

        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        
        x = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        # Apply decoder
        x = self.decoder(x)

        return x

    def configure_optimizers(self):

        opt      = self.hyper['opt']
        lr       = self.hyper['lr']
        reg      = self.hyper['reg']

        if opt == 'SGD':
            return torch.optim.SGD(self.parameters(),  lr=lr, momentum=self.hyper["mom"], weight_decay=reg)
        elif opt == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=reg)
                

    def training_step(self, batch, batch_idx,  loss_name = 'train_loss'):
        x, y = batch
        z = self.forward(x)

        loss = self.loss_fn(z, x) + self.kl
        self.log(loss_name, loss)
        return loss
    
    def validation_step(self, batch, batch_idx, loss_name = 'val_loss'):
        x, y = batch
        z = self.forward(x)
        loss = self.loss_fn(z, x) + self.kl
        self.log(loss_name, loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch #now labels matter!
        z = self.forward(x)
        self.log('accuracy', self.accuracy(z, y), prog_bar=True)
        return self.accuracy(z, y)


class SupervisedCAE(pl.LightningModule):
 
    def __init__(self,  hyper, device = "cpu",  PATH = None):

        super().__init__()

        self.hyper = hyper 

        self.encoder = Encoder(encoded_space_dim = self.hyper["encoded_space_dim"],act_func = self.hyper["act_func"],in_channels = self.hyper["in_channels"], linear_size = self.hyper["linear_size"], device = device)

        self.encoder.load_state_dict(state_dict = torch.load(PATH))

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.fine_tuner =  nn.Sequential(nn.Linear(self.hyper['encoded_space_dim'],self.hyper["out_linear_size"]),
                                         nn.ReLU(),
                                         nn.Linear(self.hyper["out_linear_size"], 10),
                                         nn.LogSoftmax()
                                         )
                                         
        self.accuracy = torchmetrics.Accuracy()
    
        self.loss_fn = nn.CrossEntropyLoss()

    def configure_optimizers(self):

        opt      = self.hyper['opt']
        lr       = self.hyper['lr']
        reg      = self.hyper['reg']

        if opt == 'SGD':
            return torch.optim.SGD(self.parameters(),  lr=lr, momentum=self.hyper["mom"], weight_decay=reg)
        elif opt == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=reg)
                
    def forward(self,  x):

        x = self.encoder(x)

        return self.fine_tuner(x)

    def training_step(self, batch, batch_idx,  loss_name = 'train_loss'):
        x, y = batch
        z = self.forward(x)
        loss = self.loss_fn(z, y)
        self.log(loss_name, loss)
        return loss
    
    def validation_step(self, batch, batch_idx, loss_name = 'val_loss'):
        x, y = batch
        z = self.forward(x)
        loss = self.loss_fn(z, y)
        self.log(loss_name, loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch 
        z = self.forward(x)
        self.log('accuracy', self.accuracy(z, y), prog_bar=True)
        return self.accuracy(z, y)
    
