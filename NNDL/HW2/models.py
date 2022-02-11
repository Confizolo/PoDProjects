import torch
import numpy as np
import torch.nn as nn

#LIGHTNING
import pytorch_lightning as pl
import torchmetrics
from utilities import add_noise

class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim, act_func = nn.ReLU, in_channels = [8,16,32], linear_size = 64, device = "cpu"):
        super().__init__()
        
        self.device = device
        self.kl = 0
        ### convolutional section
        self.encoder_cnn = nn.Sequential(
            # first convolutional layer
            nn.Conv2d(in_channels=1, out_channels=in_channels[0], kernel_size=3, 
                      stride=2, padding=1),
            act_func(True),
            # second convolutional layer
            nn.Conv2d(in_channels=in_channels[0], out_channels=in_channels[1], kernel_size=3, 
                      stride=2, padding=1),
            act_func(True),
            # third convolutional layer
            nn.Conv2d(in_channels=in_channels[1], out_channels=in_channels[2], kernel_size=3, 
                      stride=2, padding=0),
            act_func(True)
        ).to(device)
        
        ### flatten layer
        self.flatten = nn.Flatten(start_dim=1).to(device)

        ### linear section
        self.encoder_lin = nn.Sequential(
            # first linear layer
            nn.Linear(in_features=in_channels[2]*3*3, out_features=linear_size),
            act_func(True),
            # second linear layer
            nn.Linear(in_features=linear_size, out_features=encoded_space_dim)
        ).to(device)
        
    def forward(self, x):
        x = x
        # apply convolutions
        x = self.encoder_cnn(x)
        # flatten
        x = self.flatten(x)
        # apply linear layers
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim, act_func = nn.ReLU, in_channels = [32,16,8], linear_size = 64,device = "cpu"):
        super().__init__()
        self.device = device 

        ### linear section
        self.decoder_lin = nn.Sequential(
            # first linear layer
            nn.Linear(in_features=encoded_space_dim, out_features=linear_size),
            act_func(True),
            # second linear layer
            nn.Linear(in_features=linear_size, out_features=3*3*in_channels[0]),
            act_func(True)
        ).to(device)

        ### unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(in_channels[0], 3, 3)).to(device)

        ### convolutional section
        self.decoder_conv = nn.Sequential(
            # first transposed convolution
            nn.ConvTranspose2d(in_channels=in_channels[0], out_channels=in_channels[1], kernel_size=3, 
                               stride=2,  output_padding=0),
            act_func(True),
            # second transposed convolution
            nn.ConvTranspose2d(in_channels=in_channels[1], out_channels=in_channels[2], kernel_size=3, 
                               stride=2, padding=1, output_padding=1),
            act_func(True),
            # third transposed convolution
            nn.ConvTranspose2d(in_channels=in_channels[2], out_channels=1, kernel_size=3, 
                               stride=2, padding=1, output_padding=1)
        ).to(device)
        
    def forward(self, x):
        
        x = x
        # apply linear layers
        x = self.decoder_lin(x)
        # unflatten
        x = self.unflatten(x)
        # apply transposed convolutions
        x = self.decoder_conv(x)
        # apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
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
        # apply encoder
        x = self.encoder(x)

        # apply decoder
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
        self.log(loss_name, loss, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx, loss_name = 'val_loss'):
        x, y = batch
        z = self.forward(x)
        loss = self.loss_fn(z, x) 
        self.log(loss_name, loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch 
        z = self.forward(x)
        self.log('accuracy', self.accuracy(z, y), prog_bar=True, on_epoch=True, on_step=True)
        return self.accuracy(z, y)

class DenoisingAE(pl.LightningModule):


    def __init__(self,  hyper,  device = "cpu"):
        super().__init__()

        self.hyper = hyper 

        self.loss_fn = nn.MSELoss()

        self.encoder = Encoder(encoded_space_dim = self.hyper['encoded_space_dim'],act_func = self.hyper["act_func"],in_channels = self.hyper["in_channels"], linear_size = self.hyper["linear_size"], device = device)

        self.decoder = Decoder(encoded_space_dim = self.hyper['encoded_space_dim'],act_func = self.hyper["act_func"],in_channels = self.hyper["in_channels"][::-1], linear_size = self.hyper["linear_size"], device = device)


    def forward(self, x):
        # apply encoder
        x = self.encoder(add_noise(x))

        # apply decoder
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
        self.log(loss_name, loss, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx, loss_name = 'val_loss'):
        x, y = batch
        z = self.forward(x)
        loss = self.loss_fn(z, x) 
        self.log(loss_name, loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch 
        z = self.forward(x)
        self.log('accuracy', self.accuracy(z, y), prog_bar=True, on_epoch=True, on_step=True)
        return self.accuracy(z, y)

class VariationalAE(pl.LightningModule):


    def __init__(self, hyper,  device = "cpu"):
        super().__init__()

        self.device_data = device
        self.hyper = hyper 

        self.encoder = Encoder(encoded_space_dim = self.hyper["out_linear_size"], act_func = self.hyper["act_func"],in_channels = self.hyper["in_channels"], linear_size = self.hyper["linear_size"], device = device)

        # instantiation of the two linear layers giving means and variances
        self.linear2 = nn.Linear(self.hyper["out_linear_size"], self.hyper['encoded_space_dim']).to(device)
        self.linear3 = nn.Linear(self.hyper["out_linear_size"], self.hyper['encoded_space_dim']).to(device)
        
        self.decoder = Decoder(encoded_space_dim = self.hyper['encoded_space_dim'],act_func = self.hyper["act_func"],in_channels = self.hyper["in_channels"][::-1], linear_size = self.hyper["linear_size"], device = device)

        self.loss_fn = nn.functional.binary_cross_entropy
        
    def reparameterization(self, mean, var):
        eps = torch.randn_like(var).to(self.device_data)        # sampling a random variable       
        z = mean + var*eps                                      # reparametrization
        return z

    def forward(self, x):
        # apply encoder
        x = self.encoder(x)

        # calculating encoded representation of mean and variances using the output of last encoder layer
        mean =  self.linear2(x).to(self.device_data)
        log_var = self.linear3(x).to(self.device_data)

        # reparametrization and Kullback-Leibler divergence calculation
        x = self.reparameterization(mean, torch.exp(0.5 * log_var))
        self.kl = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        # apply decoder
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

        # custom loss calculation
        loss = self.loss_fn(z, x, reduction="sum") + self.kl

        self.log(loss_name, loss, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx, loss_name = 'val_loss'):
        x, y = batch
        z = self.forward(x)

        # custom loss calculation
        loss = self.loss_fn(z, x, reduction = "sum") + self.kl
        self.log(loss_name, loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch 
        z = self.forward(x)
        self.log('accuracy', self.accuracy(z, y), prog_bar=True, on_epoch=True, on_step=True)
        return self.accuracy(z, y)


class SupervisedCAE(pl.LightningModule):
 
    def __init__(self,  hyper, device = "cpu",  PATH = None):

        super().__init__()

        self.hyper = hyper 

        self.encoder = Encoder(encoded_space_dim = self.hyper["encoded_space_dim"],act_func = self.hyper["act_func"],in_channels = self.hyper["in_channels"], linear_size = self.hyper["linear_size"], device = device)

        self.encoder.load_state_dict(state_dict = torch.load(PATH))

        # prevent encoder training
        for param in self.encoder.parameters():
            param.requires_grad = False

        # fine tuning layer
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
        x = self.fine_tuner(x)
        return x

    def training_step(self, batch, batch_idx,  loss_name = 'train_loss'):
        x, y = batch
        z = self.forward(x)
        loss = self.loss_fn(z, y)
        self.log(loss_name, loss, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx, loss_name = 'val_loss'):
        x, y = batch
        z = self.forward(x)
        loss = self.loss_fn(z, y)
        self.log(loss_name, loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch 
        z = self.forward(x)
        self.log('accuracy', self.accuracy(z, y), prog_bar=True, on_epoch=True, on_step=True)
        return self.accuracy(z, y)
    
