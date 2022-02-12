import matplotlib.pyplot as plt
import torch
import seaborn as sns
import numpy as np
from pytorch_lightning import Callback
import copy 

def add_noise(inputs,noise_factor=0.3):
     # Adding noise to an image
     noisy = inputs+torch.randn_like(inputs) * noise_factor
     noisy = torch.clip(noisy,0.,1.)
     return noisy

def plot_result(img, ax):
    # Plot a in image in greyscale
    ax.imshow(img, cmap='gist_gray')
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    plt.tight_layout()

def multiple_plot(plot_func, grid_size, figsize , args):
     '''
     plot a sequence of images in greyscale
     ''' 
     fig, axes = plt.subplots(nrows=grid_size[0],ncols=grid_size[1], figsize=figsize)
     for  i,ax in enumerate(axes.flatten()):
          plot_func(**args[i], ax=ax)
     fig.show()

def loss_plot(train_loss,val_loss, ax):
     """
     Plot train and validation loss
     """

     sns.lineplot(x=np.arange(len(train_loss)),y=train_loss, label='Train loss', markers=True,  ax=ax)
     sns.lineplot(x=np.arange(len(val_loss)),y=val_loss, label='Validation loss',markers=True, ax=ax)

     ax.set_yscale("log")

     plt.xlabel('Iteration')
     plt.ylabel('Loss')


class MetricsCallback(Callback):
     """PyTorch Lightning metric callback adjusted to get train and test loss for the training process"""

     def __init__(self):
          super().__init__()
          self.metrics = {"train_loss":[], "val_loss":[]}

     def on_validation_end(self, trainer, pl_module):
          if "train_loss" in trainer.callback_metrics.keys(): 
               self.metrics["train_loss"].append(copy.deepcopy(trainer.callback_metrics["train_loss"]).numpy())
          if "val_loss" in trainer.callback_metrics.keys(): 
               self.metrics["val_loss"].append(copy.deepcopy(trainer.callback_metrics["val_loss"]).numpy())
         
     def on_train_batch_end(self, trainer, pl_module, outputs,batch,batch_idx):
          if "train_loss" in trainer.callback_metrics.keys(): 
               self.metrics["train_loss"].append(copy.deepcopy(trainer.callback_metrics["train_loss"]).numpy())
          if "val_loss" in trainer.callback_metrics.keys(): 
               self.metrics["val_loss"].append(copy.deepcopy(trainer.callback_metrics["val_loss"]).numpy())
         