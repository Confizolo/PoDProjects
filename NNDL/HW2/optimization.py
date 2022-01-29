from tabnanny import verbose
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#LIGHTNING
import pytorch_lightning as pl

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import LightningLoggerBase

def return_objective(model,train_dataset, EPOCHS, device):
    def objective(trial):
        # Try different 
        in_channels = [trial.suggest_int("in_channels_0", 10, 80,step=10),
                            trial.suggest_int("in_channels_1", 10, 80,step=10),
                            trial.suggest_int("in_channels_2", 10, 80,step=10)]
        linear_size = trial.suggest_int("linear_size", 10, 80,step=10)
        out_linear_size = trial.suggest_int("out_linear_size", 10, 200,step=10)
        encoded_space_dim = trial.suggest_int("encoded_space_dim", 1, 50, step=1)
        
        #try Adam and SGD
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
        momentum = trial.suggest_float("momentum", 0.0, 1.0)

        lr = trial.suggest_float("lr", 1e-5, 1,log=True)
        reg = trial.suggest_float("reg", 1e-5, 1e-2,log=True)

        batch_size=trial.suggest_int("batch_size", 64, 128*3,step=32)
        
        ### Define train dataloader
        train_dataset_split, val_dataset = torch.utils.data.random_split(train_dataset, [50000,10000]) 
        train_dataloader = DataLoader(train_dataset_split, batch_size=batch_size, shuffle=True, num_workers=torch.get_num_threads())
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=torch.get_num_threads())

        model_in = model({"opt":optimizer_name,"lr":lr,"reg":0,"encoded_space_dim":encoded_space_dim,
                         "act_func":nn.ReLU, "linear_size": linear_size, "in_channels":in_channels,
                          "out_linear_size":out_linear_size, "reg":reg, "mom":momentum}, device=device)

        trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        limit_val_batches= 1.,
        max_epochs=EPOCHS,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        weights_summary=None
        )

        trainer.fit(model_in,train_dataloader,val_dataloaders=val_dataloader)

        return  trainer.callback_metrics["val_loss"].item()

    return objective

