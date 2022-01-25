import torch.nn as nn
from encoders import Encoder
from decoders import Decoder
import torch.optim as toptim
from torch.utils.data import Dataset, DataLoader
from training_func import train_epoch,test_epoch
import optuna

def objective(trial, train_dataset, test_dataset, loss_fn):
    # Try different 
    in_channels_enc = [trial.suggest_int("in_channels_enc1", 10, 256,step=32),
                        trial.suggest_int("in_channels_enc2", 10, 256,step=32),
                        trial.suggest_int("in_channels_enc3", 10, 256,step=32)]
    in_channels_dec = [trial.suggest_int("in_channels_enc1", 10, 256,step=32),
                        trial.suggest_int("in_channels_enc2", 10, 256,step=32),
                        trial.suggest_int("in_channels_enc3", 10, 256,step=32)]
    act_func_enc = trial.suggest_categorical("act_func_enc", [nn.ReLU])
    act_func_dec = trial.suggest_categorical("act_func_dec", [nn.ReLU])
    linear_size_enc = trial.suggest_int("linear_size_enc", 10, 256,step=32)
    linear_size_dec = trial.suggest_int("linear_size_enc", 10, 256,step=32)
    encoded_space_dim = trial.suggest_int("encoded_space_dim", 1, 10, step=1)

    ### Initialize the two networks
    encoder = Encoder(encoded_space_dim=encoded_space_dim, act_func = act_func_enc, in_channels = in_channels_enc, linear_size = linear_size_enc)
    decoder = Decoder(encoded_space_dim=encoded_space_dim, act_func = act_func_dec, in_channels = in_channels_dec, linear_size = linear_size_dec)
    
    # Generate the optimizers.
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]
    
    # try RMSprop and SGD
    '''
    optimizer_name = trial.suggest_categorical("optimizer", ["RMSprop", "SGD"])
    momentum = trial.suggest_float("momentum", 0.0, 1.0)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr,momentum=momentum)
    '''
    #try Adam, AdaDelta adn Adagrad
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Adadelta","Adagrad"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1,log=True)
    optimizer = getattr(toptim, optimizer_name)(params_to_optimize, lr=lr)
    batch_size=trial.suggest_int("batch_size", 64, 256,step=64)
    
    ### Define train dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    ### Define test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ### Training cycle
    num_epochs = 10
    device = "cpu"
    for epoch in range(num_epochs):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs))
        ### Training (use the training function)
        train_epoch(
            encoder=encoder, 
            decoder=decoder, 
            device=device, 
            dataloader=train_dataloader, 
            loss_fn=loss_fn, 
            optimizer=optimizer)
        ### Validation  (use the testing function)
        val_loss = test_epoch(
            encoder=encoder, 
            decoder=decoder, 
            device=device, 
            dataloader=test_dataloader, 
            loss_fn=loss_fn)

        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

