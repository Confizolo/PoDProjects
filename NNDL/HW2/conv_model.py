import torch.nn as nn


class ConvolutionalNet(nn.Module):
    """
    Network with customizable set of hidden layers
    """
    def __init__(self,  dropout=0, act_func=nn.ReLU(), linear_size = 100, conv0_size = 10, conv1_size = 20):
        """
        Initialization function for the network
        
        Parameters
        ----------
        dropout: dropout probability for the function applied after the second convolutional layer
        act_func

        """
        super(ConvolutionalNet,self).__init__()
        
        # first layer settings
        self.kernel_size0 = 2
        self.padding0 = 1
        self.stride0 = 2
        self.dilation0 = 1

        # second layer settings
        self.kernel_size1 = 3
        self.padding1 = 1
        self.stride1 = 2
        self.dilation1 = 1

        # linear layer settings        
        self.linear_size = linear_size
        self.conv0_size = conv0_size
        self.conv1_size = conv1_size     
        self.act_func = act_func
        self.final_func = nn.Softmax(dim=1)
        self.N_out= 10

        self.dropout = dropout
        self.dropout_func = nn.Dropout(self.dropout)

        # network creations
        self.conv0 = nn.Conv2d(1,self.conv0_size,kernel_size=self.kernel_size0,padding=self.padding0,stride=self.stride0,dilation=self.dilation0) #(1,28,28) -> (self.conv0_size,15,15)
        
        self.conv1 = nn.Conv2d(self.conv0_size,self.conv1_size,kernel_size=self.kernel_size1,padding=self.padding1,stride=self.stride1,dilation=self.dilation1) #(self.conv0_size,15,15) -> (self.conv1_size,8,8)

        self.lin_layer = nn.Linear(in_features=self.conv1_size*8*8, out_features=self.linear_size)          
        self.out_layer = nn.Linear(in_features=self.linear_size, out_features=self.N_out)
        
    def forward(self, x):
        
        # first convolutional layer
        x = self.conv0(x)
        x = self.act_func(x)

        # second convolutional layer  
        x = self.conv1(x)
        x = self.act_func(x)

        # dropout function
        x = self.dropout_func(x)

        # linear layer
        x = x.view(-1,self.conv1_size*8*8)
        x = self.lin_layer(x)
        x = self.act_func(x)

        # output layer
        x = self.out_layer(x)
        
        x = self.final_func(x)
        return x