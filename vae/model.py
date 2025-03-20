import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import jutils.nn_utils as mynn
from jutils.logger import *
import jutils.config_classes as cf
class Encoder(nn.Module):
    """
    Sequential(
        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (3): ReLU()
        (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (5): ReLU()
        (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    """
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        C,H,W = input_shape
        self.latent_dim = latent_dim
        ##################################################################
        # TODO 2.1: Set up the network layers. First create the self.convs.
        # Then create self.fc with output dimension == self.latent_dim
        ##################################################################
        config = cf.EncoderConfig(shape=input_shape,
                                  fatten_first=True,
                                  fat=32,
                                  spatial_reduce_factor=2
                                  )
        self.convs = mynn.create_encoder([64, 128, 256], config=config)
        self.fc = nn.Linear(256 * H//8 * W//8, self.latent_dim)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, x):
        ##################################################################
        # TODO 2.1: Forward pass through the network, output should be
        # of dimension == self.latent_dim
        ##################################################################
        B = x.shape[0]
        after_conv = self.convs(x).view(B, -1)
        after_fc = self.fc(after_conv)
        return after_fc
        pass
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        ##################################################################
        # TODO 2.4: Fill in self.fc, such that output dimension is
        # 2*self.latent_dim
        ##################################################################
        input_dim_flat = input_shape[0] * input_shape[1] * input_shape[2]
        self.fc = nn.Linear(input_dim_flat, latent_dim * 2)
        self.relu = nn.ReLU()
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, x):
        ##################################################################
        # TODO 2.1: Forward pass through the network, should return a
        # tuple of 2 tensors, mu and log_std
        ##################################################################
        B = x.shape[0]
        x = x.view(B, -1)
        after_fc = self.fc(x)
        after_rel = self.relu(after_fc) 
        mu = after_fc[:,:self.latent_dim]
        log_std = after_fc[:, self.latent_dim:]
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        return mu, log_std


class Decoder(nn.Module):
    """
    Sequential(
        (0): ReLU()
        (1): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (2): ReLU()
        (3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (4): ReLU()
        (5): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (6): ReLU()
        (7): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    """
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        ##################################################################
        # TODO 2.1: Set up the network layers. First, compute
        # self.base_size, then create the self.fc and self.deconvs.
        ##################################################################

        C,W,H = output_shape
        self.base_size = 256 * (H//8) * (W//8)
        self.deconvs = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, C, kernel_size=3, stride=1, padding=1),
        )
        self.fc = nn.Linear(latent_dim, self.base_size)

        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, z):
        #TODO 2.1: forward pass through the network, 
        ##################################################################
        # TODO 2.1: Forward pass through the network, first through
        # self.fc, then self.deconvs.
        C,W,H = self.output_shape
        after_fc = self.fc(z).reshape(-1, 256, H//8, W//8)
        after_decon = self.deconvs(after_fc)
        return after_decon
        ##################################################################
        pass
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape = (3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)
        mynn.xavier_init(self.encoder)
        mynn.xavier_init(self.decoder)
    # NOTE: You don't need to implement a forward function for AEModel.
    # For implementing the loss functions in train.py, call model.encoder
    # and model.decoder directly.
