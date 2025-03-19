import argparse
import os
from utils import get_args
import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model
from jutils.logger import get_writer, get_id_cur_step
def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO 1.3: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    # why doesn't the discrim have a sigmoid layer at the end?
    """
    discrim_real = F.sigmoid(discrim_real)
    discrim_fake = F.sigmoid(discrim_fake)
    get_writer("runs/Discriminator_histogram_1", with_id="histogram").add_histogram("Discriminator output", 
                                                                                  discrim_fake, 
                                                                                  get_id_cur_step("histogram"))
    loss = torch.mean(-torch.log(discrim_real) - torch.log(1 - discrim_fake))
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    real_labels = torch.ones(discrim_real.shape[0], 1).cuda()
    fake_labels = torch.zeros(discrim_fake.shape[0], 1).cuda()
    loss = criterion(discrim_real, real_labels) + criterion(discrim_fake, fake_labels)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.3: Implement GAN loss for the generator.
    ##################################################################
    # discrim_fake = F.sigmoid(discrim_fake)
    # loss = torch.mean(-torch.log(discrim_fake))
    # get_writer("runs/generator/loss", with_id="gen").add_scalar("Gen loss",
    #                                                             loss,
    #                                                             get_id_cur_step("gen"))
    criterion = torch.nn.BCEWithLogitsLoss()
    labels = torch.ones(discrim_fake.shape[0], 1).cuda()
    loss = criterion(discrim_fake, labels)

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)


    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
