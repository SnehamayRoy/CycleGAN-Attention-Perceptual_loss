
import torch
from dataset import imageDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
from perceptual_loss import PerceptualLoss  # Ensure this file contains the VGG19-based loss

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/cycleGan')
from torchvision import models


def train_fn(
    disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse,perceptual_loss, d_scaler, g_scaler,epoch
):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (expertc, original) in enumerate(loop):
        expertc = expertc.to(config.DEVICE)
        original = original.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.amp.autocast('cuda'):  
            fake_original = gen_H(expertc)
            D_H_real = disc_H(original)
            D_H_fake = disc_H(fake_original.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_expertc = gen_Z(original)
            D_Z_real = disc_Z(expertc)
            D_Z_fake = disc_Z(fake_expertc.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.amp.autocast('cuda'):  
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_original)
            D_Z_fake = disc_Z(fake_expertc)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_expertc= gen_Z(fake_original)
            cycle_original = gen_H(fake_expertc)
            cycle_expertc_loss = l1(expertc, cycle_expertc)
            cycle_original_loss = l1(original, cycle_original)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_expertc = gen_Z(expertc)
            identity_original= gen_H(original)
            identity_expertc_loss = l1(expertc, identity_expertc)
            identity_original_loss = l1(original, identity_original)
            
            perceptual_loss_expertc = perceptual_loss(fake_expertc, expertc)
            perceptual_loss_original = perceptual_loss(fake_original, original)
            

            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_expertc_loss * config.LAMBDA_CYCLE
                + cycle_original_loss * config.LAMBDA_CYCLE
                + identity_original_loss * config.LAMBDA_IDENTITY
                + identity_expertc_loss * config.LAMBDA_IDENTITY
                + perceptual_loss_expertc * config.LAMBDA_PERCEPTUAL
                + perceptual_loss_original * config.LAMBDA_PERCEPTUAL
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_original* 0.5 + 0.5, f"saved_images/original_{idx}.png")
            save_image(fake_expertc * 0.5 + 0.5, f"saved_images/expertc_{idx}.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))
        
    writer.add_scalar('Generator H loss', loss_G_H, epoch+1)
    writer.add_scalar('Generator z loss', loss_G_Z, epoch+1)
    writer.add_scalar('cycle_expertc_loss', cycle_expertc_loss, epoch+1) 
    writer.add_scalar('cycle_original_loss', cycle_original_loss, epoch+1)
    writer.add_scalar('identity_expertc_loss', identity_expertc_loss, epoch+1) 
    writer.add_scalar('identity_original_loss', identity_original_loss, epoch+1)
    writer.add_scalar('Perceptual Loss ExpertC', perceptual_loss_expertc, epoch + 1)
    writer.add_scalar('Perceptual Loss Original', perceptual_loss_original, epoch + 1)
    writer.add_scalar('Total loss', G_loss, epoch + 1)
def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    perceptual_loss = PerceptualLoss().to(config.DEVICE)
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H,
            gen_H,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z,
            gen_Z,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H,
            disc_H,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z,
            disc_Z,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = imageDataset(
        root_original=config.TRAIN_DIR + "/original",
        root_expertc=config.TRAIN_DIR + "/expertc",
        transform=config.transforms,
    )
    val_dataset = imageDataset(
        root_original=config.TRAIN_DIR+"/original",
        root_expertc=config.TRAIN_DIR+"/expertc",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler =  torch.amp.GradScaler('cuda')
    d_scaler =  torch.amp.GradScaler('cuda')

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_H,
            disc_Z,
            gen_Z,
            gen_H,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            perceptual_loss,
            d_scaler,
            g_scaler,
            epoch,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)
    writer.close()

if __name__ == "__main__":
    main()
