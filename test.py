
import torch
from dataset import imageDataset
from utils import load_checkpoint_for_testing
from torch.utils.data import DataLoader
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def test_fn(
     gen_Z, gen_H, loader, 
):
    loop = tqdm(loader, leave=True)

    for idx, (expertc, original) in enumerate(loop):
        expertc = expertc.to(config.DEVICE)
        original = original.to(config.DEVICE)

        with torch.amp.autocast('cuda'):  
            fake_original = gen_H(expertc)
            fake_expertc = gen_Z(original)

        
        save_image(fake_original* 0.5 + 0.5, f"test_results/expertc_to_original/original_{idx}.png")
        save_image(fake_expertc * 0.5 + 0.5, f"test_results/original_to_expertc/expertc_{idx}.png")


def main():

    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    
    
    load_checkpoint_for_testing(
            config.CHECKPOINT_GEN_H,
            gen_H,
        )
    load_checkpoint_for_testing(
            config.CHECKPOINT_GEN_Z,
            gen_Z,
        )

    test_dataset = imageDataset(
        root_original=config.TEST_DIR+"/original",
        root_expertc=config.TEST_DIR+"/expertc",
        transform=config.transforms,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    test_fn(
            gen_Z,
            gen_H,
            test_loader,
        )

if __name__ == "__main__":
    main()
    print("Test completed successfully!")
