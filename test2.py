
import torch
from dataset2 import imageDataset
from utils import load_checkpoint_for_testing
from torch.utils.data import DataLoader
import config
from tqdm import tqdm
from torchvision.utils import save_image
from generator_model import Generator


def test_fn(
     gen_Z, loader, 
):
    loop = tqdm(loader, leave=True)

    for idx, ( original) in enumerate(loop):
        
        original = original.to(config.DEVICE)

        with torch.amp.autocast('cuda'):  
            
            fake_expertc = gen_Z(original)

      
        save_image(fake_expertc * 0.5 + 0.5, f"test_results/original_to_expertc/expertc_{idx}.png")


def main():

    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
   
    
  
    load_checkpoint_for_testing(
            config.CHECKPOINT_GEN_Z,
            gen_Z,
        )

    test_dataset = imageDataset(
        root_original=config.TEST_DIR+"/original",
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
            test_loader,
        )

if __name__ == "__main__":
    main()
    print("Test completed successfully!")
