
from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class imageDataset(Dataset):
    def __init__(self,  root_original, transform=None):
        
        self.root_original = root_original
        self.transform = transform

        # Sort the images by their file names to ensure they are loaded in order
      
        self.original_images = sorted(os.listdir(root_original))

        # Get the length of the dataset based on the larger set of images
        self.length_dataset = len(self.original_images)
  
        self.original_len = len(self.original_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
      
        original_img = self.original_images[index % self.original_len]

   
        original_path = os.path.join(self.root_original, original_img)

       
        original_img = np.array(Image.open(original_path).convert("RGB"))

        if self.transform:
            # Transform the original image only
            augmentations = self.transform(image=original_img)
            original_img = augmentations["image"]

        return original_img





