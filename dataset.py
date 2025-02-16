from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class imageDataset(Dataset):
    def __init__(self, root_expertc, root_original, transform=None):
        self.root_expertc = root_expertc
        self.root_original = root_original
        self.transform = transform

        self.expertc_images = os.listdir(root_expertc)
        self.original_images = os.listdir(root_original)
        self.length_dataset = max(len(self.expertc_images), len(self.original_images)) # 1500, 1500
        self.expertc_len = len(self.expertc_images)
        self.original_len = len(self.original_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        expertc_img = self.expertc_images[index % self.expertc_len]
        original_img = self.original_images[index % self.original_len]

        expertc_path = os.path.join(self.root_expertc, expertc_img)
        original_path = os.path.join(self.root_original, original_img)

        expertc_img = np.array(Image.open(expertc_path).convert("RGB"))
        original_img = np.array(Image.open(original_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=expertc_img, image0=original_img)
            expertc_img = augmentations["image"]
            original_img = augmentations["image0"]

        return expertc_img, original_img





