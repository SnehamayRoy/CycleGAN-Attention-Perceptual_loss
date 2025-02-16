import cv2
import numpy as np
import os
import re

def natural_sort_key(s):
    # Use a raw string to avoid issues with backslashes in the regex
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def calculate_psnr(original, enhanced):
    original, enhanced = original.astype(np.float64), enhanced.astype(np.float64)
    mse = np.mean((original - enhanced) ** 2,dtype=np.float64)
    if mse == 0:
        return float('inf')  # Means no difference between the images
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def calculate_mean_psnr(original_dir, enhanced_dir):
    psnr_values = []

    # Get sorted lists of filenames from the directories
    
    original_filenames = sorted(os.listdir(original_dir))
    enhanced_filenames = sorted(os.listdir(enhanced_dir), key=natural_sort_key)

    # Ensure both directories have the same number of images
    if len(original_filenames) != len(enhanced_filenames):
        print("The number of images in the directories does not match.")
        return None

    # Iterate over pairs of images based on sorted order
    for i in range(len(original_filenames)):
        
        original_file = original_filenames[i]
        enhanced_file = enhanced_filenames[i]
        # Load the original and enhanced images
        original_path = os.path.join(original_dir, original_file)
        enhanced_path = os.path.join(enhanced_dir, enhanced_file)

        original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        enhanced = cv2.imread(enhanced_path, cv2.IMREAD_GRAYSCALE)

        # Check if both images were loaded correctly
        if original is None or enhanced is None:
            print(f"Error loading images for {original_file} or {enhanced_file}. Skipping this pair.")
            continue

        # Calculate PSNR and append it to the list
        psnr = calculate_psnr(original, enhanced)
        psnr_values.append(psnr)

        print(f'PSNR for {original_file} and {enhanced_file}: {psnr} dB')

    # Calculate the mean PSNR
    if len(psnr_values) == 0:
        return 0  # To handle division by zero if no valid pairs
    mean_psnr = np.mean(psnr_values)
    return mean_psnr

# Define paths to the directories containing the original and enhanced images
original_dir = 'Original/'
enhanced_dir = 'Enhanced/'

# Calculate the mean PSNR for all image pairs
mean_psnr = calculate_mean_psnr(original_dir, enhanced_dir)
print(f'Mean PSNR: {mean_psnr} dB')
