import cv2
import numpy as np
import os
import re
def natural_sort_key(s):
    # Use a raw string to avoid issues with backslashes in the regex
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')



def calculate_mean_ssim(original_dir, enhanced_dir):
    ssim_values = []

    # Get sorted lists of filenames from the directories
    original_filenames = sorted(os.listdir(original_dir))
    enhanced_filenames = sorted(os.listdir(enhanced_dir), key=natural_sort_key)
    # Ensure both directories have the same number of images
    if len(original_filenames) != len(enhanced_filenames):
        print("The number of images in the directories does not match.")
        return None

    # Iterate over pairs of images based on sorted order
    for original_file, enhanced_file in zip(original_filenames, enhanced_filenames):
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
        ssim = calculate_ssim(original, enhanced)
        ssim_values.append(ssim)

        print(f'SSIM for {original_file} and {enhanced_file}: {ssim}')

    # Calculate the mean PSNR
    if len(ssim_values) == 0:
        return 0  # To handle division by zero if no valid pairs
    mean_ssim = np.mean(ssim_values)
    return mean_ssim

# Define paths to the directories containing the original and enhanced images
original_dir = 'Original/'
enhanced_dir = 'Enhanced/'

# Calculate the mean PSNR for all image pairs
mean_ssim = calculate_mean_ssim(original_dir, enhanced_dir)
print(f'Mean SSIM: {mean_ssim}')
