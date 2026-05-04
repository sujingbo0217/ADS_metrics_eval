"""
This file creates commonly used augmented photo data for machine learning models.
Techniques include:
1. Gaussian Blur
2. Gaussian Noise
3. Color Jitter
4. Brightness
5. Contrast

Arguments:
file-path - This should path to where the real images are and where you want to save them.
"""

import torchvision.transforms as T
import torch
import os
from PIL import Image
from functools import partial
import argparse

def augment_images(input_image_path, output_image_path, image_name, func):
    """
    Code to implement an augmentation function(func) in a directory on the input_image_path and 
    output it to the output_image_path directory.
    
    Input:
    input_image_path - str - Path to directory of real images
    output_image_path - str - Path to directory where transformed real images will be stored
    image_name - str - specific name for images you would like to add before their original name
    func - function - the augmentation function you would like to transform all real images
    """
    for filename in os.listdir(input_image_path):

        img_path = os.path.join(input_image_path, filename)
        img = Image.open(img_path).convert("RGB")

        aug_img = func(img)

        save_path = os.path.join(output_image_path, f"{image_name}_{filename}")
        aug_img.save(save_path)

def add_gaussian_noise(img, mean, std):
    """
    Adds random gaussian noise to an image.
    """
    to_tensor = T.ToTensor()
    tensor = to_tensor(img)
    noise = torch.randn_like(tensor) * std + mean
    noisy = tensor + noise
    to_pil = T.ToPILImage()
    img_out = to_pil(noisy)
    return img_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    args = parser.parse_args()
    input_image_path = f"{args.file_path}/kitti_real_images/"

    #Gaussian Noise
    output_image_path = f"{args.file_path}/gaussian_noise_level_3/"
    noise_func = partial(add_gaussian_noise, mean=0, std=0.08)
    augment_images(input_image_path, output_image_path, "gn_level_3", noise_func)

    output_image_path = f"{args.file_path}/gaussian_noise_level_2/"
    noise_func = partial(add_gaussian_noise, mean=0, std=0.04)
    augment_images(input_image_path, output_image_path, "gn_level_2", noise_func)

    output_image_path = f"{args.file_path}/gaussian_noise_level_1/"
    noise_func = partial(add_gaussian_noise, mean=0, std=0.02)
    augment_images(input_image_path, output_image_path, "gn_level_1", noise_func)
    
    #Gaussian Blur
    output_image_path = f"{args.file_path}/gaussian_blur_level_3/"
    blur_func = T.GaussianBlur(kernel_size=11, sigma=8)
    augment_images(input_image_path, output_image_path, "gb_level_3", blur_func)

    output_image_path = f"{args.file_path}/gaussian_blur_level_2/"
    blur_func = T.GaussianBlur(kernel_size=11, sigma=4)
    augment_images(input_image_path, output_image_path, "gb_level_2", blur_func)

    output_image_path = f"{args.file_path}/gaussian_blur_level_1/"
    blur_func = T.GaussianBlur(kernel_size=11, sigma=2)
    augment_images(input_image_path, output_image_path, "gb_level_1", blur_func)
    
    #ColorJitter
    output_image_path = f"{args.file_path}/color_jitter_level_3/"
    cj_func = T.ColorJitter(hue=0.5)
    augment_images(input_image_path, output_image_path, "cj_level_3", cj_func)

    output_image_path = f"{args.file_path}/color_jitter_level_2/"
    cj_func = T.ColorJitter(hue=0.3)
    augment_images(input_image_path, output_image_path, "cj_level_2", cj_func)

    output_image_path = f"{args.file_path}/color_jitter_level_1/"
    cj_func = T.ColorJitter(hue=0.1)
    augment_images(input_image_path, output_image_path, "cj_level_1", cj_func)
    
    #Brightness
    output_image_path = f"{args.file_path}/bright_level_3/"
    bright_func = T.ColorJitter(brightness=8)
    augment_images(input_image_path, output_image_path, "bright_level_3", bright_func)

    output_image_path = f"{args.file_path}/bright_level_2/"
    bright_func = T.ColorJitter(brightness=4)
    augment_images(input_image_path, output_image_path, "bright_level_2", bright_func)

    output_image_path = f"{args.file_path}/bright_level_1/"
    bright_func = T.ColorJitter(brightness=2)
    augment_images(input_image_path, output_image_path, "bright_level_1", bright_func)
    
    #Contrast
    output_image_path = f"{args.file_path}/contrast_level_3/"
    contrast_func = T.ColorJitter(contrast=1)
    augment_images(input_image_path, output_image_path, "contrast_level_3", contrast_func)

    output_image_path = f"{args.file_path}/contrast_level_2/"
    contrast_func = T.ColorJitter(contrast=0.5)
    augment_images(input_image_path, output_image_path, "contrast_level_2", contrast_func)

    output_image_path = f"{args.file_path}/contrast_level_1/"
    contrast_func = T.ColorJitter(contrast=0.2)
    augment_images(input_image_path, output_image_path, "contrast_level_1", contrast_func)
    

