import torchvision.transforms as T
import torch
import os
from PIL import Image

#Gaussian Noise 1, 4, 8
#Color Jitter 0.1, 0.3, 0.5

input_image_path = "/scratch/sim-real/kitti_real_images/"
output_image_path = "/scratch/sim-real/gaussian_noise_level_3/"

def add_gaussian_noise(img, mean=0., std=0.1):
    to_tensor = T.ToTensor()
    tensor = to_tensor(img)
    noise = torch.randn_like(tensor) * std + mean
    noisy = tensor + noise
    to_pil = T.ToPILImage()
    img_out = to_pil(noisy)
    return img_out


for filename in os.listdir(input_image_path):

    img_path = os.path.join(input_image_path, filename)
    img = Image.open(img_path).convert("RGB")

    aug_img = add_gaussian_noise(img, 0, 0.08)

    save_path = os.path.join(output_image_path, f"gn_level_3_{filename}")
    aug_img.save(save_path)


input_image_path = "/scratch/sim-real/kitti_real_images/"
output_image_path = "/scratch/sim-real/gaussian_noise_level_2/"

for filename in os.listdir(input_image_path):

    img_path = os.path.join(input_image_path, filename)
    img = Image.open(img_path).convert("RGB")

    aug_img = add_gaussian_noise(img, 0, 0.04)

    save_path = os.path.join(output_image_path, f"gn_level_2_{filename}")
    aug_img.save(save_path)


input_image_path = "/scratch/sim-real/kitti_real_images/"
output_image_path = "/scratch/sim-real/gaussian_noise_level_1/"

for filename in os.listdir(input_image_path):

    img_path = os.path.join(input_image_path, filename)
    img = Image.open(img_path).convert("RGB")

    aug_img = add_gaussian_noise(img, 0, 0.01)

    save_path = os.path.join(output_image_path, f"gn_level_1_{filename}")
    aug_img.save(save_path)


