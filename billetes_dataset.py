import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from os import path, listdir
import cv2 as cv
from PIL import Image
from math import ceil
import torchvision.transforms.functional as TF


class ToTensor(object):

    def __call__(self, sample):
        print(sample)
        # sample = {'image': sample['image'], 'r': sample['r'], 'g': sample['g'], 'b': sample['b'],}
        return sample


class BilletesDataset(Dataset):

    def __init__(self, root_dir, etiquetas, size, flip_chance=0.5, color_change_chance=0.10, gaussian_noise_chance=0.2, gaussian_noise_range=5.0, luminosity_changes_chance=0.125, transform=None):
        self.root_dir = root_dir
        self.img_files = listdir(root_dir)
        self.transform = transform
        self.size = size
        self.flip_chance = flip_chance
        self.color_change_chance = color_change_chance
        self.gaussian_noise_chance = gaussian_noise_chance
        self.luminosity_changes_chance = luminosity_changes_chance
        self.gaussian_noise_range = gaussian_noise_range
        self.etiquetas = etiquetas
        self.classes = [
            "1-frontal",
            "1-reverso",
            "2-frontal",
            "2-reverso",
            "5-frontal",
            "5-reverso",
            "10-frontal",
            "10-reverso",
            "20-frontal",
            "20-reverso",
            "50-frontal",
            "50-reverso",
            "100-frontal",
            "100-reverso",
            "200-frontal",
            "200-reverso",
            "500-frontal",
            "500-reverso"
        ]

    def safe_resize(self, pil_img):
        img_width, img_height = pil_img.size
        target_width, target_height = self.size

        scale_w = target_width/img_width
        scale_h = target_height/img_height

        factor = 0
        if scale_h >= scale_w:
            factor = scale_w
            pil_img = pil_img.resize(
                (target_width, int(pil_img.height * factor)))
            diff = (target_height - pil_img.height)
            padding_top = diff // 2
            padding_bottom = diff - padding_top
            pil_img = self.add_padding(
                pil_img, padding_top, 0, padding_bottom, 0, (0, 0, 0))
        else:
            factor = scale_h
            pil_img = pil_img.resize(
                (int(pil_img.width * factor), target_height))
            diff = (target_width - pil_img.width)
            padding_right = diff // 2
            padding_left = diff - padding_right
            pil_img = self.add_padding(
                pil_img, 0, padding_right, 0, padding_left, (0, 0, 0))
        return pil_img

    def add_padding(self, pil_img, top, right, bottom, left, color):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        image_filename = self.img_files[idx]
        pil_img = Image.open(f"{self.root_dir}/{image_filename}")
        pil_img = self.safe_resize(pil_img)

        if self.flip_chance is not None:
            # try horizontal flipping
            if random.random() < self.flip_chance:
                pil_img = TF.hflip(pil_img)

            # try vertical flipping
            if random.random() < self.flip_chance:
                pil_img = TF.vflip(pil_img)

        if self.color_change_chance is not None and random.random() < self.color_change_chance:
            # transform color by using HUE
            pil_img = TF.adjust_hue(pil_img, (random.random() * 0.45 - 0.225))

        if self.gaussian_noise_chance is not None and random.random() < self.gaussian_noise_chance:
            # add gaussian noise
            img_np = np.asarray(pil_img).astype(np.float64)
            img_np += np.random.randn(img_np.shape[0], img_np.shape[1],
                                      img_np.shape[2]) * self.gaussian_noise_range
            img_np[img_np < 0] = 0
            img_np[img_np > 255] = 255
            pil_img = Image.fromarray(img_np.astype(np.uint8))

        if self.luminosity_changes_chance is not None and random.random() < self.luminosity_changes_chance:
            # Apply random changes that affect the luminosity and Sharpness of the image

            if np.random.randn() < 0:
                # lower brightness ... uniform ... from 0.75 to 1.0
                pil_img = TF.adjust_brightness(
                    pil_img, 1.0 - np.random.rand() * 0.25)
            else:
                # increase brightness ... uniform ... from 1.0 to 1.5
                pil_img = TF.adjust_brightness(
                    pil_img, 1.0 + np.random.rand() * 0.50)

            if np.random.randn() < 0:
                # lower contrast ... uniform ... from 0.50 to 1.0
                pil_img = TF.adjust_contrast(
                    pil_img, 1.0 - np.random.rand() * 0.5)
            else:
                # increase contrast ... uniform ... from 1.0 to 2.0
                pil_img = TF.adjust_contrast(
                    pil_img, 1.0 + np.random.rand() * 1.0)

            if np.random.randn() < 0:
                # lower gamma ... uniform ... from 0.50 to 1.0
                pil_img = TF.adjust_gamma(
                    pil_img, 1.0 - np.random.rand() * 0.50)
            else:
                # increase gamma ... uniform ... from 1.0 to 2.0
                pil_img = TF.adjust_gamma(
                    pil_img, 1.0 + np.random.rand() * 1.00)

            if np.random.randn() < 0:
                # lower the saturation ... uniform ... between 0.25 to 1.0 saturation
                pil_img = TF.adjust_saturation(
                    pil_img, 1.0 - np.random.rand() * 0.75)
            else:
                # increase the saturation ... uniform ... between 1.0 to 5.0
                pil_img = TF.adjust_saturation(
                    pil_img, 1.0 + np.random.rand() * 4.0)

        # convert to tensor
        img_tensor = TF.to_tensor(pil_img)
        if self.etiquetas != {}:
            final_class = f"{self.etiquetas[image_filename]['denominacion']}-{self.etiquetas[image_filename]['lado']}"
            return img_tensor, torch.tensor(self.classes.index(final_class))
        else:
            return img_tensor, image_filename
