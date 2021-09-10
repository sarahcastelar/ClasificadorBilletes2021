import numpy as np
import random
import torch
from torch.utils.data import Dataset
from os.path import dirname
from PIL import Image
import torchvision.transforms.functional as TF


class IndoorDataset(Dataset):

    def __init__(self, root_dir, img_files, training, size, flip_chance=0.5, color_change_chance=0.10, gaussian_noise_chance=0.2, gaussian_noise_range=5.0, luminosity_changes_chance=0.125, transform=None):
        self.root_dir = root_dir
        self.img_files = img_files
        self.transform = transform
        self.training = training
        self.size = size
        self.flip_chance = flip_chance
        self.color_change_chance = color_change_chance
        self.gaussian_noise_chance = gaussian_noise_chance
        self.luminosity_changes_chance = luminosity_changes_chance
        self.gaussian_noise_range = gaussian_noise_range
        self.classes = ['airport_inside', 'artstudio', 'auditorium', 'bakery', 'bar', 'bathroom', 'bedroom', 'bookstore', 'bowling', 'buffet', 'casino', 'children_room',
                        'church_inside', 'classroom', 'cloister', 'closet', 'clothingstore', 'computerroom', 'concert_hall', 'corridor', 'deli', 'dentaloffice', 'dining_room', 'elevator', 'fastfood_restaurant', 'florist', 'gameroom', 'garage', 'greenhouse', 'grocerystore', 'gym', 'hairsalon', 'hospitalroom', 'inside_bus', 'inside_subway', 'jewelleryshop', 'kindergarden', 'kitchen', 'laboratorywet', 'laundromat', 'library', 'livingroom', 'lobby', 'locker_room',
                        'mall', 'meeting_room', 'movietheater', 'museum', 'nursery', 'office', 'operating_room', 'pantry', 'poolinside', 'prisoncell', 'restaurant', 'restaurant_kitchen', 'shoeshop', 'stairscase', 'studiomusic', 'subway', 'toystore', 'trainstation', 'tv_studio', 'videostore', 'waitingroom', 'warehouse', 'winecellar']

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
        pil_img = Image.open(
            f"{self.root_dir}/{image_filename}", mode='r',)
        pil_img = pil_img.convert('RGB')
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
        if self.training:
            final_class = dirname(image_filename)
            return img_tensor, torch.tensor(self.classes.index(final_class))
        else:
            return img_tensor, image_filename
