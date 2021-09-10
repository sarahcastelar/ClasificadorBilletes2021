import numpy as np
import random
import torch
import cv2

from torch.utils.data import Dataset
from os.path import dirname
from PIL import Image
import torchvision.transforms.functional as TF


class IndoorDataset(Dataset):

    def __init__(self, root_dir, img_files, training, size, max_padding, flip_chance=0.5, color_change_chance=0.10, gaussian_noise_chance=0.2, gaussian_noise_range=5.0, luminosity_changes_chance=0.125, transform=None):
        self.root_dir = root_dir
        self.img_files = img_files
        self.training = training
        if self.training:
            classes = []
            for img in self.img_files:
                dir_name = dirname(img)
                classes.append(dir_name)
            self.classes = sorted(list(set(classes)))
        self.transform = transform
        self.size = size
        self.max_padding = max_padding
        self.flip_chance = flip_chance
        self.color_change_chance = color_change_chance
        self.gaussian_noise_chance = gaussian_noise_chance
        self.luminosity_changes_chance = luminosity_changes_chance
        self.gaussian_noise_range = gaussian_noise_range

    def safe_cropping(self, pil_img):
        img_width, img_height = pil_img.size
        target_width, target_height = self.size

        scale_w = target_width/img_width
        scale_h = target_height/img_height

        # usar menor escala
        if scale_h >= scale_w:
            # usa escala horizontal ..
            # calcular padding proporcional
            new_h = int(img_height * scale_w)

            prop_padding = (target_height - new_h) / target_height

            # verificar ...
            if prop_padding > self.max_padding:
                # la imagen requiere mas padding del permitido
                # recortar Width (incrementara el valor de scale_w)

                # primero calculamos cuanto deberia ser la altura de la imagen
                # redimensionada antes de agregar MAX padding vertical
                # ... maxima altura a ocupar con nueva escala ...
                target_prepadded_h = target_height * (1 - self.max_padding)
                # ... escala inversa para dicha altura ...
                target_inv_scale_h = img_height / target_prepadded_h
                # ... determinar el maximo ancho que produce target_w
                # ... usando la escala objetivo que produce max_padding en H
                max_source_w = int(round(target_width * target_inv_scale_h))

                # ... sera necesario recortar la imagen original
                # ... remueve parte del ancho, preserva largo
                center_crop = (img_height, max_source_w)
            else:
                # no se necesita un corte
                center_crop = None

        else:
            new_w = int(img_width * scale_h)

            prop_padding = (target_width - new_w) / target_width

            # verificar ...
            if prop_padding > self.max_padding:
                # la imagen requiere mas padding del permitido
                # recortar Height (incrementara el valor de scale_h)

                # primero calculamos cuanto deberia ser el ancho de la imagen
                # redimensionada antes de agregar MAX padding horizontal
                # ... maximo ancho a ocupar con nueva escala ...
                target_prepadded_w = target_width * (1 - self.max_padding)
                # ... escala inversa para dicha ancho ...
                target_inv_scale_w = img_width / target_prepadded_w
                # ... determinar la maxima altura que produce target_h
                # ... usando la escala objetivo que produce max_padding en W
                max_source_h = int(round(target_height * target_inv_scale_w))

                # ... sera necesario recortar la imagen original
                # ... remueve parte del largo, preserva ancho
                center_crop = (max_source_h, img_width)
            else:
                # no se necesita un corte
                center_crop = None

        if center_crop is not None:
            # apply center cropping ...
            # debug_img = np.asarray(pil_img)
            # debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
            # cv2.imshow("Pre cut Image", debug_img)

            pil_img = TF.center_crop(pil_img, center_crop)

            # debug_img = np.asarray(pil_img)
            # debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
            # cv2.imshow("Post cut Image", debug_img)
            # cv2.waitKey()

        return pil_img

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

        if self.max_padding is not None:
            pil_img = self.safe_cropping(pil_img)

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
