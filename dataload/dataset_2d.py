import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import pywt
import random

class dataset_XNetv2(Dataset):
    def __init__(self, data_dir, augmentation_1, normalize_1, wavelet_type, alpha, beta, sup=True, num_images=None, **kwargs):
        super(dataset_XNetv2, self).__init__()

        img_paths_1 = []
        mask_paths = []

        image_dir_1 = data_dir + '/image'
        if sup:
            mask_dir = data_dir + '/mask'

        for image in os.listdir(image_dir_1):

            image_path_1 = os.path.join(image_dir_1, image)
            img_paths_1.append(image_path_1)

            if sup:
                mask_path = os.path.join(mask_dir, image)
                mask_paths.append(mask_path)

        if sup:
            assert len(img_paths_1) == len(mask_paths)

        if num_images is not None:
            len_img_paths = len(img_paths_1)
            quotient = num_images // len_img_paths
            remainder = num_images % len_img_paths

            if num_images <= len_img_paths:
                img_paths_1 = img_paths_1[:num_images]
            else:
                rand_indices = torch.randperm(len_img_paths).tolist()
                new_indices = rand_indices[:remainder]

                img_paths_1 = img_paths_1 * quotient
                img_paths_1 += [img_paths_1[i] for i in new_indices]

                if sup:
                    mask_paths = mask_paths * quotient
                    mask_paths += [mask_paths[i] for i in new_indices]

        self.img_paths_1 = img_paths_1
        self.mask_paths = mask_paths
        self.augmentation_1 = augmentation_1
        self.normalize_1 = normalize_1
        self.sup = sup
        self.kwargs = kwargs
        self.alpha = alpha
        self.beta = beta
        self.wavelet_type = wavelet_type

    def __getitem__(self, index):

        img_path_1 = self.img_paths_1[index]
        img_1 = Image.open(img_path_1)
        img_1 = np.array(img_1)

        LL, (LH, HL, HH) = pywt.dwt2(img_1, self.wavelet_type, axes=(0, 1))

        LL = (LL - np.amin(LL, (0, 1))) / (np.amax(LL, (0, 1)) - np.amin(LL, (0, 1))) * 255
        LH = (LH - np.amin(LH, (0, 1))) / (np.amax(LH, (0, 1)) - np.amin(LH, (0, 1))) * 255
        HL = (HL - np.amin(HL, (0, 1))) / (np.amax(HL, (0, 1)) - np.amin(HL, (0, 1))) * 255
        HH = (HH - np.amin(HH, (0, 1))) / (np.amax(HH, (0, 1)) - np.amin(HH, (0, 1))) * 255

        H_ = HL + LH + HH
        H_ = (H_ - np.amin(H_, (0, 1))) / (np.amax(H_, (0, 1)) - np.amin(H_, (0, 1))) * 255

        L_alpha = random.uniform(self.alpha[0], self.alpha[1])
        L = LL + L_alpha * H_
        L = (L - np.amin(L, (0, 1))) / (np.amax(L, (0, 1)) - np.amin(L, (0, 1))) * 255

        H_beta = random.uniform(self.beta[0], self.beta[1])
        H = H_ + H_beta * LL
        H = (H - np.amin(H, (0, 1))) / (np.amax(H, (0, 1)) - np.amin(H, (0, 1))) * 255

        if self.sup:
            mask_path = self.mask_paths[index]
            mask = Image.open(mask_path)
            mask = np.array(mask)

            augment_1 = self.augmentation_1(image=img_1, mask=mask, L=L, H=H)
            img_1 = augment_1['image']
            L = augment_1['L']
            H = augment_1['H']
            mask_1 = augment_1['mask']

            normalize_1 = self.normalize_1(image=img_1, mask=mask_1, L=L, H=H)
            img_1 = normalize_1['image']
            L = normalize_1['L']
            H = normalize_1['H']
            mask_1 = normalize_1['mask'].long()

            sampel = {'image': img_1, 'mask': mask_1, 'L': L, 'H': H, 'ID': os.path.split(mask_path)[1]}

        else:
            augment_1 = self.augmentation_1(image=img_1, L=L, H=H)
            img_1 = augment_1['image']
            L = augment_1['L']
            H = augment_1['H']

            normalize_1 = self.normalize_1(image=img_1, L=L, H=H)
            img_1 = normalize_1['image']
            L = normalize_1['L']
            H = normalize_1['H']

            sampel = {'image': img_1, 'L': L, 'H': H, 'ID': os.path.split(img_path_1)[1]}

        return sampel

    def __len__(self):
        return len(self.img_paths_1)


def imagefloder_XNetv2(data_dir, data_transform_1, data_normalize_1, wavelet_type, alpha, beta, sup=True, num_images=None, **kwargs):
    dataset = dataset_XNetv2(data_dir=data_dir,
                          augmentation_1=data_transform_1,
                          normalize_1=data_normalize_1,
                          wavelet_type=wavelet_type,
                          alpha=alpha,
                          beta=beta,
                          sup=sup,
                          num_images=num_images,
                           **kwargs)
    return dataset
