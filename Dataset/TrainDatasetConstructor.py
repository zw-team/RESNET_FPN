from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.utils.data as data
import random
import time
from utils import GroundTruthProcess, HSI_Calculator
import scipy.io as scio
import h5py
import math
pers_dir_path = "/home/zzn/Documents/Datasets/part_A_final/train_data/perspective_gt"

class TrainDatasetConstructor(data.Dataset):
    def __init__(self,
                 data_dir_path,
                 gt_dir_path,
                 train_num,
                 mode='whole',
                 stage='shape',
                 if_random_hsi=False,
                 if_flip=False
                 ):
        self.train_num = train_num
        self.imgs = []
        self.data_root = data_dir_path
        self.gt_root = gt_dir_path
        self.permulation = np.random.permutation(self.train_num)
        self.calcu = HSI_Calculator()
        self.mode = mode
        self.stage = stage
        self.if_random_hsi = if_random_hsi
        self.if_flip = if_flip
        self.GroundTruthProcess = GroundTruthProcess(1, 1, 4).cuda()
        count = 0
        for i in range(self.train_num):
            img_name = '/IMG_' + str(i + 1) + ".jpg"
            gt_map_name = '/GT_IMG_' + str(i + 1) + ".npy"
            
            perspective_map_name = '/IMG_' + str(i + 1) + ".mat"
            p = h5py.File(pers_dir_path + perspective_map_name)['pmap'][:][0][0]
            
            img = Image.open(self.data_root + img_name).convert("RGB")
            height = img.size[1]
            width = img.size[0]
            resize_height = height
            resize_width = width

            if resize_height <= 416:
                tmp = resize_height
                resize_height = 416
                resize_width = (resize_height / tmp) * resize_width

            if resize_width <= 416:
                tmp = resize_width
                resize_width = 416
                resize_height = (resize_width / tmp) * resize_height

            resize_height = math.ceil(resize_height / 32) * 32
            resize_width = math.ceil(resize_width / 32) * 32
            img = transforms.Resize([resize_height, resize_width])(img)
            gt_map = Image.fromarray(np.squeeze(np.load(self.gt_root + gt_map_name)))
            if True:
                self.imgs.append([img, gt_map, i+1])
                count += 1
        self.train_num = count

    def __getitem__(self, index):
        if self.mode == 'crop':
            img, gt_map, map_index = self.imgs[self.permulation[index]]
            if self.if_random_hsi:
                img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(img)
            if self.if_flip:
                flip_random = random.random()
                if flip_random > 0.5:
                    img = F.hflip(img)
                    gt_map = F.hflip(gt_map)
                    
            img = transforms.ToTensor()(img)
            gt_map = transforms.ToTensor()(gt_map)
            img_shape = img.shape  # C, H, W
            random_h = random.randint(0, img_shape[1] - 400)
            random_w = random.randint(0, img_shape[2] - 400)
            patch_height = 400
            patch_width = 400
            img = img[:, random_h:random_h + patch_height, random_w:random_w + patch_width].cuda()
            gt_map = gt_map[:, random_h:random_h + patch_height, random_w:random_w + patch_width].cuda()
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            gt_map = self.GroundTruthProcess(gt_map.view(1, 1, 400, 400))
            return map_index, img.view(3, 400, 400), gt_map.view(1, 200, 200)
          
        else:
            img, gt_map, map_index = self.imgs[self.permulation[index]]
            height = img.size[1]
            width = img.size[0]
            img = transforms.Resize([height * 2, width * 2])(img)
            if self.if_random_hsi:
                img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(img)
            if self.if_flip:
                flip_random = random.random()
                if flip_random > 0.5:
                    img = F.hflip(img)
                    gt_map = F.hflip(gt_map)
            img = transforms.ToTensor()(img).cuda()
            gt_map = transforms.ToTensor()(gt_map).cuda()
            img_shape = img.shape  # C, H, W
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            gt_map = self.GroundTruthProcess(gt_map.view(1, 1, img_shape[1] // 2, img_shape[2] // 2))
            return map_index, img, gt_map.view(1, img_shape[1] // 8, img_shape[2] // 8)

    def __len__(self):
        return self.train_num

    def shuffle(self):
        self.permulation = np.random.permutation(self.train_num)
        return self