from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as functional
import torch.utils.data as data
import scipy.io as scio
import time
from utils import GroundTruthProcess, HSI_Calculator
import torch
import math
pers_dir_path = "/home/zzn/Documents/Datasets/part_A_final/test_data/perspective_gt"

class EvalDatasetConstructor(data.Dataset):
    def __init__(self,
                 data_dir_path,
                 gt_dir_path,
                 validate_num,
                 mode='whole',
                 stage='shape'
                 ):
        self.validate_num = validate_num
        self.imgs = []
        self.data_root = data_dir_path
        self.gt_root = gt_dir_path
        self.calcu = HSI_Calculator()
        self.mode = mode
        self.stage = stage
        self.GroundTruthProcess = GroundTruthProcess(1, 1, 4).cuda()
        count= 0
        for i in range(self.validate_num):
            img_name = '/IMG_' + str(i + 1) + ".jpg"
            gt_map_name = '/GT_IMG_' + str(i + 1) + ".npy"
            
            perspective_map_name = '/IMG_' + str(i + 1) + ".mat"
            p = scio.loadmat(pers_dir_path + perspective_map_name)['pmap'][:][0][0]
            
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
#             if p < 5:
            if True:
                self.imgs.append([img, gt_map, i + 1])
                count += 1
        self.validate_num = count

    def __getitem__(self, index):
        if self.mode == 'crop':
            img, gt_map, img_index = self.imgs[index]
            img = transforms.ToTensor()(img).cuda()
            gt_map = transforms.ToTensor()(gt_map).cuda()
            img_shape = img.shape  # C, H, W
            gt_shape = gt_map.shape
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            patch_height = (img_shape[1]) // 2
            patch_width = (img_shape[2]) // 2
            imgs = []
            for i in range(3):
                for j in range(3):
                    start_h = (patch_height // 2) * i
                    start_w = (patch_width // 2) * j
                    imgs.append(img[:, start_h:start_h + patch_height, start_w:start_w + patch_width])
            imgs = torch.stack(imgs)
            gt_map = self.GroundTruthProcess(gt_map.view(1, *(gt_shape)))
            return img_index, imgs, gt_map.view(1, gt_shape[1] // 2, gt_shape[2] // 2)
        
        else:
            img, gt_map, img_index = self.imgs[index]
            height = img.size[1]
            width = img.size[0]
            img = transforms.Resize([height * 2, width * 2])(img)
            img = transforms.ToTensor()(img).cuda()
            gt_map = transforms.ToTensor()(gt_map).cuda()
            img_shape = img.shape  # C, H, W
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            gt_map = self.GroundTruthProcess(gt_map.view(1, 1, img_shape[1] // 2, img_shape[2] // 2))
            return img_index, img, gt_map.view(1, img_shape[1] // 8, img_shape[2] // 8)

    def __len__(self):
        return self.validate_num