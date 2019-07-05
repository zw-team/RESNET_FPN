from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as functional
import torch.utils.data as data
import random
import time
import scipy.io as scio
import h5py
import math

class TrainDatasetConstructor(data.Dataset):
    def __init__(self,
                 train_num,
                 data_dir_path,
                 gt_dir_path,
                 pers_dir_path=None,
                 mode='crop',
                 dataset_name="SHA"
                 device=None,
                 if_random_hsi=False,
                 if_flip=False
                 ):
        self.train_num = train_num
        self.imgs = []
        self.permulation = np.random.permutation(self.train_num)
        self.data_root = data_dir_path
        self.gt_root = gt_dir_path
        self.pers_root = pers_dir_path
        self.mode = mode
        self.device = device
        if self.device == None:
            raise Exception("Only support GPU version! Please choose the specific GPU device.")
        self.if_random_hsi = if_random_hsi
        self.if_flip = if_flip
        self.dataset_name = dataset_name
        self.kernel = torch.FloatTensor(torch.ones(1, 1, 2, 2)).to(self.device)
        for i in range(self.train_num):
            i_n, g_n, p_n = self.get_path_tuple(i, self.dataset_name, not(self.pers_root==None))
            self.imgs.append([self.data_root + i_n, self.gt_root + g_n, self.pers_root + p_n, i + 1])
    
    def __getitem__(self, index):
        if self.mode == 'crop':
            img_path, gt_map_path, map_index = self.imgs[self.permulation[index]]
            img = Image.open(img_path).convert("RGB")
            img = self.resize(img, self.dataset_name)
            img = transforms.ToTensor()(img).to(self.device)
            gt_map = Image.fromarray(np.squeeze(np.load(gt_map_path)))
            gt_map = transforms.ToTensor()(gt_map).to(self.device)
            if not(self.pers_root == None):
                p_m = Image.fromarray((h5py.File(pers_path)['pmap'][:] / 100).T)
            else:
                p_m = None
            if self.if_random_hsi:
                img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(img)
            if self.if_flip:
                flip_random = random.random()
                if flip_random > 0.5:
                    img = F.hflip(img)
                    gt_map = F.hflip(gt_map)
                    if not(self.pers_root == None):
                        p_m = F.hflip(p_m)
                    
            img = transforms.ToTensor()(img)
            gt_map = transforms.ToTensor()(gt_map)
            p_m = transforms.ToTensor()(p_m) if not(self.pers_root == None) else None
            img_shape = img.shape  # C, H, W
            rh, rw = random.randint(0, img_shape[1] - 400), random.randint(0, img_shape[2] - 400)
            p_h, p_w = 400, 400
            img = img[:, rh:rh + p_h, rw:rw + p_w].to(self.device)
            gt_map = gt_map[:, rh:rh + p_h, rw:rw + p_w].to(self.device)
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            gt_map = functional.conv2d(gt_map.view(1, 1, 400, 400), self.kernel, bias=None, stride=2, padding=0)
            p_m = pm[:, rh:rh + p_h, rw:rw + p_w].to(self.device) if not(self.pers_root == None) else None
            return map_index, img.view(3, 400, 400), gt_map.view(1, 200, 200), p_m

    def __len__(self):
        return self.train_num

    def shuffle(self):
        self.permulation = np.random.permutation(self.train_num)
        return self
    
    def get_path_tuple(self, i, dataset_name = "SHA", is_pers=True):
        if dataset_name == "SHA" or dataset_name == "SHB":
            img_name = '/IMG_' + str(i + 1) + ".jpg"
            gt_map_name = '/GT_IMG_' + str(i + 1) + ".npy"
            perspective_map_name = ""
            if is_pers:
                perspective_map_name = '/IMG_' + str(i + 1) + ".mat"
        elif dataset_name == "QNRF":
            assert not(is_pers), "QNRF dataset did not provide perspective map."
            img_name = "/img_" + ("%04d" % (i + 1)) + ".jpg"
            gt_map_name = '/GT_IMG_' + str(i + 1) + ".npy"
        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF")
            return img_name, gt_map_name, perspective_map_name
    
    def resize(self, img, dataset_name="SHA"):
        height = img.size[1]
        width = img.size[0]
        resize_height = height
        resize_width = width
        if dataset_name == "SHA":
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
        elif dataset_name == "SHB":
            break
        elif dataset_name == "UCF_QNRF":
            resize_height = 768
            resize_width = 1024
        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF")
        img = transforms.Resize([resize_height, resize_width])(img)
        return img