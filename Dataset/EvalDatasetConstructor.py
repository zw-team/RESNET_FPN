from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as functional
import torch.utils.data as data
import scipy.io as scio
import h5py
import time
import torch
import math

class EvalDatasetConstructor(data.Dataset):
    def __init__(self,
                 validate_num,
                 data_dir_path,
                 gt_dir_path,
                 pers_dir_path=None,
                 mode="crop",
                 dataset_name="SHA"
                 device=None,
                 ):
        self.validate_num = validate_num
        self.imgs = []
        self.data_root = data_dir_path
        self.gt_root = gt_dir_path
        self.pers_root = pers_dir_path
        self.mode = mode
        self.device = device
        if self.device == None:
            raise Exception("Only support GPU version! Please choose the specific GPU device.")
        self.dataset_name = dataset_name
        self.kernel = torch.FloatTensor(torch.ones(1, 1, 2, 2)).to(self.device)
        for i in range(self.validate_num):
            i_n, g_n, p_n = self.get_path_tuple(i, self.dataset_name, not(self.pers_root==None))
            self.imgs.append([self.data_root + i_n, self.gt_root + g_n, self.pers_root + p_n, i + 1])

    def __getitem__(self, index):
        if self.mode == 'crop':
            img_path, gt_map_path, pers_path, img_index = self.imgs[index]
            img = Image.open(img_path).convert("RGB")
            img = self.resize(img, self.dataset_name)
            img = transforms.ToTensor()(img).to(self.device)
            gt_map = Image.fromarray(np.squeeze(np.load(gt_map_path)))
            gt_map = transforms.ToTensor()(gt_map).to(self.device)
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
            gt_map = functional.conv2d(gt_map.view(1, *(gt_shape)), self.kernel, bias=None, stride=2, padding=0)
            p_m = None
            if not(self.pers_root == None):
                p_m = Image.fromarray((scio.loadmat(pers_path)['pmap'][:] / 100))
#                 p_m = Image.fromarray((h5py.File(pers_path)['pmap'][:] / 100).T)
                p_m = transforms.ToTensor()(p_m).to(self.device)
            return img_index, imgs, gt_map.view(1, gt_shape[1] // 2, gt_shape[2] // 2), p_m

    def __len__(self):
        return self.validate_num
    
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
