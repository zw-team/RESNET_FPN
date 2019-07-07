import random
import math
import numpy as np
import sys
from PIL import Image
from utils import show
from metrics import AEBatch, SEBatch
import time
import torch
import scipy.io as scio

class Estimator(object):
    def __init__(self, setting, eval_loader, criterion=torch.nn.MSELoss(reduction="sum")):
        self.setting = setting
        self.ae_batch = AEBatch().to(self.setting.cuda_device)
        self.se_batch = SEBatch().to(self.setting.cuda_device)
        self.criterion = criterion
        self.eval_loader = eval_loader
        
    def evaluate(self, model, is_show=True):
        net = model.eval()
        MAE_, MSE_, loss_ = [], [], []
        rand_number, cur, time_cost = random.randint(0, self.setting.eval_num - 1), 0, 0
        for eval_img_index, eval_img, eval_gt, eval_pers in self.eval_loader:
            start = time.time()
            eval_patchs, eval_pers = torch.squeeze(eval_img), torch.squeeze(eval_pers, dim=0)
            eval_gt_shape = eval_gt.shape
            prediction_map = torch.zeros(eval_gt_shape).to(self.setting.cuda_device)
            img_index = eval_img_index.cpu().numpy()[0]
            with torch.no_grad():
                eval_prediction = net(eval_patchs, eval_pers)
                eval_patchs_shape = eval_prediction.shape
                # test cropped patches
                self.test_crops(eval_patchs_shape, eval_prediction, prediction_map)
                gt_counts = self.get_gt_num(img_index)
                # calculate metrics
                batch_ae = self.ae_batch(prediction_map, gt_counts).data.cpu().numpy()
                batch_se = self.se_batch(prediction_map, gt_counts).data.cpu().numpy()
                loss = self.criterion(prediction_map, eval_gt)
                loss_.append(loss.data.item())
                MAE_.append(batch_ae)
                MSE_.append(batch_se)
                # show sample
                if rand_number == cur and is_show:
                    validate_pred_map = np.squeeze(prediction_map.permute(0, 2, 3, 1).data.cpu().numpy())
                    validate_gt_map = np.squeeze(eval_gt.permute(0, 2, 3, 1).data.cpu().numpy())
                    pred_counts = np.sum(validate_pred_map)
                    self.show_sample(img_index, gt_counts, pred_counts, validate_gt_map, validate_pred_map)
            cur += 1
            torch.cuda.synchronize()
            end = time.time()
            time_cost += (end - start)

        # return the validate loss, validate MAE and validate RMSE
        MAE_, MSE_, loss_ = np.reshape(MAE_, [-1]), np.reshape(MSE_, [-1]), np.reshape(loss_, [-1])
        return np.mean(MAE_), np.sqrt(np.mean(MSE_)), np.mean(loss_), time_cost
    
    def get_gt_num(self, index):
        if self.setting.dataset_name == "QNRF":
            gt_path = self.setting.eval_gt_path + "/img_" + ("%04d" % (index)) + "_ann.mat"
            gt_counts = len(scio.loadmat(gt_path)['annPoints'])
        elif self.setting.dataset_name == "SHA" or self.setting.dataset_name == "SHB":
            gt_path = self.setting.eval_gt_path + "/GT_IMG_" + str(index) + ".mat"
            gt_counts = len(scio.loadmat(gt_path)['image_info'][0][0][0][0][0])
        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF")
        return gt_counts
    
    def show_sample(self, index, gt_counts, pred_counts, eval_gt_map, eval_pred_map):
        if self.setting.dataset_name == "QNRF":
            origin_image = Image.open(self.setting.eval_img_path + "/img_" + ("%04d" % index) + ".jpg")
        elif self.setting.dataset_name == "SHA" or self.setting.dataset_name == "SHB":
            origin_image = Image.open(self.setting.eval_img_path + "/IMG_" + str(index) + ".jpg")
        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF")
        show(origin_image, eval_gt_map, eval_pred_map, index)
        sys.stdout.write('The gt counts of the above sample:{}, and the pred counts:{}\n'.format(gt_counts, pred_counts))
    
    def test_crops(self, eval_shape, eval_p, pred_m):
        for i in range(3):
            for j in range(3):
                start_h, start_w = math.floor(eval_shape[2] / 4), math.floor(eval_shape[3] / 4)
                valid_h, valid_w = eval_shape[2] // 2, eval_shape[3] // 2
                pred_h = math.floor(3 * eval_shape[2] / 4) + (eval_shape[2] // 2) * (i - 1)
                pred_w = math.floor(3 * eval_shape[3] / 4) + (eval_shape[3] // 2) * (j - 1)
                if i == 0:
                    valid_h = math.floor(3 * eval_shape[2] / 4)
                    start_h = 0
                    pred_h = 0
                elif i == 2:
                    valid_h = math.ceil(3 * eval_shape[2] / 4)

                if j == 0:
                    valid_w = math.floor(3 * eval_shape[3] / 4)
                    start_w = 0
                    pred_w = 0
                elif j == 2:
                    valid_w = math.ceil(3 * eval_shape[3] / 4)
                pred_m[:, :, pred_h:pred_h + valid_h, pred_w:pred_w + valid_w] += eval_p[i * 3 + j:i * 3 + j + 1, :,start_h:start_h + valid_h, start_w:start_w + valid_w]