# config
import sys
import numpy as np
import torch
assert torch.cuda.is_available()
f = open("/home/zzn/PycharmProjects/RESNET_FPN/Logs/RES_50_PERS_FPN_A_7_7.txt", "w")
cuda_device = torch.device("cuda:0")
torch.cuda.set_device(cuda_device)
from config import config
setting = config(
    cuda_device, 
    model_save_name="/home/zzn/PycharmProjects/RESNET_FPN/StateDicts/RES_50_PERS_FPN_A_7_7.pkl", 
    dataset_name="SHA",
    lr=1e-4, 
    batch_size=5, 
    eval_per_step=120
)
# data_load
from Dataset.DatasetConstructor import TrainDatasetConstructor,EvalDatasetConstructor
train_dataset = TrainDatasetConstructor(
    setting.train_num,
    setting.train_img_path,
    setting.train_gt_map_path,
    setting.train_pers_path,
    mode=setting.mode,
    dataset_name=setting.dataset_name,
    device=setting.cuda_device,
    if_random_hsi=setting.if_random_hsi,
    if_flip=setting.if_random_hsi)
eval_dataset = EvalDatasetConstructor(
    setting.eval_num,
    setting.eval_img_path,
    setting.eval_gt_map_path,
    setting.eval_pers_path,
    mode=setting.mode,
    dataset_name=setting.dataset_name,
    device=setting.cuda_device)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=setting.train_batch_size)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=1)
# model construct
from net.RES_PERS_FPN.PERS_FPN import PERS_FPN
from eval.Estimator import Estimator
# net = FPN().to(setting.cuda_device)
net = PERS_FPN(sigma=[1, 1, 0, 1], updates_signal=[True, True, True, True]).to(setting.cuda_device)
optimizer = torch.optim.Adam(net.parameters(), setting.learning_rate)
criterion = torch.nn.MSELoss(reduction='sum').to(setting.cuda_device)
estimator = Estimator(setting, eval_loader, criterion=criterion)

import time
step = 0
eval_loss, eval_mae, eval_rmse = [], [], []
for epoch_index in range(setting.epoch):
    dataset = train_dataset.shuffle()
    loss_list = []
    time_per_epoch = 0
    for train_img_index, train_img, train_gt, train_pers in train_loader:
        if step % setting.eval_per_step == 0:
            validate_MAE, validate_RMSE, validate_loss, time_cost = estimator.evaluate(net, False)
            eval_loss.append(validate_loss)
            eval_mae.append(validate_MAE)
            eval_rmse.append(eval_rmse)
            f.write(
                'In step {}, epoch {}, loss = {}, eval_mae = {}, eval_rmse = {}, time cost eval = {}s\n'
                .format(step, epoch_index, validate_loss, validate_MAE,
                        validate_RMSE, time_cost))
            f.flush()
            #             save model
            if setting.min_mae > validate_MAE:
                setting.min_mae = validate_MAE
                torch.save(net.state_dict(), setting.model_save_path)
        net.train()
        torch.cuda.empty_cache()
        x, y = train_img, train_gt
        start = time.time()
        prediction = net(x, train_pers)
        loss = criterion(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        loss_list.append(loss.data.item())
        optimizer.step()

        step += 1
        torch.cuda.empty_cache()
        end = time.time()
        time_per_epoch += end - start