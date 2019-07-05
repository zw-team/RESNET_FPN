from net.FPN import FPN
import sys
import numpy as np
import random
import math
from eval.eval_by_cropping import eval_model
from Dataset.EvalDatasetConstructor import EvalDatasetConstructor
from Dataset.TrainDatasetConstructor import TrainDatasetConstructor
from metrics import AEBatch, SEBatch
from PIL import Image
import time
import torch
# obtain the gpu device
assert torch.cuda.is_available()
cuda_device = torch.device("cuda:6")
torch.cuda.set_device(cuda_device)
# torch.backends.cudnn.benchmark=True
# %matplotlib inline
f = open("/data/wangyezhen/Logs/RESNET_FPN/RESNet_6-21_A.txt", "w")
# config
config = {
'min_mae':10240000,
'min_loss':10240000,
'eval_num':182,
'train_num':300,
'learning_rate': 1e-4,
'train_batch_size': 10,
'epoch': 20000,
'eval_per_step': 30,
'mode':'crop',
'if_random_hsi':True,
'if_flip':True,
'stage':'numeration',
'gt_map_path':"/data/wangyezhen/datasets/part_A_final/train_data/gt_map_sigma=4_k=7",
'img_path':"/data/wangyezhen/datasets/part_A_final/train_data/images",
'cuda_device':cuda_device,
'gt_map_path_t':"/data/wangyezhen/datasets/part_A_final/test_data/gt_map_sigma=4_k=7",
'img_path_t':"/data/wangyezhen/datasets/part_A_final/test_data/images",
'gt_path_t':"/data/wangyezhen/datasets/part_A_final/test_data/ground_truth",
'model_save_path':"/data/wangyezhen/checkpoints/RESNET_FPN/RESNet_6-21_A.pkl"
}

# data_load
train_dataset = TrainDatasetConstructor(
    config['img_path'],
    config['gt_map_path'],
    config['train_num'],
    mode=config['mode'],
    stage=config['stage'],
    device=config['cuda_device'],
    if_random_hsi=config['if_random_hsi'],
    if_flip=config['if_flip'])
eval_dataset = EvalDatasetConstructor(
    config['img_path_t'],
    config['gt_map_path_t'],
    config['eval_num'],
    mode=config['mode'],
    stage=config['stage'],
    device=config['cuda_device'])
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=config['train_batch_size'])
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=1)

# model construct
model = FPN().to(config['cuda_device'])
net = torch.nn.DataParallel(model, device_ids=[6, 7], output_device=cuda_device).to(config['cuda_device'])
optimizer = torch.optim.Adam(net.parameters(), config['learning_rate'])
criterion = torch.nn.MSELoss(reduction='sum').to(config['cuda_device'])
ae_batch = AEBatch().to(config['cuda_device'])
se_batch = SEBatch().to(config['cuda_device'])
modules = {
    'model': model,
    'shape': None,
    'ae': ae_batch,
    'se': se_batch,
    'loss': criterion
}

step = 0
eval_loss = []
eval_mae = []
eval_rmse = []
for epoch_index in range(config['epoch']):
    dataset = train_dataset.shuffle()
    loss_list = []
    time_per_epoch = 0
    for train_img_index, train_img, train_gt in train_loader:
        if step % config['eval_per_step'] == 0:
            validate_MAE, validate_RMSE, validate_loss, time_cost = eval_model(
                config, eval_loader, modules, False)
            eval_loss.append(validate_loss)
            eval_mae.append(validate_MAE)
            eval_rmse.append(eval_rmse)
            f.write(
                'In step {}, epoch {}, loss = {}, eval_mae = {}, eval_rmse = {}, time cost eval = {}s\n'
                .format(step, epoch_index, validate_loss, validate_MAE,
                        validate_RMSE, time_cost))
            f.flush()
            #             save model
            if config['stage'] == 'numeration' and config[
                    'min_mae'] > validate_MAE:
                config['min_mae'] = validate_MAE
                torch.save(net.state_dict(), config['model_save_path'])
        net.train()
        torch.cuda.empty_cache()
        x = train_img
        y = train_gt
        start = time.time()
        prediction = net(x)
        loss = criterion(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        loss_list.append(loss.data.item())
        optimizer.step()

        step += 1
        torch.cuda.empty_cache()
        end = time.time()
        time_per_epoch += end - start