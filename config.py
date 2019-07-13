class config(object):
    def __init__(self, cuda_device, model_save_name="", dataset_name="SHA", lr=1e-4, batch_size=5, eval_per_step=30):
        self.min_mae = 10240000
        self.min_loss = 10240000
        self.dataset_name = dataset_name
        if self.dataset_name == "SHA":
            self.eval_num = 182
            self.train_num = 300
            
            self.train_gt_map_path = "/home/zzn/Documents/Datasets/part_A_final/train_data/gt_map_sigma=4_k=7"
            self.train_img_path = "/home/zzn/Documents/Datasets/part_A_final/train_data/images"
            self.train_pers_path = "/home/zzn/Documents/Datasets/part_A_final/train_data/perspective_gt"
            self.eval_gt_map_path = "/home/zzn/Documents/Datasets/part_A_final/test_data/gt_map_sigma=4_k=7"
            self.eval_img_path = "/home/zzn/Documents/Datasets/part_A_final/test_data/images"
            self.eval_gt_path = "/home/zzn/Documents/Datasets/part_A_final/test_data/ground_truth"
            self.eval_pers_path = "/home/zzn/Documents/Datasets/part_A_final/test_data/perspective_gt"
            
        elif self.dataset_name == "SHB":
            self.eval_num = 316
            self.train_num = 400
            
            self.train_gt_map_path = "/home/zzn/Documents/Datasets/part_B_final/train_data/gt_map_sigma=4_k=7"
            self.train_img_path = "/home/zzn/Documents/Datasets/part_B_final/train_data/images"
            self.train_pers_path = "/home/zzn/Documents/Datasets/part_B_final/train_data/perspective_gt"
            self.eval_gt_map_path = "/home/zzn/Documents/Datasets/part_B_final/test_data/gt_map_sigma=4_k=7"
            self.eval_img_path = "/home/zzn/Documents/Datasets/part_B_final/test_data/images"
            self.eval_gt_path = "/home/zzn/Documents/Datasets/part_B_final/test_data/ground_truth"
            self.eval_pers_path = "/home/zzn/Documents/Datasets/part_B_final/test_data/perspective_gt"
            
        elif self.dataset_name == "QNRF":
            self.eval_num = 334
            self.train_num = 1201
            
            self.train_gt_map_path = "/home/zzn/Documents/Datasets/UCF-QNRF_ECCV18/train_data/gt_map_sigma=4_k=7"
            self.train_img_path = "/home/zzn/Documents/Datasets/UCF-QNRF_ECCV18/train_data/images"
            self.train_pers_path = None
            self.eval_gt_map_path = "/home/zzn/Documents/Datasets/UCF-QNRF_ECCV18/test_data/gt_map_sigma=4_k=7"
            self.eval_img_path = "/home/zzn/Documents/Datasets/UCF-QNRF_ECCV18/test_data/images"
            self.eval_gt_path = "/home/zzn/Documents/Datasets/UCF-QNRF_ECCV18/test_data/ground_truth"
            self.eval_pers_path = None
        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF")
        self.learning_rate = lr
        self.train_batch_size = batch_size
        self.epoch = 20000
        self.eval_per_step = eval_per_step
        self.mode = 'crop'
        self.if_random_hsi =True
        self.if_flip = True
#         self.model_save_path = "/data/wangyezhen/checkpoints/RESNET_FPN/RESFPN_ADAMDILATION_A_7_6.pkl"
        self.model_save_path = model_save_name
        self.cuda_device = cuda_device