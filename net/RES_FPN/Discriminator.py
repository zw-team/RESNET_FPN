import torch
import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense_1 = nn.Sequential(nn.Linear(2048, 512, bias=True), nn.ReLU(inplace=True))
        self.dense_2 = nn.Sequential(nn.Linear(512, 1, bias=True), nn.Sigmoid())
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = self.dense_1(x.view(*x.shape[:2]))
        x = self.dense_2(x)
        return x