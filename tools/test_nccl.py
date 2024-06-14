import os
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net = nn.Linear(10, 5)
    def forward(self, x):
        return self.net(self.relu(self.net1(x)))


# os.environ['MASTER_ADDR'] = 'gpu14'
dist.init_process_group(backend='nccl')
print(f"[ {os.getpid()} ] world_size = {dist.get_world_size()}, " + f"rank = {dist.get_rank()}, backend={dist.get_backend()}")
model = ToyModel().cuda(device=0)
print('model = ToyModel().cuda(device=0) called')
ddp_model = DistributedDataParallel(model, device_ids=[0])
print('ddp_model = DistributedDataParallel(model, device_ids=[0]) called')
