import os
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.nccl.version())
