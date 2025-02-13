# USAGE
# python train.py
# import the necessary packages
import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

from matplotlib import pyplot as plt
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from dataset import ImageLoaderDataset
import config
from torch.nn import MSELoss
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from unet.model_unet import UNet
from unet.model_unet_smaller import UNetSmaller

info_file = None


parser = argparse.ArgumentParser(description='UNet Training')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def write_info(args):
    # info_file = open(os.path.join(config.BASE_OUTPUT, "info.md"), "w")
    info_file.writelines([
        "# Training info for NN model U-Net\n\n",
        f"Start time is: {config.initial_time}\n\n",
        "\n\n",
        f"Epochs: {config.NUM_EPOCHS}\n\n",
        f"LR: {config.INIT_LR}\n\n",
        f"Weight decay: {config.WEIGHT_DECAY}\n\n",
        f"Optimizer: Adam with weight decay\n\n",
        f"Dropout: {config.DROPOUT}\n\n",
        f"Image input size: {config.INPUT_IMAGE_WIDTH}x{config.INPUT_IMAGE_HEIGHT}\n\n",
        f"Batch normalization: {config.BATCH_NORM}\n\n",
        f"Load model: {config.LOAD_MODEL}\n\n",
        f"Fine tune (freeze encoder layers): {config.FINE_TUNE}\n\n",
        f"\n\n",
        f"Training images: {config.IMAGE_DATASET_PATH}\n\n",
        f"Ground truth images: {config.GT_DATASET_PATH}\n\n",
        f"Train/eval split (of training images): {config.EVAL_SPLIT}\n\n",
        "\n\n",
    ])
    info_file.flush()


def main():
    args = parser.parse_args()
    # write_info(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn("nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1

    model = UNet()
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = MSELoss().to(device)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_loader, val_loader, train_sampler = load_data(args)

    train(train_loader, val_loader, model, criterion, optimizer, device, args)


def load_data(args):
    # load the image and mask filepaths in a sorted manner
    shadow_image = []
    gt_image = []
    for image_dir in config.IMAGE_DATASET_PATHS:
        shadow_image.extend(sorted(list(paths.list_images(image_dir))))
    for image_dir in config.GT_DATASET_PATHS:
        gt_image.extend(sorted(list(paths.list_images(image_dir))))

    # partition the data into training and evaluation splits using part of
    # the data for training and the remaining for evaluation during training
    split = train_test_split(shadow_image, gt_image, test_size=config.EVAL_SPLIT, random_state=42)

    # unpack the data split
    (train_si, eval_si) = split[:2]
    (train_gti, eval_gti) = split[2:]

    # TODO: Disable on real training
    # train_si = train_si[1000:]
    # train_gti = train_gti[1000:]
    # eval_si = eval_si[100:]
    # eval_gti = eval_gti[100:]
    train_si = train_si[0::15]
    train_gti = train_gti[0::15]
    eval_si = eval_si[0::15]
    eval_gti = eval_gti[0::15]

    # define transformations
    test_transform = []

    transforms_ds = ['Resize']

    train_transforms = [
        'RandomResizedCrop',
        'ColorJitter',
        'RandomHorizontalFlip'
        'RandomVerticalFlip',
        'RandomRotation',
    ]

    pretrain_transforms = transforms.Compose([
        'Resize',
        'ColorJitter',
        'GaussianNoise',
        'RandomHorizontalFlip'
        'RandomVerticalFlip',
        'RandomRotation',
    ])

    # create the train and evaluation datasets
    train_dataset = ImageLoaderDataset(train_paths=train_si, gt_paths=train_gti, transforms=train_transforms)
    val_dataset = ImageLoaderDataset(train_paths=eval_si, gt_paths=eval_gti, transforms=train_transforms)
    print(f"[INFO] found {len(train_dataset)} examples in the training set...")
    print(f"[INFO] found {len(val_dataset)} examples in the eval set...")

    info_file.writelines([
        f"Train set transforms: {train_transforms}\n\n",
        f"Evaluation set transforms: {train_transforms}\n\n",
        f"Train set contains {len(train_dataset)} image pairs\n\n",
        f"Evaluation set contains {len(val_dataset)} image pairs\n\n",
    ])
    info_file.flush()

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    # create the training and eval data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    print("[INFO] Train data split successfully...")

    return train_loader, val_loader, train_sampler


def train(train_loader, val_loader, model, criterion, optimizer, device, args):

    # initialize a dictionary to store training history
    H = {"train_loss": [], "eval_loss": []}

    # loop over epochs
    print("[INFO] training the network...")
    for epoch in range(args.start_epoch, args.epochs):
        # switch to train mode
        model.train()

        # initialize the total training and validation loss
        total_train_loss = 0
        total_eval_loss = 0
        train_steps = 0
        val_steps = 0

        # loop over the training set
        for (x, y) in tqdm(train_loader):
            # send the input to the device
            (x, y) = (x.to(device, non_blocking=True), y.to(device, non_blocking=True))
            train_steps += 1

            # perform a forward pass and calculate the training loss
            prediction = model(x)
            loss = criterion(prediction, y)

            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add the loss to the total training loss so far
            total_train_loss += loss

        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            val_steps += 1

            # loop over the validation set
            for (x, y) in val_loader:
                # send the input to the device
                (x, y) = (x.to(device), y.to(device))

                # make the predictions and calculate the validation loss
                prediction = model(x)
                total_eval_loss += criterion(prediction, y)

        # calculate the average training and validation loss
        avg_train_loss = total_train_loss / train_steps
        avg_eval_loss = total_eval_loss / val_steps

        # update our training history
        H["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        H["eval_loss"].append(avg_eval_loss.cpu().detach().numpy())

        # print the model training and validation information
        print(f"[INFO] EPOCH: {epoch + 1}/{args.epochs}")
        print("Train loss: {:.6f}, Eval loss: {:.4f}".format(avg_train_loss, avg_eval_loss))

        info_file.writelines([
            f"[INFO] EPOCH: {epoch + 1}/{args.epochs}\n",
            "Train loss: {:.6f}, Eval loss: {:.4f}\n".format(avg_train_loss, avg_eval_loss),
        ])
        info_file.flush()

        show_plot(H)
        if (epoch+1) % 5 == 0:
            torch.save(model, os.path.join(config.BASE_OUTPUT, f"unet_shadow_{config.initial_time}_e{epoch+1}.pth"))
        torch.save(model, os.path.join(config.BASE_OUTPUT, f"unet_shadow_{config.initial_time}.pth"))

    # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
    #                                             and args.rank % ngpus_per_node == 0):
    #     save_checkpoint({
    #         'epoch': epoch + 1,
    #         'arch': args.arch,
    #         'state_dict': model.state_dict(),
    #         'best_acc1': best_acc1,
    #         'optimizer': optimizer.state_dict(),
    #         'scheduler': scheduler.state_dict()
    #     }, is_best)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def print_results(startTime, endTime, H):
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

    show_plot(H)


def show_plot(H):
    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["eval_loss"], label="eval_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig(config.PLOT_PATH)
    plt.close()



if __name__ == '__main__':
    if not os.path.exists(config.BASE_OUTPUT):
        os.mkdir(config.BASE_OUTPUT)
    info_file = open(os.path.join(config.BASE_OUTPUT, "info.md"), "w")

    main()

    info_file.close()
