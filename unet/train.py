# USAGE
# python train.py
# import the necessary packages
import os

from piqa import SSIM

from dataset import ImageLoaderDataset
import config
from torch.nn import MSELoss, Module
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time

from unet.ShadowPresenceLoss import SPLoss
from unet.model_unet import UNet
from unet.model_unet_smaller import UNetSmaller

info_file = None


class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)


class Loss(Module):
    def __init__(self, alpha=0.5, ):
        super(Loss, self).__init__()
        self.alpha = alpha
        self.sp_loss = SPLoss()
        self.mse_loss = MSELoss()
        # self.ssim = SSIM()

    def to(self, device):
        self.mse_loss.to(device)
        # self.ssim.to(device)
        self.sp_loss.to(device)
        return self

    def forward(self, x, y):
        mse = self.mse_loss(x, y)
        spl = self.sp_loss(x, y)
        # ssim_loss = 1. - self.ssim(x, y)
        combined_loss = self.alpha * mse + (1. - self.alpha) * spl
        return combined_loss


def write_info():
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


def load_data():
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
    # train_si = train_si[2000:]
    # train_gti = train_gti[2000:]
    # eval_si = eval_si[100:]
    # eval_gti = eval_gti[100:]
    # train_si = train_si[0::15]
    # train_gti = train_gti[0::15]
    # eval_si = eval_si[0::15]
    # eval_gti = eval_gti[0::15]

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
    trainDS = ImageLoaderDataset(train_paths=train_si, gt_paths=train_gti, transforms=train_transforms)
    evalDS = ImageLoaderDataset(train_paths=eval_si, gt_paths=eval_gti, transforms=train_transforms)
    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(evalDS)} examples in the eval set...")

    info_file.writelines([
        f"Train set transforms: {train_transforms}\n\n",
        f"Evaluation set transforms: {train_transforms}\n\n",
        f"Train set contains {len(trainDS)} image pairs\n\n",
        f"Evaluation set contains {len(evalDS)} image pairs\n\n",
    ])
    info_file.flush()

    # create the training and eval data loaders
    trainLoader = DataLoader(trainDS, shuffle=False, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)
    evalLoader = DataLoader(evalDS, shuffle=False, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)

    # calculate steps per epoch for training and eval set
    trainSteps = len(trainDS) // config.BATCH_SIZE
    evalSteps = len(evalDS) // config.BATCH_SIZE

    print("[INFO] Train data split successfully...")

    return trainLoader, evalLoader, trainSteps, evalSteps


def load_model():
    # initialize our UNet model
    unet = UNet()
    # unet = UNetSmaller()

    if config.LOAD_MODEL is not None:
        unet = torch.load(config.LOAD_MODEL)

        if config.FINE_TUNE:
            # Freeze encoder layers
            for name, p in unet.named_parameters():
                if "enc" in name:
                    p.requires_grad = False

    unet.to(config.DEVICE)

    return unet


def train(unet, trainLoader, evalLoader, trainSteps, evalSteps):

    # initialize loss function and optimizer
    lossFunc = MSELoss().to(config.DEVICE)
    # lossFunc = SSIMLoss().to(config.DEVICE)
    # lossFunc = SMLoss(0.5).to(config.DEVICE)
    # lossFunc = SPLoss().to(config.DEVICE)
    # lossFunc = Loss(0.5).to(config.DEVICE)
    optimizer = AdamW(unet.parameters(), lr=config.INIT_LR, weight_decay=config.WEIGHT_DECAY)

    torch.backends.cudnn.benchmark = True

    # initialize a dictionary to store training history
    H = {"train_loss": [], "eval_loss": []}

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in range(config.START_EPOCH, config.NUM_EPOCHS):
        # set the model in training mode
        unet.train()

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalEvalLoss = 0

        # loop over the training set
        for (x, y) in tqdm(trainLoader):
            # send the input to the device
            (x, y) = (x.to(config.DEVICE, non_blocking=True), y.to(config.DEVICE, non_blocking=True))

            # perform a forward pass and calculate the training loss
            prediction = unet(x)
            # loss = lossFunc(prediction)
            loss = lossFunc(prediction, y)

            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add the loss to the total training loss so far
            totalTrainLoss += loss

        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            unet.eval()

            # loop over the validation set
            for (x, y) in evalLoader:
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

                # make the predictions and calculate the validation loss
                prediction = unet(x)
                # totalEvalLoss += lossFunc(prediction)
                totalEvalLoss += lossFunc(prediction, y)

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgEvalLoss = totalEvalLoss / evalSteps

        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["eval_loss"].append(avgEvalLoss.cpu().detach().numpy())

        # print the model training and validation information
        print(f"[INFO] EPOCH: {e + 1}/{config.NUM_EPOCHS}")
        print("Train loss: {:.6f}, Eval loss: {:.4f}".format(avgTrainLoss, avgEvalLoss))

        info_file.writelines([
            f"[INFO] EPOCH: {e + 1}/{config.NUM_EPOCHS}\n",
            "Train loss: {:.6f}, Eval loss: {:.4f}\n".format(avgTrainLoss, avgEvalLoss),
        ])
        info_file.flush()

        show_plot(H)
        if (e+1) % 5 == 0:
            torch.save(unet, os.path.join(config.BASE_OUTPUT, f"unet_shadow_{config.initial_time}_e{e+1}.pth"))
        torch.save(unet, os.path.join(config.BASE_OUTPUT, f"unet_shadow_{config.initial_time}.pth"))

    # display the total time needed to perform the training
    endTime = time.time()

    print_results(startTime, endTime, H)


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
    write_info()
    trainLoader, evalLoader, trainSteps, evalSteps = load_data()
    unet = load_model()
    train(unet, trainLoader, evalLoader, trainSteps, evalSteps)
    info_file.close()
