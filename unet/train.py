# USAGE
# python train.py
import os

from piqa import SSIM

from dataset import ImageLoaderDataset
import config
from torch.nn import MSELoss, BatchNorm2d, Dropout, Conv2d, MaxPool2d, Module, ReLU, Sigmoid, Sequential, \
    ConvTranspose2d, L1Loss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time

from unet.ShadowPresenceLoss import SPLoss
from unet.model_unet import UNet, Encoder, Decoder, Block, Bottleneck, AttentionBlock

info_file = None

torch.serialization.add_safe_globals(
    [UNet, set, MSELoss, BatchNorm2d, Dropout, Conv2d, MaxPool2d, Module, ReLU, Sigmoid, Sequential,
    ConvTranspose2d, L1Loss, UNet, Encoder, Decoder, Block, Bottleneck, AttentionBlock])


class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)


class MSESSIMLoss(Module):
    def __init__(self, alpha=0.5):
        super(MSESSIMLoss, self).__init__()
        self.alpha = alpha
        self.mse = MSELoss()
        self.ssim = SSIMLoss()

    def to(self, device):
        self.mse.to(device)
        self.ssim.to(device)
        return self

    def forward(self, x, y):
        mse = self.mse(x, y)
        ssim_loss = self.ssim(x, y)
        return self.alpha * mse + (1. - self.alpha) * ssim_loss


class L1SSIMLoss(Module):
    def __init__(self, alpha=0.5):
        super(L1SSIMLoss, self).__init__()
        self.alpha = alpha
        self.l1 = L1Loss()
        self.ssim = SSIMLoss()

    def to(self, device):
        self.l1.to(device)
        self.ssim.to(device)
        return self

    def forward(self, x, y):
        l1 = self.l1(x, y)
        ssim = self.ssim(x, y)
        return self.alpha * l1 + (1. - self.alpha) * ssim


def write_info():
    info_file.writelines([
        "# Training info for NN model U-Net\n\n",
        f"Start time is: {config.initial_time}\n\n",
        "\n\n",
        f"Epochs: {config.NUM_EPOCHS}\n\n",
        f"LR: {config.INIT_LR}\n\n",
        f"Weight decay: {config.WEIGHT_DECAY}\n\n",
        f"Loss function: {config.LOSS_FUNCTION}\n\n",
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
    # split = train_test_split(shadow_image, gt_image, test_size=config.EVAL_SPLIT, random_state=42)
    split = train_test_split(shadow_image, gt_image, test_size=0.01, random_state=42)

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

    # create the train and evaluation datasets
    train_ds = ImageLoaderDataset(train_paths=train_si, gt_paths=train_gti, transforms=config.TRANSFORMS,
                                  mean=config.MEAN, std=config.STD)
    eval_ds = ImageLoaderDataset(train_paths=eval_si, gt_paths=eval_gti, transforms=config.TRANSFORMS, mean=config.MEAN,
                                 std=config.STD)
    print(f"[INFO] found {len(train_ds)} examples in the training set...")
    print(f"[INFO] found {len(eval_ds)} examples in the eval set...")

    info_file.writelines([
        f"Transforms: {config.TRANSFORMS}\n\n",
        f"Train set contains {len(train_ds)} image pairs\n\n",
        f"Evaluation set contains {len(eval_ds)} image pairs\n\n",
        f"Dataset mean: {config.MEAN}\n\n",
        f"Dataset std: {config.STD}\n\n",
    ])
    info_file.flush()

    # create the training and eval data loaders
    train_loader = DataLoader(train_ds, shuffle=False, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)
    eval_loader = DataLoader(eval_ds, shuffle=False, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)

    # calculate steps per epoch for training and eval set
    train_steps = len(train_ds) // config.BATCH_SIZE
    eval_steps = len(eval_ds) // config.BATCH_SIZE

    print("[INFO] Train data split successfully...")

    return train_loader, eval_loader, train_steps, eval_steps


def load_model():
    # initialize our UNet model
    model = UNet()
    # model = UNetSmaller()

    if config.LOAD_MODEL is not None:
        model = torch.load(config.LOAD_MODEL, weights_only=True)

        if config.FINE_TUNE:
            # Freeze encoder layers
            for name, p in model.named_parameters():
                if "enc" in name:
                    p.requires_grad = False

    model.to(config.DEVICE)

    return model


def train(model, train_loader, eval_loader, train_steps, eval_steps):
    # initialize loss function and optimizer
    if config.LOSS_FUNCTION == "MSE":
        loss_func = MSELoss().to(config.DEVICE)
    elif config.LOSS_FUNCTION == "SSIM":
        loss_func = SSIMLoss().to(config.DEVICE)
    elif config.LOSS_FUNCTION == "MSE_SSIM":
        loss_func = MSESSIMLoss(0.5).to(config.DEVICE)
    elif config.LOSS_FUNCTION == "L1_SSIM":
        loss_func = L1SSIMLoss(0.5).to(config.DEVICE)
    elif config.LOSS_FUNCTION == "SP":
        loss_func = SPLoss().to(config.DEVICE)
    elif config.LOSS_FUNCTION == "L1":
        loss_func = L1Loss().to(config.DEVICE)
    else:
        raise Exception("Unknown loss function")
    optimizer = AdamW(model.parameters(), lr=config.INIT_LR, weight_decay=config.WEIGHT_DECAY)

    torch.backends.cudnn.benchmark = True

    # initialize a dictionary to store training history
    history = {"train_loss": [], "eval_loss": []}

    # loop over epochs
    print("[INFO] training the network...")
    start_time = time.time()
    for e in range(config.START_EPOCH, config.NUM_EPOCHS):
        # set the model in training mode
        model.train()

        # initialize the total training and validation loss
        total_train_loss = 0
        total_eval_loss = 0

        # loop over the training set
        for (x, y) in tqdm(train_loader):
            # send the input to the device
            (x, y) = (x.to(config.DEVICE, non_blocking=True), y.to(config.DEVICE, non_blocking=True))

            # perform a forward pass and calculate the training loss
            prediction = model(x)
            # loss = lossFunc(prediction)
            loss = loss_func(prediction, y)

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

            # loop over the validation set
            for (x, y) in eval_loader:
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

                # make the predictions and calculate the validation loss
                prediction = model(x)
                # totalEvalLoss += lossFunc(prediction)
                total_eval_loss += loss_func(prediction, y)

        # calculate the average training and validation loss
        avg_train_loss = total_train_loss / train_steps
        avg_eval_loss = total_eval_loss / eval_steps

        # update our training history
        history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        history["eval_loss"].append(avg_eval_loss.cpu().detach().numpy())

        # print the model training and validation information
        print(f"[INFO] EPOCH: {e + 1}/{config.NUM_EPOCHS}")
        print("Train loss: {:.6f}, Eval loss: {:.4f}".format(avg_train_loss, avg_eval_loss))

        info_file.writelines([
            f"[INFO] EPOCH: {e + 1}/{config.NUM_EPOCHS}\n",
            "Train loss: {:.6f}, Eval loss: {:.4f}\n".format(avg_train_loss, avg_eval_loss),
        ])
        info_file.flush()

        show_plot(history)
        if (e + 1) % 5 == 0:
            torch.save(model, os.path.join(config.BASE_OUTPUT, f"unet_shadow_{config.initial_time}_e{e + 1}.pth"))
        torch.save(model, os.path.join(config.BASE_OUTPUT, f"unet_shadow_{config.initial_time}.pth"))

    # display the total time needed to perform the training
    end_time = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(end_time - start_time))
    info_file.writelines(["[INFO] total time taken to train the model: {:.2f}s".format(end_time - start_time)])
    info_file.flush()

    show_plot(history)


def show_plot(history):
    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["eval_loss"], label="eval_loss")
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
