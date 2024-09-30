# USAGE
# python train.py
# import the necessary packages
import os

from dataset import ImageLoaderDataset
import config
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time

from unet.model_unet import UNet


def load_data():
    # load the image and mask filepaths in a sorted manner
    shadow_image = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
    gt_image = sorted(list(paths.list_images(config.GT_DATASET_PATH)))

    # partition the data into training and evaluation splits using part of
    # the data for training and the remaining for evaluation during training
    split = train_test_split(shadow_image, gt_image, test_size=config.EVAL_SPLIT, random_state=42)

    # unpack the data split
    (train_si, eval_si) = split[:2]
    (train_gti, eval_gti) = split[2:]

    # TODO: Disable on real training
    # train_si = train_si[:15]
    # train_gti = train_gti[:15]

    # define transformations
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
        ]),
        transforms.ToTensor()
    ])

    pretrain_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
        ]),
        transforms.ToTensor()
    ])

    # create the train and evaluation datasets
    trainDS = ImageLoaderDataset(train_paths=train_si, gt_paths=train_gti, transforms=train_transforms)
    evalDS = ImageLoaderDataset(train_paths=eval_si, gt_paths=eval_gti, transforms=test_transform)
    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(evalDS)} examples in the eval set...")

    # create the training and eval data loaders
    trainLoader = DataLoader(trainDS, shuffle=True, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)
    evalLoader = DataLoader(evalDS, shuffle=False, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)

    # calculate steps per epoch for training and eval set
    trainSteps = len(trainDS) // config.BATCH_SIZE
    evalSteps = len(evalDS) // config.BATCH_SIZE

    print("[INFO] Train data split successfully...")

    return trainLoader, evalLoader, trainSteps, evalSteps


def load_model():
    # initialize our UNet model
    unet = UNet()

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
    optimizer = Adam(unet.parameters(), lr=config.INIT_LR)

    torch.backends.cudnn.benchmark = True

    # initialize a dictionary to store training history
    H = {"train_loss": [], "eval_loss": []}

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in (range(config.NUM_EPOCHS)):
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

        show_plot(H)
        torch.save(unet, config.MODEL_PATH)

    # display the total time needed to perform the training
    endTime = time.time()

    print_results(startTime, endTime, H)

    # serialize the model to disk
    torch.save(unet, config.MODEL_PATH)


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
    trainLoader, evalLoader, trainSteps, evalSteps = load_data()
    unet = load_model()
    train(unet, trainLoader, evalLoader, trainSteps, evalSteps)
