# USAGE
# python train.py
# import the necessary packages
import os

from dataset import ImageLoaderDataset
import config
from torch.nn import BCEWithLogitsLoss, L1Loss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time

from unet.model_my2 import UNet


def train():
    # load the image and mask filepaths in a sorted manner
    shadow_image = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
    shadowless_image = sorted(list(paths.list_images(config.GT_DATASET_PATH)))

    # partition the data into training and testing splits using 85% of
    # the data for training and the remaining 15% for testing
    split = train_test_split(shadow_image, shadowless_image, test_size=config.TEST_SPLIT, random_state=42)

    # unpack the data split
    (train_si, test_si) = split[:2]
    (train_sli, test_sli) = split[2:]

    # TODO: Disable on real training
    # train_si = train_si[:15]
    # train_sli = train_sli[:15]

    # write the testing image paths to disk so that we can use then
    # when evaluating/testing our model
    print("[INFO] saving testing image paths...")
    os.mkdir(config.BASE_OUTPUT)
    f = open(config.TEST_PATHS, "w")
    f.write("\n".join(test_si))
    f.close()

    # define transformations
    ds_transforms = transforms.Compose([transforms.Resize((config.INPUT_IMAGE_HEIGHT,
                                                           config.INPUT_IMAGE_WIDTH)),
                                         transforms.ToTensor()])
    # create the train and test datasets
    trainDS = ImageLoaderDataset(shadow_paths=train_si, shadowless_paths=train_sli, transforms=ds_transforms)
    testDS = ImageLoaderDataset(shadow_paths=test_si, shadowless_paths=test_sli, transforms=ds_transforms)
    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")

    # create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)
    testLoader = DataLoader(testDS, shuffle=False, batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY)

    # calculate steps per epoch for training and test set
    trainSteps = len(trainDS) // config.BATCH_SIZE
    testSteps = len(testDS) // config.BATCH_SIZE

    print("[INFO] Train data split successfully...")

    # initialize our UNet model
    unet = UNet().to(config.DEVICE)

    # initialize loss function and optimizer
    lossFunc = MSELoss().to(config.DEVICE)
    optimizer = Adam(unet.parameters(), lr=config.INIT_LR)

    torch.backends.cudnn.benchmark = True

    # initialize a dictionary to store training history
    H = {"train_loss": [], "test_loss": []}

    # loop over epochs
    print("[INFO] training the network...")
    startTime = time.time()
    for e in (range(config.NUM_EPOCHS)):
        # set the model in training mode
        unet.train()

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0

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
            for (x, y) in testLoader:
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

                # make the predictions and calculate the validation loss
                prediction = unet(x)
                totalTestLoss += lossFunc(prediction, y)

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps

        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))

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
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig(config.PLOT_PATH)


if __name__ == '__main__':
    train()
