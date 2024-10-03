from datetime import datetime
import torch
import os

initial_time = datetime.now().strftime("%Y%m%d%H%M%S")
# initial_time = "20240930182854"

# base path of the dataset
DATASET_PATH = os.path.join("..", "dataset", "unity_dataset", "mixed_visibility_dataset", "train")
# DATASET_PATH = os.path.join("..", "dataset", "unity_dataset", "mixed_visibility_dataset_320/train")
# DATASET_PATH = os.path.join("..", "dataset", "ortophoto_pretraining")
TESTSET_PATH = os.path.join("..", "dataset", "unity_dataset", "mixed_visibility_dataset", "test")

# define the path to the shadow images and shadowless images dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "train_A")
GT_DATASET_PATH = os.path.join(DATASET_PATH, "train_C")

SAVE_TRANSFORMS = False

# define the test split
EVAL_SPLIT = 0.1

# Path to the saved checkpoint
# LOAD_MODEL = None
LOAD_MODEL = os.path.join("output/output_20241002213808/unet_shadow_20241002213808_e60.pth")

# Freezes encoder layers
FINE_TUNE = True

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input
NUM_CHANNELS = 3

BATCH_NORM = True

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.0001
NUM_EPOCHS = 500
BATCH_SIZE = 8

# define the input image dimensions
INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256

# define the path to the base output directory
BASE_OUTPUT = f"./output/output_{initial_time}"

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, f"unet_shadow_{initial_time}_e{NUM_EPOCHS}.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, f"plot_{initial_time}.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, f"test_paths_{initial_time}.txt"])
