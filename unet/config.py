from datetime import datetime
import torch
import os

initial_time = datetime.now().strftime("%Y%m%d%H%M%S")
# initial_time = "20240930182854"

# base path of the dataset
DATASET_PATH = os.path.join("..", "dataset", "unity_dataset", "mixed_visibility_dataset", "train")
# DATASET_PATH = os.path.join("..", "dataset", "unity_dataset", "mixed_visibility_dataset", "train_small")
# TESTSET_PATH = os.path.join("..", "dataset", "unity_dataset", "mixed_visibility_dataset", "test")
# DATASET_PATH = os.path.join("..", "dataset", "SRD", "SRD_Train", "Train")
# TESTSET_T_PATH = os.path.join("..", "dataset/unity_dataset/mixed_visibility_dataset/test/train_A", "*.png")
TESTSET_T_PATH = os.path.join("..", "dataset", "ortophoto_pretraining", "train_A", "*.png")
TESTSET_GT_PATH = os.path.join("..", "dataset", "SRD", "SRD_Test", "SRD", "shadow_free", "*.jpg")
# DATASET_PATH = os.path.join("..", "dataset", "ortophoto_pretraining")
# TESTSET_PATH = os.path.join("..", "dataset", "ortophoto_pretraining")

# MEAN=[0.485, 0.456, 0.406] # ImageNet dataset
# STD=[0.229, 0.224, 0.225] # ImageNet dataset
MEAN=[0.4880, 0.4950, 0.3880] # USOS dataset
STD=[0.3099, 0.2588, 0.2466] # USOS dataset

# define the path to the shadow images and shadowless images dataset
# IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "shadow")
# GT_DATASET_PATH = os.path.join(DATASET_PATH, "shadow_free")

IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "train_A")
GT_DATASET_PATH = os.path.join(DATASET_PATH, "train_C")

IMAGE_DATASET_PATHS = [
    IMAGE_DATASET_PATH,
    # "C:\\Users\\lukag\\Documents\\Projects\\Faks\\MAG\\datasets\\SRD_Train\\Train\\shadow"
]

GT_DATASET_PATHS = [
    GT_DATASET_PATH,
    # "C:\\Users\\lukag\\Documents\\Projects\\Faks\\MAG\\datasets\\SRD_Train\\Train\\shadow_free"
]

SAVE_TRANSFORMS = False

# define the test split
EVAL_SPLIT = 0.1

# Path to the saved checkpoint
# LOAD_MODEL = None
LOAD_MODEL = os.path.join("output/output_usos_20250801222705/unet_shadow_20250801222705_e200.pth")
# LOAD_MODEL = os.path.join("output/output_20241024223406/unet_shadow_20241024223406_e100.pth")
# LOAD_MODEL = os.path.join("output/output_20241108171754/unet_shadow_20241108171754.pth")

# Freezes encoder layers
FINE_TUNE = True
# FINE_TUNE = False

MULTI_GPU = False
# MULTI_GPU = True

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input
NUM_CHANNELS = 3

BATCH_NORM = True

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 1e-6
DROPOUT = 0.5
START_EPOCH = 0
NUM_EPOCHS = 200
BATCH_SIZE = 10
WEIGHT_DECAY = 1e-4

# define the input image dimensions
INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256

# define the path to the base output directory
DATASET = "usos"
BASE_OUTPUT = f"./output/output_{DATASET}_{initial_time}"

# define the path to the output serialized model, model training
# plot, and testing image paths
# MODEL_PATH = os.path.join(BASE_OUTPUT, f"unet_shadow_{initial_time}_{DATASET}_e{NUM_EPOCHS}.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, f"plot_{DATASET}_{initial_time}.png"])
