from datetime import datetime
import torch
import os

initial_time = datetime.now().strftime("%Y%m%d%H%M%S")
# initial_time = "20240930182854"

DATASET = "pretraining"


DATASET_PATH = None
IMAGE_DATASET_PATH = ""
GT_DATASET_PATH = ""
TESTSET_T_PATH = ""
TESTSET_GT_PATH = ""
MEAN = []
STD = []
TRANSFORMS = []
LOAD_MODEL = None
FINE_TUNE = False
LOSS_FUNCTION = ''
INIT_LR = 0
DROPOUT = 0
START_EPOCH =0
NUM_EPOCHS = 0
BATCH_SIZE = 0
WEIGHT_DECAY = 0

if DATASET == "pretraining":
    # DATASET_PATH = os.path.join("..", "dataset", "ortophoto_pretraining_small")
    DATASET_PATH = os.path.join("..", "dataset", "ortophoto_pretraining")
    IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "train_A")
    GT_DATASET_PATH = os.path.join(DATASET_PATH, "train_A") # It is the same dataset
    TESTSET_T_PATH = os.path.join("..", "dataset", "ortophoto_pretraining", "train_A", "*.png")
    TESTSET_GT_PATH = os.path.join("..", "dataset", "SRD", "SRD_Test", "SRD", "shadow_free", "*.jpg")

    MEAN=[0.3690, 0.4038, 0.3678] # pretraining t dataset
    STD=[0.1869, 0.1622, 0.1470] # pretraining t dataset
    # MEAN=[0.3690, 0.4038, 0.3678] # pretraining gt dataset
    # STD=[0.1869, 0.1622, 0.1470] # pretraining gt dataset
    # MEAN=[0.3690, 0.4038, 0.3678] # pretraining dataset
    # STD=[0.1869, 0.1622, 0.1470] # pretraining dataset

    TRANSFORMS = [
            'RandomResizedCrop',
            # 'ColorJitter',
            'RandomHorizontalFlip'
            'RandomVerticalFlip',
            'RandomRotation',
        ]

    LOAD_MODEL = None
    FINE_TUNE = False

    LOSS_FUNCTION = 'MSE'
    INIT_LR = 1e-3
    DROPOUT = 0.0
    START_EPOCH = 0
    NUM_EPOCHS = 50
    BATCH_SIZE = 8
    WEIGHT_DECAY = 1e-5
else:
    if DATASET == "SRD":
        DATASET_PATH = os.path.join("..", "dataset", "SRD", "SRD_Train", "Train")
        IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "shadow")
        GT_DATASET_PATH = os.path.join(DATASET_PATH, "shadow_free")
        print('no mean/std')
    elif DATASET == "ISTD":
        DATASET_PATH = os.path.join("..", "dataset", "ISTD_Dataset", "train")
        TESTSET_PATH = os.path.join("..", "dataset", "ISTD_Dataset", "test")

        # define the path to the shadow images and shadowless images dataset
        IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "train_A")
        GT_DATASET_PATH = os.path.join(DATASET_PATH, "train_C")

        MEAN = [0.5491, 0.5430, 0.5157]  # ISTD dataset
        STD = [0.1675, 0.1555, 0.1709]  # ISTD dataset
    elif DATASET == "USOS":
        DATASET_PATH = os.path.join("..", "dataset", "unity_dataset", "usos", "train")
        DATASET_PATH = os.path.join("..", "dataset", "unity_dataset", "usos_small", "train_small")
        DATASET_PATH = os.path.join("..", "dataset", "unity_dataset", "usos_xs", "train")
        DATASET_PATH = os.path.join("..", "dataset", "unity_dataset", "usos_single", "train")
        TESTSET_PATH = os.path.join("..", "dataset", "unity_dataset", "mixed_visibility_dataset", "test")

        # define the path to the shadow images and shadowless images dataset
        IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "train_A")
        GT_DATASET_PATH = os.path.join(DATASET_PATH, "train_C")

        # MEAN=[0.4880, 0.4950, 0.3880] # USOS t+gt dataset
        # STD=[0.3099, 0.2588, 0.2466] # USOS t+gt dataset
        MEAN=[0.4179, 0.4282, 0.3439] # USOS t dataset
        STD=[0.3274, 0.2811, 0.2549] # USOS t dataset
        # MEAN=[0.5580, 0.5620, 0.4327] # USOS gt dataset
        # STD=[0.2741, 0.2139, 0.2294] # USOS gt dataset
    else:
        MEAN=[0.485, 0.456, 0.406] # ImageNet dataset
        STD=[0.229, 0.224, 0.225] # ImageNet dataset

    TRANSFORMS = [
            'RandomResizedCrop',
            # 'ColorJitter',
            'RandomHorizontalFlip'
            'RandomVerticalFlip',
            'RandomRotation',
        ]

    # Path to the saved checkpoint
    # LOAD_MODEL = None
    # LOAD_MODEL = os.path.join("output/output_usos_20250801222705/unet_shadow_20250801222705_e200.pth")
    # LOAD_MODEL = os.path.join("output/output_20241024223406/unet_shadow_20241024223406_e100.pth")
    # LOAD_MODEL = os.path.join("output/output_20241108171754/unet_shadow_20241108171754.pth")
    # LOAD_MODEL = os.path.join("output/output_pretraining_20250805193927/unet_shadow_20250805193927_e175.pth")
    # LOAD_MODEL = os.path.join("output/output_pretraining_20250902193310/unet_shadow_20250902193310_e105.pth")
    LOAD_MODEL = os.path.join("output/output_pretraining_20250920121910/unet_shadow_20250920121910_e50.pth")
    # LOAD_MODEL = os.path.join("output/output_usos_20250921214439/unet_shadow_20250921214439_e50.pth")

    # Freezes encoder layers
    FINE_TUNE = True

    LOSS_FUNCTION = 'L1_SSIM'
    INIT_LR = 1e-5
    DROPOUT = 0.1
    START_EPOCH = 0
    NUM_EPOCHS = 150
    BATCH_SIZE = 8
    WEIGHT_DECAY = 1e-6


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

MULTI_GPU = False
# MULTI_GPU = True

# determine the device to be used for training and evaluation
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input
NUM_CHANNELS = 3

BATCH_NORM = True

# define the input image dimensions
INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256

# define the path to the base output directory
BASE_OUTPUT = f"./output/output_{DATASET}_{initial_time}"

# define the path to the output serialized model, model training
# plot, and testing image paths
# MODEL_PATH = os.path.join(BASE_OUTPUT, f"unet_shadow_{initial_time}_{DATASET}_e{NUM_EPOCHS}.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, f"plot_{DATASET}_{initial_time}.png"])
