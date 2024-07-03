import os
from glob import glob
import shutil

if __name__ == "__main__":
    if not os.path.exists("../img/merged/tiffs/tiles/select_A"):
        os.mkdir("../img/merged/tiffs/tiles/select_A")
    if not os.path.exists("../img/merged/tiffs/tiles/select_C"):
        os.mkdir("../img/merged/tiffs/tiles/select_C")
    if not os.path.exists("../img/merged/tiffs/tiles/train_A"):
        os.mkdir("../img/merged/tiffs/tiles/train_A")
    if not os.path.exists("../img/merged/tiffs/tiles/train_C"):
        os.mkdir("../img/merged/tiffs/tiles/train_C")

    images = glob("../img/merged/tiffs/tiles/select_C/*.tif")
    for image_path in images:
        shutil.copy(image_path.replace('select_C', 'train_A'), image_path.replace('select_C', 'select_A'))
