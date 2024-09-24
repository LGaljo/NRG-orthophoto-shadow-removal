import glob
import os.path
import concurrent.futures

import numpy as np
from PIL import Image
from tqdm import tqdm


subset_path = "C:\\Users\\lukag\\Documents\\Projects\\Faks\\MAG\\ortophoto\\"
selector = "*.tif"

with_mp = True

image_normal = '../dataset/ortophoto_pretraining/train_C'
image_noise = '../dataset/ortophoto_pretraining/train_A'

tile_size = 150
resize_to = 128

def process_image(image_path):
    filename = os.path.basename(image_path).split('.')[0]
    print(filename)

    image = Image.open(image_path)
    image = np.array(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x][y].shape[0] == 4 and image[x][y][3] == 0:
                image[x][y] = [255, 255, 255, 255]
            elif image[x][y].shape[0] == 3 and sum(image[x][y]) == 0:
                image[x][y] = [255, 255, 255]

    if image.shape[2] == 4:
        image = Image.fromarray(image.astype(np.uint8), 'RGBA')
    elif image.shape[2] == 3:
        image = Image.fromarray(image.astype(np.uint8), 'RGB')

    image.save(f"{subset_path}fixed/{filename}.tif", "TIFF")


if __name__ == '__main__':
    if not os.path.exists(image_noise):
        os.mkdir(image_noise)
    if not os.path.exists(image_normal):
        os.mkdir(image_normal)

    subset_images = glob.glob(subset_path + "original\\" + selector)
    done_images = glob.glob(subset_path + "fixed\\" + selector)

    todo = []

    for element in [os.path.basename(img) for img in subset_images]:
        if element not in [os.path.basename(img) for img in done_images]:
            todo.append(subset_path + "original\\" + element)

    # todo = ["C:\\Users\\lukag\\Documents\\Projects\\Faks\\MAG\\ortophoto\\original\\DOF5-20240920-E0717.tif"]

    if with_mp:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(process_image, todo), total=len(todo)))
    else:
        for image in tqdm(todo, total=len(todo)):
            process_image(image)