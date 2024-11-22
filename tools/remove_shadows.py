from datetime import datetime

import cv2
import numpy as np
import torch

image_path = "../DJI_20241015105147_0009_D.JPG"

with_mp = True
stride_height, stride_width = (128, 128)
tile_height, tile_width = (256, 256)


class RemoveShadows:
    def __init__(self, path_to_model):
        self.model = torch.load(path_to_model, weights_only=False).to("cuda")

    def remove_shadows(self, tile):
        input_size = tile.shape
        if (input_size[0] != tile_height) or (input_size[1] != tile_width):
            tile = np.resize(tile, (tile_height, tile_width, 3))
        tile = tile.astype(np.float32)
        tile = np.array(tile) / 255
        tile = np.transpose(tile, (2, 0, 1))
        tile = np.expand_dims(tile, 0)

        tensor = torch.from_numpy(tile).to("cuda")
        # prediction = tensor.squeeze()
        prediction = self.model(tensor).squeeze()
        prediction = prediction.cpu().detach().numpy()
        prediction = np.transpose(prediction, (1, 2, 0))
        prediction = prediction * 255
        prediction = prediction.astype(np.uint8)

        if (input_size[0] != tile_height) or (input_size[1] != tile_width):
            prediction = np.resize(prediction, input_size)
        return prediction

    def process_image(self, image):
        img_height, img_width, img_channels = image.shape

        output_image = np.zeros((img_height, img_width, img_channels), dtype=np.float32)
        weights = np.zeros((img_height, img_width, img_channels), dtype=np.float32)

        extended_height = ((img_height + tile_height - 1) // tile_height) * tile_height
        extended_width = ((img_width + tile_width - 1) // tile_width) * tile_width


        for y in range(0, extended_height - tile_height + 1, stride_height):
            for x in range(0, extended_width - tile_width + 1, stride_width):
                tile = (image[y:y + tile_height, x:x + tile_width])
                tile = self.remove_shadows(tile)

                tile_weight = np.ones((tile.shape[0], tile.shape[1], img_channels), dtype=np.float32)
                output_image[y:y + tile.shape[0], x:x + tile.shape[1]] += tile
                weights[y:y + tile.shape[0], x:x + tile.shape[1]] += tile_weight

        # To avoid division by zero, set zeros to one x/1 = x
        weights[weights == 0] = 1.0
        output_image = output_image / weights
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)

        return output_image


if __name__ == '__main__':
    image = cv2.imread(image_path)
    # remover = RemoveShadows("../unet/output/output_20241003154040/unet_shadow_20241003154040_e95.pth")
    # remover = RemoveShadows("../unet/output/output_20241003154040/unet_shadow_20241003154040_e115.pth")
    # remover = RemoveShadows("../unet/output/output_20241019231437/unet_shadow_20241019231437_e70.pth")
    remover = RemoveShadows("../unet/output/output_20241108171754/unet_shadow_20241108171754_e25.pth")
    # remover = RemoveShadows("../unet/output/output_20241111072901/unet_shadow_20241111072901_e200.pth")
    output = remover.process_image(image)
    # Save or display the result
    cv2.imwrite(f'../unet/merged_image_{datetime.now().strftime("%Y%m%d%H%M%S")}.jpg', output)
    cv2.imshow('Merged Image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
