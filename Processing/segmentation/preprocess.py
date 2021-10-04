import os

from PIL import Image


def to_rgb_and_resize(image_folder: str, dim: tuple[int, int], output_folder: str):
    """
    Resize image to the dimensions in dim.

    :param image_folder: path to the image folder
    :param dim: width, height
    :param output_folder: path to the folder to save the images to
    """
    image_folder = os.path.abspath(image_folder)
    output_folder = os.path.abspath(output_folder)

    for file in os.listdir(image_folder):
        image = Image.open(os.path.join(image_folder, file)).convert("RGB")
        image = image.resize(dim, Image.BICUBIC)

        image.save(os.path.join(output_folder, file), format="png")

        print(f"resized image {file}")
