import os
import numpy as np
import mask_utils

from PIL import Image
from xml.etree import ElementTree


def rescale_mask(image_folder, mask_folder, save_folder):
    """
    Rescale masks, based on the images dimension

    :param image_folder: path to the images
    :param mask_folder: path to the masks
    :param save_folder: save new masks to
    """
    image_folder = os.path.abspath(image_folder)
    mask_folder = os.path.abspath(mask_folder)
    save_folder = os.path.abspath(save_folder)

    for file in os.listdir(mask_folder):
        img_file = file.split(".")
        img_file[0] = img_file[0].removesuffix('-')
        img_file = f"{img_file[0].removesuffix('_')}.{img_file[1]}"

        mask = Image.open(os.path.join(mask_folder, file))
        mask = mask.resize(Image.open(os.path.join(image_folder, img_file)).size, Image.NEAREST)
        mask = np.array(mask)
        mask[mask > 0] = 255
        Image.fromarray(mask).save(os.path.join(save_folder, img_file), format="png")

        print(f"resized mask {file}")


def combine_left_right_mask_folder(mask_folder_left, mask_folder_right, save_folder):
    """
    Combine to masks into one

    :param mask_folder_left: masks one
    :param mask_folder_right: masks two
    :param save_folder: save new masks to
    """
    mask_folder_left = os.path.abspath(mask_folder_left)
    mask_folder_right = os.path.abspath(mask_folder_right)
    save_folder = os.path.abspath(save_folder)

    for file in os.listdir(mask_folder_left):
        save_path = os.path.join(save_folder, file)
        save_path = save_path.removesuffix(f".{os.path.basename(save_path).split('.')[-1]}")

        mask_left = np.array(Image.open(os.path.join(mask_folder_left, file)).convert("L"))
        mask_right = np.asarray(Image.open(os.path.join(mask_folder_right, file)).convert("L"))

        mask = np.maximum(mask_left, mask_right)
        Image.fromarray(mask).save(f"{save_path}.png", format="png")

        print(f"combined mask {file}")


def rescale_bbox(bbox_folder, bbox_heart_folder, save_folder):
    """
    Rescale new heart bounding boxes, based on the old heart bounding boxes

    :param bbox_folder: path to the new bounding boxes
    :param bbox_heart_folder: path to the old bounding boxes
    :param save_folder: save new bounding boxes to
    """
    bbox_folder = os.path.abspath(bbox_folder)
    bbox_heart_folder = os.path.abspath(bbox_heart_folder)
    save_folder = os.path.abspath(save_folder)

    for file in os.listdir(bbox_heart_folder):
        with open(os.path.join(bbox_folder, file), "r") as data:
            xml = data.read()

        dom = ElementTree.parse(os.path.join(bbox_folder, file))
        size = dom.find("size")
        bbox_size = (int(size.find("width").text), int(size.find("height").text))

        #
        bbox_dict = mask_utils.read_pascal_voc_format_dict(os.path.join(bbox_heart_folder, file))

        dom = ElementTree.parse(os.path.join(bbox_heart_folder, file))
        size = dom.find("size")
        bbox_old_size = (int(size.find("width").text), int(size.find("height").text))

        #
        x_factor = bbox_size[0] / bbox_old_size[0]
        y_factor = bbox_size[1] / bbox_old_size[1]

        xml = xml.split("</annotation>")[0] + f"""
    <object>
        <name>heart</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{round(bbox_dict["heart"][0] * x_factor)}</xmin>
            <ymin>{round(bbox_dict["heart"][1] * y_factor)}</ymin>
            <xmax>{round(bbox_dict["heart"][2] * x_factor)}</xmax>
            <ymax>{round(bbox_dict["heart"][3] * y_factor)}</ymax>
        </bndbox>
    </object>
</annotation>
        """

        with open(os.path.join(save_folder, file), "w") as data:
            data.write(xml)

        print(f"resized heart bbox in {file}")

