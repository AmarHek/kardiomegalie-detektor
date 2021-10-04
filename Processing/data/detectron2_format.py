import os
import numpy as np
import json
import data.mask_utils as mask_utils

from PIL import Image


def create_detectron2_dataset(img_folder: str, bbox_folders: list[str],
                              lung_mask_folders: list[str], heart_mask_folders: list[str]):
    """
    Creates a detectron2 formatted dataset json file.

    :param img_folder: path to the images
    :param bbox_folders: paths to the bounding boxes
    :param lung_mask_folders: paths to the lung masks
    :param heart_mask_folders: paths to the heart masks
    """
    image_folder = os.path.abspath(img_folder)
    bbox_folders = [os.path.abspath(f) for f in bbox_folders]
    lung_mask_folders = [os.path.abspath(f) for f in lung_mask_folders]
    heart_mask_folders = [os.path.abspath(f) for f in heart_mask_folders]

    dataset_json = list()

    image_number = 0
    for file in os.listdir(image_folder):
        dataset_json.append(
            create_image_entry(
                image_number,
                os.path.join(image_folder, file),
                [os.path.join(f, f"{file.split('.')[0]}.xml") for f in bbox_folders],
                [os.path.join(f, file) for f in lung_mask_folders],
                [os.path.join(f, file) for f in heart_mask_folders]
            )
        )
        image_number += 1

        print(f"processed {file}")

    with open("dataset.json", "w") as file:
        file.write(json.dumps(dataset_json))


def create_image_entry(image_id: int, image_path: str, bbox_paths: list[str],
                       lung_mask_paths: list[str], heart_mask_paths: list[str]):
    """
    Creates an image entry for a detectron2 formatted dataset.

    :param image_id: image_id of the entry
    :param image_path: path to the image
    :param bbox_paths: paths to the bounding boxes
    :param lung_mask_paths: paths to the lung mask
    :param heart_mask_paths: paths to the heart mask
    :return: dict of the entry
    """
    image_json = dict()

    height, width = np.asarray(Image.open(image_path).convert("L")).shape

    image_json["file_name"] = os.path.basename(image_path)
    image_json["height"] = height
    image_json["width"] = width
    image_json["image_id"] = image_id
    image_json["annotations"] = list()

    #
    bbox_path = [p for p in bbox_paths if os.path.isfile(p)][0]
    lung_mask_path = [p for p in lung_mask_paths if os.path.isfile(p)][0]
    heart_mask_path = [p for p in heart_mask_paths if os.path.isfile(p)][0]

    bbox = mask_utils.read_pascal_voc_format_dict(bbox_path)

    lr_lung_bbox = [bbox["left_lung"], bbox["right_lung"]]

    y_min = min(lr_lung_bbox[0][1], lr_lung_bbox[1][1])
    y_max = max(lr_lung_bbox[0][3], lr_lung_bbox[1][3])
    lung_bbox = [lr_lung_bbox[0][0], y_min, lr_lung_bbox[1][2], y_max]

    heart_bbox = bbox["heart"]

    #
    factor = 1
    if os.path.basename(image_path).startswith("JPC"):
        factor = 2

    left_lung_seg = mask_utils.get_rle(lung_mask_path, factor=factor, bbox=lr_lung_bbox[0])
    right_lung_seg = mask_utils.get_rle(lung_mask_path, factor=factor, bbox=lr_lung_bbox[1])
    lung_seg = mask_utils.get_rle(lung_mask_path, factor=factor, bbox=lung_bbox)

    heart_seg = mask_utils.get_rle(heart_mask_path, factor=factor, bbox=heart_bbox)

    """
    left lung
    """
    image_json["annotations"].append(dict())

    image_json["annotations"][0]["bbox"] = lr_lung_bbox[0]
    image_json["annotations"][0]["bbox_mode"] = 0
    image_json["annotations"][0]["category_id"] = 0
    # image_json["annotations"][0]["segmentation"] = [polygons, polygons, ...]
    image_json["annotations"][0]["segmentation"] = left_lung_seg

    """
    right lung
    """
    image_json["annotations"].append(dict())

    image_json["annotations"][1]["bbox"] = lr_lung_bbox[1]
    image_json["annotations"][1]["bbox_mode"] = 0
    image_json["annotations"][1]["category_id"] = 1
    image_json["annotations"][1]["segmentation"] = right_lung_seg

    """
    lung
    """
    image_json["annotations"].append(dict())

    image_json["annotations"][2]["bbox"] = lung_bbox
    image_json["annotations"][2]["bbox_mode"] = 0
    image_json["annotations"][2]["category_id"] = 2
    image_json["annotations"][2]["segmentation"] = lung_seg

    """
    heart
    """
    image_json["annotations"].append(dict())

    image_json["annotations"][3]["bbox"] = heart_bbox
    image_json["annotations"][3]["bbox_mode"] = 0
    image_json["annotations"][3]["category_id"] = 3
    image_json["annotations"][3]["segmentation"] = heart_seg

    return image_json


def create_detectron2_dataset_template(img_folder: str, bbox_folders: list[str], mask_folders: list[list[str]]):
    """
    Creates a detectron2 formatted dataset json file.

    :param img_folder: path to the images
    :param bbox_folders: paths to the bounding boxes
    :param mask_folders: paths to different mask types
    """
    image_folder = os.path.abspath(img_folder)
    bbox_folders = [os.path.abspath(f) for f in bbox_folders]
    mask_folders = [[os.path.abspath(f) for f in type_folders] for type_folders in mask_folders]

    dataset_json = list()

    image_number = 0
    for file in os.listdir(image_folder):
        dataset_json.append(
            create_image_entry_template(
                image_number,
                os.path.join(image_folder, file),
                [os.path.join(f, f"{file.split('.')[0]}.xml") for f in bbox_folders],
                [[os.path.join(f, file) for f in type_folders] for type_folders in mask_folders]
            )
        )
        image_number += 1

        print(f"processed {file}")

    with open("dataset.json", "w") as file:
        file.write(json.dumps(dataset_json))


def create_image_entry_template(image_id: int, image_path: str, bbox_paths: list[str], mask_paths: list[list[str]]):
    """
    Creates an image entry for a detectron2 formatted dataset.
    Must be implemented for specific dataset.

    :param image_id: image_id of the entry
    :param image_path: path to the image
    :param bbox_paths: paths to the bounding boxes
    :param mask_paths: paths to the different mask types
    :return: dict of the entry
    """
    raise NotImplementedError
