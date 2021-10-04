import os
import numpy as np
import json
import mask_utils

from PIL import Image


def load_detectron2_dataset(data_path):
    """
    Loads a detectron2 formatted dataset from json and encodes the rle using utf-8.

    Hint: cfg.INPUT.MASK_FORMAT = "bitmask" - has to be set, if run length encoding is used

    :param data_path: path to dataset json file
    :return: dataset in list[dict] format
    """
    data_path = os.path.abspath(data_path)

    with open(data_path, "r") as file:
        dataset_json = file.read()

    dataset = json.loads(dataset_json)

    data_path = os.path.dirname(data_path)
    for img in dataset:
        img["file_name"] = os.path.join(data_path, img["file_name"][2:])

        img["annotations"][0]["segmentation"]["counts"] = \
            img["annotations"][0]["segmentation"]["counts"].encode("utf-8")

    return dataset


def create_detectron2_dataset(img_folder, image_path_from_json, bbox_folder, lung_mask_folder, heart_mask_folder):
    """
    Creates a detectron2 formatted dataset.

    :param img_folder: path to the images
    :param image_path_from_json: path to the images from the dataset json file
    :param bbox_folder: path to the bounding boxes
    :param lung_mask_folder: path to the lung masks
    :param heart_mask_folder: path to the heart masks
    """
    dataset_json = list()
    image_folder = os.path.abspath(img_folder)
    bbox_folder = os.path.abspath(bbox_folder)
    lung_mask_folder = os.path.abspath(lung_mask_folder)
    heart_mask_folder = os.path.abspath(heart_mask_folder)

    image_number = 0
    for file in os.listdir(image_folder):
        dataset_json.append(
            create_image_entry(
                image_number,
                os.path.join(image_folder, file),
                os.path.join(image_path_from_json, file),
                os.path.join(bbox_folder, f"{file.split('.')[0]}.xml"),
                os.path.join(lung_mask_folder, file),
                os.path.join(heart_mask_folder, file)
            )
        )
        image_number += 1

        print(f"processed {file}")

    with open("test_data/test_raw/detectron2Formatted.json", "w") as file:
        file.write(json.dumps(dataset_json, indent=4))


def create_image_entry(image_id, image_path, image_path_from_json, bbox_path, lung_mask_path, heart_mask_path):
    """
    Creates an image entry for a detectron2 formatted dataset.

    :param image_id: image_id of the entry
    :param image_path: path to the image
    :param image_path_from_json: path to the image from the dataset json file
    :param bbox_path: path to the bounding boxes
    :param lung_mask_path: path to the lung mask
    :param heart_mask_path: path to the heart mask
    :return: dict of the entry
    """
    image_json = dict()

    height, width = np.asarray(Image.open(image_path).convert("L")).shape

    image_json["file_name"] = image_path_from_json
    image_json["height"] = height
    image_json["width"] = width
    image_json["image_id"] = image_id
    image_json["annotations"] = list()

    if os.path.isfile(bbox_path):
        bbox = mask_utils.read_pascal_voc_format_dict(bbox_path)
        lung_bbox = [bbox["left_lung"], bbox["right_lung"]]
        heart_bbox = [bbox["heart"]]
    else:
        lung_bbox = mask_utils.get_bbox(lung_mask_path)
        heart_bbox = mask_utils.get_bbox(heart_mask_path)

    factor = 1
    if os.path.basename(image_path).startswith("JPC"):
        factor = 2

    lung_seg = mask_utils.get_individual_rle(lung_mask_path, factor=factor)
    heart_seg = mask_utils.get_rle(heart_mask_path, factor=factor)

    """
    left lung
    """
    image_json["annotations"].append(dict())

    image_json["annotations"][0]["bbox"] = lung_bbox[0]
    image_json["annotations"][0]["bbox_mode"] = 0
    image_json["annotations"][0]["category_id"] = 0
    # image_json["annotations"][0]["segmentation"] = [polygons, polygons, ...]
    image_json["annotations"][0]["segmentation"] = lung_seg[0]

    """
    right lung
    """
    image_json["annotations"].append(dict())

    image_json["annotations"][1]["bbox"] = lung_bbox[1]
    image_json["annotations"][1]["bbox_mode"] = 0
    image_json["annotations"][1]["category_id"] = 1
    image_json["annotations"][1]["segmentation"] = lung_seg[1]

    """
    lung
    """
    image_json["annotations"].append(dict())

    image_json["annotations"][2]["bbox"] = [lung_bbox[0][0], lung_bbox[0][1], lung_bbox[1][2], lung_bbox[1][3]]
    image_json["annotations"][2]["bbox_mode"] = 0
    image_json["annotations"][2]["category_id"] = 2
    image_json["annotations"][2]["segmentation"] = lung_seg[2]

    """
    heart
    """
    image_json["annotations"].append(dict())

    image_json["annotations"][3]["bbox"] = heart_bbox[0]
    image_json["annotations"][3]["bbox_mode"] = 0
    image_json["annotations"][3]["category_id"] = 3
    image_json["annotations"][3]["segmentation"] = heart_seg

    return image_json
