import os
import json

from typing import List
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog, DatasetCatalog, detection_utils
from detectron2.data.transforms import Augmentation


def build_aug(cfg: CfgNode, is_train: bool,
              before_resize: List[Augmentation] = None,
              after_resize: List[Augmentation] = None) -> List[Augmentation]:
    """
    Returns augmentation.

    :param cfg: The models config
    :param is_train: Get augmentation for training or testing
    :return: List with augmentation

    :param cfg: the config
    :param is_train: augmentations for training of testing
    :param before_resize: augmentations to apply before resizing
    :param after_resize: augmentations to apply after resizing
    :return: augmentations
    """
    augmentations = detection_utils.build_augmentation(cfg, is_train)

    if is_train:
        if before_resize is not None:
            for aug in reversed(before_resize):
                augmentations.insert(0, aug)

        if after_resize is not None:
            for aug in after_resize:
                augmentations.append(aug)

    return augmentations


def load_data(dataset_name: str, path_to_dataset_json: str, classes: List[str]):
    """
    Registers the dataset in the path_to_dataset_json json file under the name dataset_name and
    sets thing_classes to classes.

    For loading the dataset, load_dataset_from_json method is used.
    Therefore cfg.INPUT.MASK_FORMAT = "bitmask" has to be set.

    :param dataset_name: register name
    :param path_to_dataset_json: path to the json file containing the dataset
    :param classes: corresponding strings to classes ids
    """
    path_to_dataset_json = os.path.abspath(path_to_dataset_json)

    DatasetCatalog.register(dataset_name,
                            lambda p=path_to_dataset_json: load_dataset_from_json(p))
    MetadataCatalog.get(dataset_name).set(thing_classes=classes)


def load_dataset_from_json(path_to_json: str) -> List[dict]:
    """
    Loads a detectron2 formatted dataset from json and encodes the rle using utf-8.
    Also sets the images location to an image folder in the json's directory

    Hint: cfg.INPUT.MASK_FORMAT = "bitmask" has to be set

    :param path_to_json: path to dataset json file
    :return: dataset in list[dict] format
    """
    path_to_json = os.path.abspath(path_to_json)

    with open(path_to_json, "r") as file:
        dataset_json = file.read()

    dataset = json.loads(dataset_json)

    path_to_json = os.path.dirname(path_to_json)
    for img in dataset:
        img["file_name"] = os.path.join(path_to_json, "images", img["file_name"])

        img["annotations"][0]["segmentation"]["counts"] = \
            img["annotations"][0]["segmentation"]["counts"].encode("utf-8")

    return dataset
