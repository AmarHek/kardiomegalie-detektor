import os
import json
import numpy as np

from typing import List, Tuple
from PIL import Image


class JsonDumpSegEvaluator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

        self.thing_classes = ["left_lung", "right_lung", "lung", "heart"]

        self.result = dict()
        self.result["mIoU_bbox"] = dict()
        self.result["mIoU_mask"] = dict()
        self.result["mDice_bbox"] = dict()
        self.result["mDice_mask"] = dict()

        self.metrics = dict()

        for metric in ["iou", "dice"]:
            self.metrics[metric] = dict()

            for name in self.thing_classes:
                self.metrics[metric][name] = list()

    def reset(self):
        self.result = dict()
        self.result["mIoU_bbox"] = dict()
        self.result["mIoU_mask"] = dict()
        self.result["mDice_bbox"] = dict()
        self.result["mDice_mask"] = dict()

        self.metrics = dict()

        for metric in ["iou", "dice"]:
            self.metrics[metric] = dict()

            for name in self.thing_classes:
                self.metrics[metric][name] = list()

    def process(self, idx, file_name, ground_truth, lung_box, heart_box):
        image_id = idx

        #
        gt_bbox = dict()
        for category in ground_truth:
            gt_bbox[self.thing_classes[category["category_id"]]] = category["bbox"]

        bbox = dict()
        bbox["left_lung"] = None
        bbox["right_lung"] = None
        bbox["lung"] = lung_box
        bbox["heart"] = heart_box

        #
        self.result[image_id] = dict()
        self.result[image_id]["file_name"] = os.path.basename(file_name)

        #
        for name in self.thing_classes:
            self.result[image_id][name] = dict()

            iou_bbox, dice_bbox = iou_dice(gt_bbox[name], bbox[name]) if bbox[name] is not None else (0, 0)

            self.metrics["iou"][name].append(iou_bbox)
            self.metrics["dice"][name].append(dice_bbox)

            self.result[image_id][name]["mask"] = bbox[name]
            self.result[image_id][name]["bbox"] = bbox[name]

    def evaluate(self):
        for name in self.thing_classes:
            self.result["mIoU_mask"][name] = sum(self.metrics["iou"][name]) / len(self.metrics["iou"][name])

            self.result["mDice_mask"][name] = \
                sum(self.metrics["dice"][name]) / len(self.metrics["dice"][name])

        #
        with open(os.path.join(self.output_dir, "seg_result.json"), "w") as file:
            file.write(json.dumps(self.result, indent=4))


def iou_dice(bbox1: List[int], bbox2: List[int]) -> Tuple[float, float]:
    """
    Calculates the intersection over union and the dice-score of two bounding boxes

    :param bbox1: bounding box one
    :param bbox2: bounding box two
    :return: intersection over union, dice-score
    """
    x_min = max(bbox1[0], bbox2[0])
    y_min = max(bbox1[1], bbox2[1])
    x_max = min(bbox1[2], bbox2[2])
    y_max = min(bbox1[3], bbox2[3])

    intersection = max(x_max - x_min, 0) * max(y_max - y_min, 0)

    gt_area = abs((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
    area = abs((bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]))

    return intersection / (gt_area + area - intersection), 2 * intersection / (gt_area + area)


def get_bbox(mask: np.ndarray) -> list[int]:
    """
    Takes a 2D array and calculates a bounding box around all nonzero values.

    :param mask: 2D array
    :return: bounding box in XYXY
    """
    mask = np.asarray(mask)

    y = np.nonzero(mask)

    ymin = y[0][0]
    ymax = y[0][-1]

    x = np.nonzero(np.rot90(mask, 3))
    xmin = x[0][0]
    xmax = x[0][-1]

    return [int(xmin), int(ymin), int(xmax), int(ymax)]


def segmentation_results(lungs_folder: str, hearts_folder: str, ground_truth_path: str, output_folder: str):
    """
    Processes the results from Nico's segmentation model like the JsonDumpEvaluator.

    :param lungs_folder: path to the folder with the lung masks
    :param hearts_folder: path to the folder with the heart masks
    :param ground_truth_path: the corresponding dataset in detectron2's format
    :param output_folder: folder to save result.json to
    """
    with open(ground_truth_path, "r") as file:
        ground_truth_raw = json.loads(file.read())

    total_images = len(ground_truth_raw)

    ground_truth = {
        image["image_id"]: image
        for image in ground_truth_raw
    }

    evaluator = JsonDumpSegEvaluator(output_folder)
    evaluator.reset()

    for idx in range(total_images):
        lung_mask = Image.open(os.path.join(lungs_folder, ground_truth[idx]["file_name"]))
        lung_mask.resize((ground_truth[idx]["width"], ground_truth[idx]["height"]))
        lung_mask = np.asarray(lung_mask.convert("RGB"))

        heart_mask = Image.open(os.path.join(hearts_folder, ground_truth[idx]["file_name"]))
        heart_mask.resize((ground_truth[idx]["width"], ground_truth[idx]["height"]))
        heart_mask = np.asarray(heart_mask.convert("RGB"))

        lung_box = get_bbox(lung_mask)
        heart_box = get_bbox(heart_mask)

        evaluator.process(idx, ground_truth[idx]["file_name"], ground_truth[idx]["annotations"], lung_box, heart_box)

        print(f"processed {ground_truth[idx]['file_name']}")

    evaluator.evaluate()
