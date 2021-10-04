import os
import json
import numpy as np

from typing import List, Tuple
from detectron2.evaluation import DatasetEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures.masks import BitMasks


class JsonDumpEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name: str, output_dir: str):
        self.output_dir = output_dir

        self.ground_truth = {
            dataset["image_id"]: dataset["annotations"]
            for dataset in DatasetCatalog.get(dataset_name)
        }
        self.thing_classes = MetadataCatalog.get(dataset_name).thing_classes

        self.result = dict()
        self.result["mIoU_bbox"] = dict()
        self.result["mIoU_mask"] = dict()
        self.result["mDice_bbox"] = dict()
        self.result["mDice_mask"] = dict()

        self.metrics_bbox = dict()
        self.metrics_mask = dict()

        for metric in ["iou", "dice"]:
            self.metrics_bbox[metric] = dict()
            self.metrics_mask[metric] = dict()

            for name in self.thing_classes:
                self.metrics_bbox[metric][name] = list()
                self.metrics_mask[metric][name] = list()

    def reset(self):
        self.result = dict()
        self.result["mIoU_bbox"] = dict()
        self.result["mIoU_mask"] = dict()
        self.result["mDice_bbox"] = dict()
        self.result["mDice_mask"] = dict()

        self.metrics_bbox = dict()
        self.metrics_mask = dict()

        for metric in ["iou", "dice"]:
            self.metrics_bbox[metric] = dict()
            self.metrics_mask[metric] = dict()

            for name in self.thing_classes:
                self.metrics_bbox[metric][name] = list()
                self.metrics_mask[metric][name] = list()

    def process(self, inputs, outputs):
        for metadata, output in zip(inputs, outputs):
            output = output["instances"].to("cpu")

            #
            image_id = metadata["image_id"]

            gt_bbox = dict()
            for category in self.ground_truth[image_id]:
                gt_bbox[self.thing_classes[category["category_id"]]] = category["bbox"]

            #
            idx = dict()
            classes = output.pred_classes.numpy()
            for i, name in enumerate(self.thing_classes):
                idx[name] = np.argmax(classes == i) if i in classes else None

            bbox = dict()
            boxes = output.pred_boxes.tensor.tolist()
            for name in self.thing_classes:
                bbox[name] = boxes[idx[name]] if idx[name] is not None else None

            mask = dict()
            try:
                masks = BitMasks(output.pred_masks).get_bounding_boxes().tensor.tolist()
                for name in self.thing_classes:
                    mask[name] = masks[idx[name]] if idx[name] is not None else None
            except AttributeError:
                for name in self.thing_classes:
                    mask[name] = None

            #
            self.result[image_id] = dict()
            self.result[image_id]["file_name"] = os.path.basename(metadata["file_name"])

            #
            for name in self.thing_classes:
                self.result[image_id][name] = dict()

                iou_bbox, dice_bbox = iou_dice(gt_bbox[name], bbox[name]) if bbox[name] is not None else (0, 0)
                iou_mask, dice_mask = iou_dice(gt_bbox[name], mask[name]) if mask[name] is not None else (0, 0)

                self.metrics_bbox["iou"][name].append(iou_bbox)
                self.metrics_bbox["dice"][name].append(dice_bbox)

                self.metrics_mask["iou"][name].append(iou_mask)
                self.metrics_mask["dice"][name].append(dice_mask)

                self.result[image_id][name]["bbox"] = bbox[name]
                self.result[image_id][name]["mask"] = mask[name]

    def evaluate(self):
        for name in self.thing_classes:
            self.result["mIoU_bbox"][name] = sum(self.metrics_bbox["iou"][name]) / len(self.metrics_bbox["iou"][name])
            self.result["mIoU_mask"][name] = sum(self.metrics_mask["iou"][name]) / len(self.metrics_mask["iou"][name])

            self.result["mDice_bbox"][name] = \
                sum(self.metrics_bbox["dice"][name]) / len(self.metrics_bbox["dice"][name])
            self.result["mDice_mask"][name] = \
                sum(self.metrics_mask["dice"][name]) / len(self.metrics_mask["dice"][name])

        #
        with open(os.path.join(self.output_dir, "result.json"), "w") as file:
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
