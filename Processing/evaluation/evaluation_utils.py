import json
import math

from typing import Optional, Union


def results_iterator(results_path: Optional[str], ground_truth_path: str):
    """
    This generator returns result and ground truth data for each item in the ground truth.

    :param results_path: path to the result.json file (optional)
    :param ground_truth_path: path to the json file containing the dataset (detectron2 format)
    :return: index, name, result data, ground truth data
    """
    with open(ground_truth_path, "r") as file:
        ground_truth_raw = json.loads(file.read())

    total_images = len(ground_truth_raw)

    ground_truth = {
        image["image_id"]: image
        for image in ground_truth_raw
    }

    if results_path is not None:
        with open(results_path, "r") as file:
            data = json.loads(file.read())

    for idx in range(total_images):
        yield idx, ground_truth[idx]["file_name"], \
              data[str(idx)] if results_path is not None else None, \
              ground_truth[idx]


def iou_dice(bbox1: list[int], bbox2: list[int]) -> tuple[float, float]:
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


def htq_card(card_threshold: float,
             bbox_heart: list[int],
             bbox_lung: list[int],
             bbox_right_lung: list[int] = None) -> tuple[float, Optional[bool]]:
    """
    Calculates the HTQ from a heart and a lung bounding box, which must not be None.

    :param card_threshold: cardiomegaly is true, if htq > card_threshold
    :param bbox_heart: heart bounding box
    :param bbox_lung: lung bounding box or left lung bounding box
    :param bbox_right_lung: optional right lung bounding box
    :return: htq, cardiomegaly
    """
    heart_diameter = (bbox_heart[2] - bbox_heart[0]) if bbox_heart is not None else 0
    try:
        htq = heart_diameter / lung_diameter(bbox_lung, bbox_right_lung)
    except ZeroDivisionError:
        htq = 0

    # difference of manual htq to automatic from ground truth 0.984
    # htq = htq * 0.99

    return htq, htq > card_threshold if htq != 0 else None


def lung_diameter(bbox_lung: list[int], bbox_right_lung: list[int] = None) -> float:
    """
    Calculates the lung diameter (internal diameter) from a heart and a lung bounding box.

    A None bounding box is set to [0, 0, 0, 0]

    :param bbox_lung: lung bounding box or left lung bounding box
    :param bbox_right_lung: optional right lung bounding box
    :return: lung diameter
    """
    if bbox_right_lung is None:
        if bbox_lung is None:
            ld = 0
        else:
            ld = bbox_lung[2] - bbox_lung[0]

    elif bbox_lung is None:
        ld = bbox_right_lung[2] - bbox_right_lung[0]
    else:
        ld = bbox_right_lung[2] - bbox_lung[0]

    return ld


def mld_mrd(gt_bbox_lung: list[int], bbox_heart: list[int]) -> tuple[float, float]:
    """
    Calculates MLD and MRD from a ground truth lung bounding box and a heart bounding box.

    :param gt_bbox_lung: ground truth lung bounding box
    :param bbox_heart: heart bounding box
    :return: mld and mrd
    """
    if bbox_heart is None:
        return 0, 0

    midline = (gt_bbox_lung[0] + gt_bbox_lung[2]) / 2

    mrd = midline - bbox_heart[0]
    mld = bbox_heart[2] - midline

    return mld, mrd


def difference(ground_truth: list[Union[int, float]], result: list[Union[int, float]]) -> tuple[float, float]:
    """
    Calculate the mean difference between the input lists and its standard deviation.\n
    (result - ground truth)

    :param ground_truth: ground truth values
    :param result: result values
    :return: mean and standard deviation
    """
    diff = [result[i] - ground_truth[i] for i in range(0, len(result))]

    mean = sum(diff) / len(diff)
    var = sum([(d - mean)**2 for d in diff]) / (len(diff) - 1)

    return mean, math.sqrt(var)


def correlation_coefficient(x: list[float], y: list[float]) -> Optional[float]:
    """
    Calculates the correlation coefficient of the two input lists x and y.

    :param x: list with data points
    :param y: another list of data points corresponding to x
    :return: correlation coefficient
    """
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    top = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(0, len(x))])

    dev_x = sum([(val - mean_x)**2 for val in x])
    dev_y = sum([(val - mean_y)**2 for val in y])

    bottom = math.sqrt(dev_x * dev_y)

    return (top / bottom) if bottom != 0 else None
