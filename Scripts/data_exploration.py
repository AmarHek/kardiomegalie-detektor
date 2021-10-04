import os
import math
import json
import matplotlib.pyplot as plt

from xml.etree import ElementTree


def mean_bbox_area_ratio(dataset_json_path, categories):
    """
    Creates a json file containing the count of bounding boxes of a specific area and ratio.

    :param dataset_json_path: path to the json file containing the dataset
    :param categories: categories/ classes in order of the corresponding ids in the json file
    """
    results = dict()

    for category in categories:
        results[category] = dict()
        results[category]["area"] = dict()
        results[category]["ratio"] = dict()

    with open(dataset_json_path, "r") as file:
        dataset_json = file.read()

    dataset = json.loads(dataset_json)

    for img in dataset:
        annotations = img["annotations"]
        for annotation in annotations:
            category = categories[annotation["category_id"]]
            bbox = annotation["bbox"]

            height = bbox[3] - bbox[1]
            width = bbox[2]-bbox[0]

            area = height*width
            ratio = height/width

            results[category]["area"].setdefault(area, 0)
            results[category]["ratio"].setdefault(ratio, 0)

            results[category]["area"][area] += 1
            results[category]["ratio"][ratio] += 1

    with open("./mean_area_ratio.json", "w") as file:
        file.write(json.dumps(results, indent=4))


def visualize_mean_area_ratio(data_json_path):
    """
    Visualizes the data created through the function mean_bbox_area_ratio.

    :param data_json_path: path to the json file with the data
    """
    with open(data_json_path, "r") as file:
        data_json = file.read()

    data = json.loads(data_json)

    results = dict()
    for category in data:
        results[category] = dict()
        results[category]["area"] = dict()
        results[category]["area"]["area"] = list()
        results[category]["area"]["count"] = list()

        for area in data[category]["area"]:
            results[category]["area"]["area"].append(math.sqrt(int(area)))
            print(math.sqrt(int(area)))
            results[category]["area"]["count"].append(data[category]["area"][area])

        results[category]["ratio"] = dict()
        results[category]["ratio"]["ratio"] = list()
        results[category]["ratio"]["count"] = list()

        for ratio in data[category]["ratio"]:
            results[category]["ratio"]["ratio"].append(ratio)
            results[category]["ratio"]["count"].append(data[category]["ratio"][ratio])

    plt.bar(results["left_lung"]["area"]["area"], results["left_lung"]["area"]["count"], label="area")
    plt.bar(results["left_lung"]["ratio"]["ratio"], results["left_lung"]["ratio"]["count"], label="ratio")
    plt.legend()
    plt.show()
    plt.close()


def find_pascal_voc_difficult(bbox_folder):
    """
    Prints all pascal voc files in a folder, that contain a difficult tag.

    :param bbox_folder: path to the folder of pascal voc files
    """
    bbox_folder = os.path.abspath(bbox_folder)

    for file in os.listdir(bbox_folder):
        dom = ElementTree.parse(os.path.join(bbox_folder, file))
        objects = dom.findall("object")

        for obj in objects:
            if int(obj.find("difficult").text) == 1:
                print(f"{file} contains a difficult bbox")
                break
