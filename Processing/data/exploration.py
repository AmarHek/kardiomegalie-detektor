import os
import math
import json
import matplotlib.pyplot as plt


def bbox_area_ratio(dataset_json_path, classes, save=False, output_folder="./output"):
    """
    Plots areas and ratios of the bounding boxes in the dataset.

    :param dataset_json_path: path to the json file containing the dataset
    :param classes: categories/ classes in order of the corresponding ids in the json file
    :param save: if true, save the plots
    :param output_folder: save plots to this folder
    """
    results = dict()

    for category in classes:
        results[category] = dict()
        results[category]["area"] = list()
        results[category]["ratio"] = list()

    with open(dataset_json_path, "r") as file:
        dataset = json.loads(file.read())

    for idx, img in enumerate(dataset):
        size = math.sqrt(img["height"] * img["width"])

        annotations = img["annotations"]
        for annotation in annotations:
            category = classes[annotation["category_id"]]
            bbox = annotation["bbox"]

            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]

            rel_area = math.sqrt(height*width) / size

            try:
                ratio = height/width
            except ZeroDivisionError:
                ratio = 0

            if rel_area <= 0.1:
                print(f"check suspicious image {img['file_name']}")

            results[category]["area"].append(rel_area)
            results[category]["ratio"].append(ratio)

    for category in classes:
        sorted_area = sorted(results[category]["area"])
        length_area = len(sorted_area)

        mid = int(length_area / 2)
        quarter = int(length_area / 4)

        left_median = sorted_area[quarter]
        median = sorted_area[mid]
        right_median = sorted_area[-quarter]
        print(f"{category} area\nleft median: {left_median} | median: {median} | right median: {right_median}\n")

        plt.plot([i for i in range(length_area)],
                 sorted_area,
                 label=f"{category}")

        plt.legend()
        plt.xlabel("Röntgenbilder")
        plt.ylabel("Anteil der Fläche der Boundingbox")

        if save:
            plt.savefig(os.path.join(output_folder, f"{category}_area.svg"),
                        bbox_inches="tight",
                        format="svg")

        plt.show()
        plt.close()

        sorted_ratio = sorted(results[category]["ratio"])
        length_ratio = len(sorted_ratio)
        mid = int(length_ratio / 2)
        quarter = int(length_ratio / 4)

        left_median = sorted_ratio[quarter]
        median = sorted_ratio[mid]
        right_median = sorted_ratio[-quarter]
        print(f"{category} ratio\nleft median: {left_median} | median: {median} | right median: {right_median}\n")

        plt.plot([i for i in range(length_ratio)],
                 sorted_ratio,
                 label=f"{category}")

        plt.legend()
        plt.xlabel("Röntgenbilder")
        plt.ylabel("Seitenverhätnis")

        if save:
            plt.savefig(os.path.join(output_folder, f"{category}_ratio.svg"),
                        bbox_inches="tight",
                        format="svg")

        plt.show()
        plt.close()
