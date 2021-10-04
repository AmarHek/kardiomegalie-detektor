import os
import json
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from detectron2.utils.visualizer import Visualizer


def visualise_ids(ids: list[int],
                  mode: str,
                  classes: list[str],
                  result_path: str,
                  image_folder: str,
                  save=False, output_path="./vis_out"):
    """
    Visualises the bounding boxes of some images from the output of a JsonDumpEvaluator.

    :param ids: image ids to visualise
    :param mode: "bbox" or "mask" (bounding boxes from masks)
    :param classes: list of classes to visualise (None for a class not to visualise)
    :param result_path: path to the output json file
    :param image_folder: folder with the images
    :param save: if true, save the image
    :param output_path: save images to path (*'id'.png)
    """
    with open(result_path, "r") as file:
        instances = json.loads(file.read())

    metadata = {"thing_classes": classes}

    for idx in ids:
        idx = str(idx)

        image = np.asarray(Image.open(os.path.join(image_folder, instances[idx]["file_name"])).convert("RGB"))
        visualizer = Visualizer(image, metadata=metadata)   # scale=0.5

        draw_dict = dict()
        draw_dict["height"], draw_dict["width"] = image.shape[:2]
        draw_dict["annotations"] = list()

        for cls, name in enumerate(metadata["thing_classes"]):
            if name is None:
                continue

            outputs = dict()

            outputs["bbox"] = instances[idx][name][mode]
            outputs["bbox_mode"] = 0
            outputs["category_id"] = cls

            draw_dict["annotations"].append(outputs)

        visualizer = visualizer.draw_dataset_dict(draw_dict)

        plt.imshow(visualizer.get_image())

        if save:
            plt.savefig(f"{output_path}{idx}.png",
                        bbox_inches="tight",
                        format="png",
                        dpi=1200)

        plt.show()
        plt.close()


def visualise_metric(metrics: list[str],
                     metrics_paths: list[str],
                     labels: list[str],
                     save=False, output_path="./metric_plot"):
    """
    Draw a plot from the file containing the training metrics.

    :param metrics: which metrics to include in the plot
    :param metrics_paths: path to the json file containing the metrics
    :param labels: corresponding labels to the metrics_path for the plot legend
    :param save: if true, save the plot
    :param output_path: save plot to path (*.svg)
    """
    iteration = dict()
    values = dict()
    for path in metrics_paths:
        iteration[path] = dict()
        values[path] = dict()

        for metric in metrics:
            iteration[path][metric] = list()
            values[path][metric] = list()

    for path in metrics_paths:
        with open(path, "r") as file:
            data = file.read()

        for line in data.split("\n"):
            if line != "":
                info: dict = json.loads(line)

                for metric in metrics:
                    if info.get(metric) is not None:
                        iteration[path][metric].append(info["iteration"])
                        values[path][metric].append(info[metric])

    for idx, path in enumerate(metrics_paths):
        for metric in metrics:
            plt.plot(iteration[path][metric], values[path][metric],
                     label=f"{labels[idx]} - {metric}")

    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")

    if save:
        plt.savefig(f"{output_path}.svg",
                    bbox_inches="tight",
                    format="svg")

    plt.show()
    plt.close()
