# import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer

import detectron2_format


def show(dataset_path, count):
    p = dataset_path

    DatasetCatalog.register("test_dataset",
                            lambda path=p: detectron2_format.load_detectron2_dataset(path))
    MetadataCatalog.get("test_dataset").set(thing_classes=["left_lung", "right_lung", "lung", "heart"])

    dataset = DatasetCatalog.get("test_dataset")
    metadata = MetadataCatalog.get("test_dataset")

    for i in range(0, count):
        # img = cv2.imread(dataset[0]["file_name"])
        img = np.array(Image.open(dataset[i]["file_name"]).convert("RGB"))
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(dataset[i])

        # cv2.imshow("Input", out.get_image()[:, :, ::-1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()
        plt.close()
