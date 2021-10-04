import numpy as np
import matplotlib.pyplot as plt
import detectron2.data.transforms as T

from typing import List
from PIL import Image
from pycocotools.mask import encode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.utils.visualizer import Visualizer

import train
import card_detection.mapper


def vis_dataset(indexes: List[int], dataset_name: str, save=False, output_path="./data_sample"):
    """
    Shows samples form the dataset.

    :param indexes: show samples with these indexes
    :param dataset_name: registered name of the dataset
    :param save: if true, saves the outputs
    :param output_path: save outputs to (*<index>.png) and (*_lung_<index>.png)
    """
    dataset = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    for idx in indexes:
        print(idx)

        boxes1 = dataset[idx]["annotations"][0:2]
        boxes1.append(dataset[idx]["annotations"][3])

        boxes2 = [dataset[idx]["annotations"][2]]

        img = np.array(Image.open(dataset[idx]["file_name"]).convert("RGB"))
        visualizer = Visualizer(img, metadata=metadata, scale=0.5)

        dataset[idx]["annotations"] = boxes1
        out = visualizer.draw_dataset_dict(dataset[idx])

        plt.imshow(out.get_image())
        if save:
            plt.savefig(f"{output_path}{idx}.png",
                        bbox_inches="tight",
                        format="png",
                        dpi=1200)
        plt.show()
        plt.close()

        #
        visualizer = Visualizer(img, metadata=metadata, scale=0.5)

        dataset[idx]["annotations"] = boxes2
        out = visualizer.draw_dataset_dict(dataset[idx])

        plt.imshow(out.get_image())
        if save:
            plt.savefig(f"{output_path}_lung_{idx}.png",
                        bbox_inches="tight",
                        format="png",
                        dpi=1200)
        plt.show()
        plt.close()


def vis_augmentation(dataset_name: str, config_file: str, mask: bool, batches: int, batch_size: int):
    """
    Shows augmentation samples form the dataset.

    :param dataset_name: registered name of the dataset
    :param config_file: config Node
    :param mask: show masks (needs mask r-cnn config file)
    :param batches: show n batches
    :param batch_size: the batch size
    """
    cfg = train.setup(Args(config_file))
    cfg.defrost()
    cfg.DATASETS.TRAIN = (dataset_name,)

    augmentations = [
        T.RandomContrast(0.9, 1.2)
    ]

    mapper = card_detection.mapper.CustomMapper(cfg, is_train=True, augmentations=augmentations)

    loader = build_detection_train_loader(cfg, mapper=mapper)

    metadata = MetadataCatalog.get(dataset_name)

    for idx, inputs in enumerate(loader):
        if idx >= batches:
            break

        for batch_idx in range(0, batch_size):
            inp = inputs[batch_idx]

            img = np.transpose(inp["image"], (1, 2, 0))

            draw_dict = dict()
            draw_dict["height"] = inp["height"]
            draw_dict["width"] = inp["width"]
            draw_dict["annotations"] = list()

            draw_dict["annotations"].append(dict())
            draw_dict["annotations"][0]["bbox"] = inp["instances"].gt_boxes.tensor.tolist()[2]
            draw_dict["annotations"][0]["bbox_mode"] = 0
            draw_dict["annotations"][0]["category_id"] = 0

            if mask:
                mask = inp["instances"].gt_masks.tensor[2]

                draw_dict["annotations"][0]["segmentation"] = dict()
                draw_dict["annotations"][0]["segmentation"] = encode(np.asarray(mask, order="F"))

            v = Visualizer(img, metadata=metadata, scale=0.5)
            out = v.draw_dataset_dict(draw_dict)

            plt.imshow(out.get_image())
            plt.show()
            plt.close()


class Args:
    def __init__(self, config_file):
        self.config_file = config_file
        self.num_gpus = 0
        self.opts = ()
