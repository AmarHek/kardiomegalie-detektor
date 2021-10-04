import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import detectron2.data.transforms as T

from PIL import Image
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer


def predict_dataset(cfg, dataset_name):
    """
    Shows predictions on a dataset.

    :param cfg: config Node
    :param dataset_name: registered name of the dataset
    :return:
    """
    metadata = MetadataCatalog.get(dataset_name)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.model.eval()

    val_loader = build_detection_test_loader(cfg, dataset_name)

    for i in val_loader[0:1]:
        with torch.no_grad():
            prediction = trainer.model(i)[0]

        prediction = prediction["instances"].to("cpu")
        print(prediction)

        img = np.asarray(Image.open(i[0]["file_name"]).convert("RGB"))
        v = Visualizer(img, metadata=metadata, scale=0.5,)
        out = v.draw_instance_predictions(prediction)

        plt.imshow(out.get_image())
        plt.show()
        plt.close()


class Predictor:
    def __init__(self, cfg, threshold):
        """
        Creates a simple Predictor. Loads weights form cfg.OUTPUT_DIR/model_final.pth.

        :param cfg: config Node
        :param threshold: only output instances with score >= threshold
        """
        self.cfg = cfg.clone()

        if self.cfg.is_frozen():
            self.cfg.defrost()

        if os.path.isfile(os.path.join(cfg.OUTPUT_DIR, "model_final.pth")):
            self.cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

    def __call__(self, original_image):
        with torch.no_grad():
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]

            return predictions
