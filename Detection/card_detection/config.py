from detectron2 import model_zoo
from detectron2.config import CfgNode


def add_base(cfg: CfgNode):
    """
    Add the base configuration of your project.

    :param cfg: add to this config
    """
    cfg.DATASETS.TRAIN = ("chestXray",)
    cfg.DATASETS.TEST = ("testXray",)
    cfg.DATALOADER.NUM_WORKERS = 4

    #
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.MASK_FORMAT = "bitmask"

    cfg.INPUT.RANDOM_FLIP = "none"

    cfg.INPUT.CROP.ENABLED = False
    cfg.INPUT.CROP.TYPE = "relative_range"
    cfg.INPUT.CROP.SIZE = [0.9, 0.9]

    #
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[64, 128, 256, 512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.7, 1.05, 1.85, 2.15, 2.5]]

    #
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 1

    cfg.SOLVER.IMS_PER_BATCH = 4

    cfg.SOLVER.MAX_ITER = 30000
    cfg.SOLVER.CHECKPOINT_PERIOD = 706

    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = [23333, 27780]

    cfg.SOLVER.BASE_LR = 0.00025


def add_model_zoo(cfg: CfgNode, yaml_path: str):
    """
    Adds a model from detectron2's model zoo to the config.

    :param cfg: add to this config
    :param yaml_path: path to the yaml file starting from .../detectron2/configs
    """
    cfg.merge_from_file(model_zoo.get_config_file(yaml_path),
                        allow_unsafe=False)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml_path)

    #
    add_base(cfg)


def add_file(cfg: CfgNode, yaml_path: str):
    """
    Adds a configfile to the config.

    :param cfg: config to add model to
    :param yaml_path: path to the yaml file containing the configuration
    """
    raise NotImplementedError


def add_config(cfg: CfgNode, config_name: str):
    """
    Helper methode to keep training command short.

    Adds one of the following configs to cfg:

    - faster_rcnn_r_101_fpn_3x
    - mask_rcnn_r_101_fpn_3x

    :param cfg: config
    :param config_name: config name from the list above
    """
    if config_name == "mask_rcnn_r_101_fpn_3x":
        add_model_zoo(cfg, "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

    elif config_name == "mask_rcnn_r_50_fpn_3x":
        add_model_zoo(cfg, "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    elif config_name == "mask_rcnn_x_101_fpn_3x":
        add_model_zoo(cfg, "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

    elif config_name == "faster_rcnn_r_101_fpn_3x":
        add_model_zoo(cfg, "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

    elif config_name == "faster_rcnn_r_50_fpn_3x":
        add_model_zoo(cfg, "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    elif config_name == "faster_rcnn_x_101_fpn_3x":
        add_model_zoo(cfg, "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    else:
        raise Exception("no model was specified")
