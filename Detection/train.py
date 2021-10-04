import os
import detectron2.data.transforms as T

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.data import build_detection_train_loader
from detectron2.checkpoint import DetectionCheckpointer

from card_detection import add_config, load_data, build_aug, \
    JsonDumpEvaluator, LossEvalHook, \
    CustomMapper, RandomCropBoxConstraint, RandomNoise


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return JsonDumpEvaluator(dataset_name, cfg.OUTPUT_DIR)

    @classmethod
    def build_train_loader(cls, cfg):
        augmentations = build_aug(cfg, True,
                                  [RandomCropBoxConstraint("relative_range", (0.9, 0.9), 100)],
                                  [T.RandomContrast(0.9, 1.2),
                                   T.RandomApply(RandomNoise(0, 0.5), 0.5)])

        mapper = CustomMapper(cfg, is_train=True, augmentations=augmentations)

        return build_detection_train_loader(cfg, mapper=mapper)

    def build_hooks(self):
        hooks = super().build_hooks()

        loss_eval_hook = LossEvalHook(self.cfg, "valXray")
        hooks.insert(-1, loss_eval_hook)

        return hooks


def setup(arguments):
    cfg = get_cfg()
    add_config(cfg, arguments.config_file)
    cfg.merge_from_list(arguments.opts)

    if arguments.num_gpus == 0:
        cfg.MODEL.DEVICE = "cpu"

    cfg.freeze()

    arguments.config_file = ''
    default_setup(cfg, arguments)

    return cfg


def main(arguments):
    cfg = setup(arguments)

    load_data("chestXray",
              "../chestXray/dataset.json",
              ["left_lung", "right_lung", "lung", "heart"])

    load_data("testXray",
              "../testXray/dataset.json",
              ["left_lung", "right_lung", "lung", "heart"])

    load_data("valXray",
              "../valXray/dataset.json",
              ["left_lung", "right_lung", "lung", "heart"])

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if arguments.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=arguments.resume
        )
        res = Trainer.test(cfg, model)

        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=arguments.resume)

    return trainer.train()


if __name__ == '__main__':
    """
    Training:
        python train.py --config-file faster_rcnn_r_101_fpn_3x --num-gpus 8 OUTPUT_DIR output
    
    Evaluating:
        python train.py --config-file faster_rcnn_r_101_fpn_3x --eval-only MODEL.WEIGHTS output/model_final.pth
    """
    args = default_argument_parser().parse_args()
    print("Command Line Args: ", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
