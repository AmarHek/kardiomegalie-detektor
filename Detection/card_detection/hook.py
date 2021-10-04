import torch

from detectron2.utils import comm
from detectron2.config import CfgNode
from detectron2.engine import HookBase
from detectron2.data import DatasetCatalog, DatasetMapper, build_detection_train_loader
from detectron2.data.samplers import InferenceSampler

from card_detection import build_aug


class LossEvalHook(HookBase):
    def __init__(self, cfg: CfgNode, dataset_name: str):
        cfg = cfg.clone()
        cfg.defrost()
        cfg.DATASETS.TRAIN = dataset_name
        cfg.freeze()

        self.period = cfg.SOLVER.CHECKPOINT_PERIOD

        dataset_length = len(DatasetCatalog.get(dataset_name))
        mapper = DatasetMapper(cfg, is_train=True, augmentations=build_aug(cfg, is_train=False))
        self.data_loader = build_detection_train_loader(cfg, mapper=mapper,
                                                        sampler=InferenceSampler(dataset_length))

    def do_eval(self):
        losses = []

        with torch.no_grad():
            for inputs in self.data_loader:
                outputs = self.trainer.model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # loss_dict = {
                #     k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
                #     for k, v in outputs.items()
                # }
                # total_losses_reduced = sum(loss for loss in loss_dict.values())
                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(outputs).items()}
                total_losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                losses.append(total_losses_reduced)

        # mean_loss = np.mean(losses)
        mean_loss = torch.mean(torch.as_tensor(losses))

        if comm.is_main_process():
            self.trainer.storage.put_scalars(validation_loss=mean_loss)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if (next_iter >= self.trainer.max_iter) or (self.period > 0 and next_iter % self.period == 0):
            self.do_eval()
