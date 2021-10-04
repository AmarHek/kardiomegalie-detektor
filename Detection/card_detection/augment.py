import numpy as np

from typing import Union, Tuple
from detectron2.data.transforms import RandomCrop, Augmentation, CropTransform, ColorTransform


class RandomCropBoxConstraint(Augmentation):
    """
    Similar to :class:`RandomCrop`, but find a cropping window such that no single bounding box is cropped.
    """

    def __init__(
        self,
        crop_type: str,
        crop_size,
        padding: Union[int, Tuple[int, int, int, int]]
    ):
        """
        :param crop_type: same as in RandomCrop
        :param crop_size: same as in RandomCrop
        :param padding: expands minimum cropping window
        """
        super().__init__()

        if isinstance(padding, int):
            padding = padding, padding, padding, padding

        self.padding = padding
        self.crop_aug = RandomCrop(crop_type, crop_size)

    def get_transform(self, image, boxes):
        h, w = image.shape[:2]
        crop_h, crop_w = self.crop_aug.get_crop_size((h, w))
        assert h >= crop_h and w >= crop_w, "Shape computation in {} has bugs.".format(self)
        y0 = np.random.randint(h - crop_h + 1)
        x0 = np.random.randint(w - crop_w + 1)

        for box in boxes:
            if x0 > box[0] - self.padding[0]:
                x0 = box[0] - self.padding[0]

            if y0 > box[1] - self.padding[1]:
                y0 = box[1] - self.padding[1]

            if x0 + crop_w < box[2] + self.padding[2]:
                crop_w = box[2] + self.padding[2] - x0

            if y0 + crop_h < box[3] + self.padding[3]:
                crop_h = box[3] + self.padding[3] - y0

        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x0 + crop_w > w:
            crop_w = w - x0
        if y0 + crop_h > h:
            crop_h = h - y0

        return CropTransform(x0, y0, crop_w, crop_h)


class RandomNoise(Augmentation):
    """
    Adds gaussian noise to the image.\n
    RandomApply is very important for this augmentation method, as clean data should also be kept.
    """

    def __init__(self, mean: float, sigma: float):
        """
        :param mean: mean for calculating gaussian distribution
        :param sigma: sigma for calculating gaussian distribution
        """
        super().__init__()

        self.mean = mean
        self.sigma = sigma

    def get_transform(self, image):
        gauss = np.random.normal(self.mean, self.sigma, image.shape).astype("f")
        gauss = gauss.reshape(image.shape)

        return ColorTransform(lambda x: x + gauss)
