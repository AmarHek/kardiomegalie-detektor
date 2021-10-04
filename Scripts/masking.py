import os
import numpy as np
import mask_utils

from PIL import Image


def mask_with_bbox(bbox_path, mask_path, output_path):
    """
    Mask a segmentation mask with a bounding box

    :param bbox_path: path of the pascal voc file, containing the bounding box
    :param mask_path: path to the mask
    :param output_path: save masked mask to
    """
    bbox = mask_utils.read_pascal_voc_format_dict(bbox_path)["heart"]
    mask = np.array(Image.open(mask_path).convert("L"))

    bbox_mask = np.zeros(mask.shape)
    bbox_mask[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1] = 1

    mask[bbox_mask < 1] = 0
    Image.fromarray(mask).save(output_path, format="png")


def mask_with_bbox_folder(bbox_folder, mask_folder, output_folder):
    """
    Call mask_with_bbox for a whole folder

    :param bbox_folder: path of the pascal voc files
    :param mask_folder: path of the masks
    :param output_folder: save new masks to
    """
    bbox_folder = os.path.abspath(bbox_folder)
    mask_folder = os.path.abspath(mask_folder)
    output_folder = os.path.abspath(output_folder)

    for file in os.listdir(bbox_folder):
        img_file = f"{file.split('.')[0]}.png"
        if os.path.isfile(os.path.join(bbox_folder, file)) and os.path.isfile(os.path.join(mask_folder, img_file)):
            mask_with_bbox(os.path.join(bbox_folder, file),
                           os.path.join(mask_folder, img_file),
                           os.path.join(output_folder, img_file))

        print(f"masked mask {img_file} with bbox {file}")


def overlay_bbox(image_path, bbox_path, output_path):
    """
    Overlay a bounding box on an image

    :param image_path: path to the image
    :param bbox_path: path to the pascal voc file, containing the bounding box
    :param output_path: save overlayed image to
    """
    image = np.array(Image.open(image_path).convert("RGB"))
    bboxes = mask_utils.read_pascal_voc_format_dict(bbox_path)

    for name in ["left_lung", "right_lung", "heart"]:
        bbox = bboxes[name]

        draw = np.zeros(image.shape)
        draw[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1] = 1
        draw[bbox[1] + 3:bbox[3] - 2, bbox[0] + 3:bbox[2] - 2] = 0

        image[draw > 0] = 0

    Image.fromarray(image).save(output_path, format="png")


def overlay_bbox_folder(image_folder, bbox_folder, output_folder):
    """
    Call overlay_bbox for a whole folder

    :param image_folder: path to the images
    :param bbox_folder: path to the pascal voc files
    :param output_folder: save overlayed image to
    """
    image_folder = os.path.abspath(image_folder)
    bbox_folder = os.path.abspath(bbox_folder)
    output_folder = os.path.abspath(output_folder)

    for file in os.listdir(image_folder):
        xml_file = f"{file.split('.')[0]}.xml"
        if os.path.isfile(os.path.join(bbox_folder, xml_file)):
            overlay_bbox(os.path.join(image_folder, file),
                         os.path.join(bbox_folder, xml_file),
                         os.path.join(output_folder, file))

        print(f"drew rect from {xml_file} on {file}")
