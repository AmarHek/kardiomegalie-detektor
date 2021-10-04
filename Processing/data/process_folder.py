import os
import data.mask_utils as mask_utils


def to_bitmask_folder(polygon_folder: str, classes: list[str], output_folder: str):
    """
    Call mask_utils.polygon_to_bitmask for an whole folder.

    :param polygon_folder: path to the folder containing the polygons
    :param classes: classes to include
    :param output_folder: path to the folder to save to
    """
    polygon_folder = os.path.abspath(polygon_folder)
    output_folder = os.path.abspath(output_folder)

    for file in os.listdir(polygon_folder):
        mask_utils.polygon_to_bitmask(os.path.join(polygon_folder, file),
                                      classes,
                                      output_folder)

        print(f"converted {file}")


def generate_bbox(load_folder: str,
                  lung_folder: str,
                  heart_folder: str,
                  save_folder: str,
                  factor=1):
    """
    Goes through a whole folder of images generating bounding boxes with mask_utils.bbox_to_pascal_voc.

    :param load_folder: path to the image folder
    :param lung_folder: path to the lung masks folder
    :param heart_folder: path to the heart masks folder
    :param save_folder: path to the folder to save to
    :param factor: scale masks by factor
    """
    load_folder = os.path.abspath(load_folder)
    save_folder = os.path.abspath(save_folder)

    if lung_folder is not None:
        lung_folder = os.path.abspath(lung_folder)

    if heart_folder is not None:
        heart_folder = os.path.abspath(heart_folder)

    for file in os.listdir(load_folder):
        mask_utils.bbox_to_pascal_voc(os.path.join(load_folder, file),
                                      os.path.join(lung_folder, file) if lung_folder is not None else None,
                                      os.path.join(heart_folder, file) if heart_folder is not None else None,
                                      os.path.join(save_folder),
                                      factor=factor)

        print(f"generated bbox for {file}")

