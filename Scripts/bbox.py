import os
import mask_utils


def generate_bbox(load_folder_path, lung_folder_path, heart_folder_path, save_folder_path, factor=1):
    """
    Goes through a whole folder of images generates bounding boxes with mask_utils.bbox_to_pascal_voc.

    :param load_folder_path: path to the image folder
    :param lung_folder_path: path to the lung masks folder
    :param heart_folder_path: path to the heart masks folder
    :param save_folder_path: path to the folder to save to
    :param factor: scale masks by factor
    :return:
    """
    load_folder_path = os.path.abspath(load_folder_path)
    save_folder_path = os.path.abspath(save_folder_path)

    if lung_folder_path is not None:
        lung_folder_path = os.path.abspath(lung_folder_path)

    if heart_folder_path is not None:
        heart_folder_path = os.path.abspath(heart_folder_path)

    for file in os.listdir(load_folder_path):
        mask_utils.bbox_to_pascal_voc(os.path.join(load_folder_path, file),
                                      os.path.join(lung_folder_path, file) if lung_folder_path is not None else None,
                                      os.path.join(heart_folder_path, file) if heart_folder_path is not None else None,
                                      os.path.join(save_folder_path),
                                      factor=factor)

        print(f"generated bbox for {file}")
