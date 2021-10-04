import os
import numpy as np

from PIL import Image
from cv2 import findContours, RETR_TREE, CHAIN_APPROX_NONE
from pycocotools import mask as mask_util
from xml.etree import ElementTree


def read_pascal_voc_format_dict(pascal_voc_path):
    """
    Reads bounding boxes from pascal voc format.
    Multiple bounding boxes with same label are ignored.

    :param pascal_voc_path: path to pascal voc file
    :return: dict with bbox to each label
    """
    bbox = dict()

    dom = ElementTree.parse(pascal_voc_path)
    objects = dom.findall("object")

    for obj in objects:
        cls = obj.find("name").text

        bnd_box = obj.find("bndbox")
        x_min = int(bnd_box.find("xmin").text)
        y_min = int(bnd_box.find("ymin").text)
        x_max = int(bnd_box.find("xmax").text)
        y_max = int(bnd_box.find("ymax").text)

        bbox[cls] = [x_min, y_min, x_max, y_max]

    return bbox


def read_pascal_voc_format_list(pascal_voc_path, classes):
    """
    Reads bounding boxes from pascal voc format and converts label to id

    :param pascal_voc_path: path to pascal voc file
    :param classes: list of all label classes in order of ids
    :return: list with bboxes and ids
    """
    bbox = list()

    dom = ElementTree.parse(pascal_voc_path)
    objects = dom.findall("object")

    for obj in objects:
        cls = obj.find("name").text
        c_id = 0
        for i in range(0, len(classes)):
            if cls == classes[i]:
                c_id = i

        bnd_box = obj.find("bndbox")
        x_min = int(bnd_box.find("xmin").text)
        y_min = int(bnd_box.find("ymin").text)
        x_max = int(bnd_box.find("xmax").text)
        y_max = int(bnd_box.find("ymax").text)

        bbox.append([c_id, x_min, y_min, x_max, y_max])

    return sorted(bbox, key=lambda x: x[0])


def bbox_to_pascal_voc(image_path, lung_mask_path, heart_mask_path, save_folder, factor=1):
    """
    Takes an image, lung and heart mask and saves the bounding boxes to pascal vov format.

    :param image_path: path to the image
    :param lung_mask_path: path to the lung mask
    :param heart_mask_path: path to the heart mask
    :param save_folder: path to folder to save to
    :param factor: scale the masks by factor
    """
    image_path = os.path.abspath(image_path)

    image_file_name = os.path.basename(image_path)
    image_folder_name = os.path.basename(os.path.dirname(image_path))

    dimensions = np.asarray(Image.open(image_path).convert("L")).shape

    lung_xml = ""
    if lung_mask_path is not None:
        bbox_lung = get_bbox(lung_mask_path, factor=factor)
        lung_xml = f"""
    <object>
        <name>left_lung</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{bbox_lung[0][0]}</xmin>
            <ymin>{bbox_lung[0][1]}</ymin>
            <xmax>{bbox_lung[0][2]}</xmax>
            <ymax>{bbox_lung[0][3]}</ymax>
        </bndbox>
    </object>
    <object>
        <name>right_lung</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{bbox_lung[1][0]}</xmin>
            <ymin>{bbox_lung[1][1]}</ymin>
            <xmax>{bbox_lung[1][2]}</xmax>
            <ymax>{bbox_lung[1][3]}</ymax>
        </bndbox>
    </object>
"""

    heart_xml = ""
    if heart_mask_path is not None:
        bbox_heart = get_bbox(heart_mask_path, factor=factor)
        heart_xml = f"""
    <object>
        <name>heart</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{bbox_heart[0][0]}</xmin>
            <ymin>{bbox_heart[0][1]}</ymin>
            <xmax>{bbox_heart[0][2]}</xmax>
            <ymax>{bbox_heart[0][3]}</ymax>
        </bndbox>
    </object>
"""

    xml = f"""
<annotation>
    <folder>{image_folder_name}</folder>
    <filename>{image_file_name}</filename>
    <path>{image_path}</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>{dimensions[1]}</width>
        <height>{dimensions[0]}</height>
        <depth>1</depth>
    </size>
    <segmented>0</segmented>
    {lung_xml}{heart_xml}
</annotation>
    """

    image_file_name = image_file_name.split(".")[0]

    with open(f"{save_folder}/{image_file_name}.xml", "w") as file:
        file.write(xml)


def get_bbox(mask_path, factor=1):
    """
    Finds the corresponding bounding box to a segmentation mask.

    :param mask_path: path to mask file
    :param factor: scale the mask by factor
    :return: individual bboxes in the mask from left to right
    """
    # all_mask = cv2.imread(mask_path, 0)
    all_mask = np.asarray(Image.open(mask_path).convert("L"))
    contour, _ = findContours(all_mask, RETR_TREE, CHAIN_APPROX_NONE)    # CHAIN_APPROX_SIMPLE

    objects = list()

    for obj in contour:
        poly_x = list()
        poly_y = list()
        for point in obj:
            poly_x.append(int(point[0][0]))
            poly_y.append(int(point[0][1]))

        objects.append([poly_x, poly_y])

    bbox = list()

    for obj in objects:
        bbox.append([min(obj[0])*factor, min(obj[1])*factor, max(obj[0])*factor, max(obj[1])*factor])

    return sorted(bbox, key=lambda x: x[0])


def get_individual_rle(mask_path, decode=True, encoding="utf-8", factor=1):
    """
    Finds two maks separated through a horizontal gap and converts them to run length encoding.

    :param mask_path: path to mask file
    :param decode: if true decodes bytes to string
    :param encoding: used encoding
    :param factor: scale the mask by factor
    :return: left mask, right mask and all masks in rle format
    """
    # all_mask = cv2.imread(mask_path, 0)

    all_mask = Image.open(mask_path)
    all_mask = all_mask.resize((all_mask.size[0] * factor, all_mask.size[1] * factor), Image.NEAREST)
    all_mask = np.array(all_mask.convert("L"))
    all_mask[all_mask > 0] = 255

    rotated = np.rot90(all_mask)

    flag = False
    row_number = 0
    for row in reversed(rotated):
        black = True

        for px in row:
            if px == 255:
                flag = True
                black = False

        if flag and black:
            break

        row_number += 1

    left_mask = np.copy(all_mask)
    for y in range(0, len(left_mask)):
        for x in range(0, len(left_mask[0])):
            if x >= row_number:
                left_mask[y, x] = 0

    right_mask = np.copy(all_mask)
    for y in range(0, len(right_mask)):
        for x in range(0, len(right_mask[0])):
            if x <= row_number:
                right_mask[y, x] = 0

    #
    left_rle = mask_util.encode(np.asarray(left_mask, order="F"))
    right_rle = mask_util.encode(np.asarray(right_mask, order="F"))
    all_rle = mask_util.encode(np.asarray(all_mask, order="F"))

    if decode:
        left_rle["counts"] = left_rle["counts"].decode(encoding)
        right_rle["counts"] = right_rle["counts"].decode(encoding)
        all_rle["counts"] = all_rle["counts"].decode(encoding)

    return left_rle, right_rle, all_rle


def get_rle(mask_path, decode=True, encoding="utf-8", factor=1):
    """
    Convert mask to run length encoding.

    :param mask_path: path to mask file
    :param decode: if true decodes bytes to string
    :param encoding: used encoding
    :param factor: scale the mask by factor
    :return: mask in rle format
    """
    mask = Image.open(mask_path)
    mask = mask.resize((mask.size[0] * factor, mask.size[1] * factor), Image.NEAREST)
    mask = np.array(mask.convert("L"))
    mask[mask > 0] = 255

    rle = mask_util.encode(np.asarray(mask, order="F"))

    if decode:
        rle["counts"] = rle["counts"].decode(encoding)

    return rle
