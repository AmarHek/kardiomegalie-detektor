import json
import csv
import matplotlib.pyplot as plt

from evaluation import evaluation_utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def accuracy_precision_recall_f1(card_threshold: float,
                                 result_path: str,
                                 ground_truth_path: str,
                                 classes: list[str],
                                 output_path: str,
                                 if_none: int = 1,
                                 ground_truth_htq_path: str = None):
    """
    Calculate accuracy, precision, recall and f1 and save them to a json file.

    :param card_threshold: cardiomegaly is present if the htq > card_threshold
    :param result_path: path to the json file containing the results
    :param ground_truth_path: path to the ground truth dataset (detectron2 format)
    :param classes: classes to the ids in the dataset in order of the ids
    :param output_path: save metrics to path (*.json)
    :param if_none: if a result ist None, choose this class
    :param ground_truth_htq_path: path to a json file with custom htq's
    """
    if ground_truth_htq_path is not None:
        with open(ground_truth_htq_path, "r") as file:
            htq_s = json.loads(file.read())

    mask_types = ["bbox", "mask"]
    lung_types = ["lung", "left_right_lung"]

    cardiomegaly_gt = list()

    cardiomegaly = dict()
    for mask_type in mask_types:
        cardiomegaly[mask_type] = dict()
        for lung_type in lung_types:
            cardiomegaly[mask_type][lung_type] = list()

    for idx, name, result, ground_truth in evaluation_utils.results_iterator(result_path, ground_truth_path):
        if ground_truth_htq_path is not None:
            gt_card = True if htq_s[str(idx)]["htq"] > card_threshold else False
        else:
            gt = dict()
            for category in ground_truth["annotations"]:
                gt[classes[category["category_id"]]] = category["bbox"]

            _, gt_card = evaluation_utils.htq_card(card_threshold,
                                                   gt["heart"],
                                                   gt["lung"])

        cardiomegaly_gt.append(gt_card)

        for mask_type in mask_types:
            for lung_type in lung_types:
                if lung_type == "lung":
                    if (result["heart"][mask_type] is None) \
                            or (result["lung"][mask_type] is None):
                        card = if_none
                    else:
                        _, card = evaluation_utils.htq_card(card_threshold,
                                                            result["heart"][mask_type],
                                                            result["lung"][mask_type])
                else:
                    if (result["heart"][mask_type] is None) \
                            or (result["left_lung"][mask_type] is None) \
                            or (result["right_lung"][mask_type] is None):
                        card = if_none
                    else:
                        _, card = evaluation_utils.htq_card(card_threshold,
                                                            result["heart"][mask_type],
                                                            result["left_lung"][mask_type],
                                                            result["right_lung"][mask_type])

                cardiomegaly[mask_type][lung_type].append(card)

    output = dict()

    for mask_type in mask_types:
        output[mask_type] = dict()

        for lung_type in lung_types:
            output[mask_type][lung_type] = dict()

            output[mask_type][lung_type]["accuracy"] = accuracy_score(cardiomegaly_gt,
                                                                      cardiomegaly[mask_type][lung_type])
            output[mask_type][lung_type]["precision"] = precision_score(cardiomegaly_gt,
                                                                        cardiomegaly[mask_type][lung_type],
                                                                        average=None, zero_division=0)[1]
            output[mask_type][lung_type]["recall"] = recall_score(cardiomegaly_gt,
                                                                  cardiomegaly[mask_type][lung_type],
                                                                  average=None, zero_division=0)[1]
            output[mask_type][lung_type]["specificity"] = recall_score(cardiomegaly_gt,
                                                                       cardiomegaly[mask_type][lung_type],
                                                                       average=None, zero_division=0)[0]
            output[mask_type][lung_type]["f1-score"] = f1_score(cardiomegaly_gt,
                                                                cardiomegaly[mask_type][lung_type],
                                                                average=None, zero_division=0)[1]

            # when using three classes, average="macro" has to be added to precision_, recall_ and f1_score

    with open(f"{output_path}.json", "w") as file:
        file.write(json.dumps(output, indent=4))


def difference_correlation(result_path,
                           ground_truth_path,
                           classes: list[str],
                           output_path: str,
                           ground_truth_htq_path: str = None):
    """
    Calculate difference with mean and standard deviation and correlation coefficient between
    ground truth and intersection over union of the following values

    - htq_lung, htq_left_right_lung
    - mld, mrd
    - id_lung, id_left_right_lung

    and save them to a json file.

    :param result_path: path to the json file containing the results
    :param ground_truth_path: path to the ground truth dataset (detectron2 format)
    :param classes: classes to the ids in the dataset in order of the ids
    :param output_path: save metrics to path (*_diff.json) and (*_corr.json)
    :param ground_truth_htq_path: path to a json file with custom htq's
    """
    if ground_truth_htq_path is not None:
        with open(ground_truth_htq_path, "r") as file:
            htq_s = json.loads(file.read())

    mask_types = ["bbox", "mask"]
    eval_types = ["htq_lung", "htq_left_right_lung", "mld", "mrd", "id_lung", "id_left_right_lung"]

    gt_lists = dict()
    for eval_type in eval_types:
        gt_lists[eval_type] = list()

    iou_lists = dict()
    for mask_type in mask_types:
        iou_lists[mask_type] = dict()
        for eval_type in eval_types:
            iou_lists[mask_type][eval_type] = list()

    lists = dict()
    for mask_type in mask_types:
        lists[mask_type] = dict()
        for eval_type in eval_types:
            lists[mask_type][eval_type] = list()

    for idx, name, result, ground_truth in evaluation_utils.results_iterator(result_path, ground_truth_path):
        gt = dict()
        for category in ground_truth["annotations"]:
            gt[classes[category["category_id"]]] = category["bbox"]

        if ground_truth_htq_path is not None:
            gt_htq = htq_s[str(idx)]["htq"]
        else:
            gt_htq, _ = evaluation_utils.htq_card(0.5, gt["heart"], gt["lung"])

        gt_lists["htq_lung"].append(gt_htq)
        gt_lists["htq_left_right_lung"].append(gt_htq)

        gt_mld, gt_mrd = evaluation_utils.mld_mrd(gt["lung"], gt["heart"])
        gt_lists["mld"].append(gt_mld / ground_truth["width"])
        gt_lists["mrd"].append(gt_mrd / ground_truth["width"])

        gt_ld = evaluation_utils.lung_diameter(gt["lung"])
        gt_lists["id_lung"].append(gt_ld / ground_truth["width"])
        gt_lists["id_left_right_lung"].append(gt_ld / ground_truth["width"])

        for mask_type in mask_types:
            htq_lung, _ = evaluation_utils.htq_card(0.5,
                                                    result["heart"][mask_type],
                                                    result["lung"][mask_type])

            htq_left_right_lung, _ = evaluation_utils.htq_card(0.5,
                                                               result["heart"][mask_type],
                                                               result["left_lung"][mask_type],
                                                               result["right_lung"][mask_type])

            lists[mask_type]["htq_lung"].append(htq_lung)
            lists[mask_type]["htq_left_right_lung"].append(htq_left_right_lung)

            #
            mld, mrd = evaluation_utils.mld_mrd(gt["lung"], result["heart"][mask_type])

            lists[mask_type]["mld"].append(mld / ground_truth["width"])
            lists[mask_type]["mrd"].append(mrd / ground_truth["width"])

            #
            ld_lung = evaluation_utils.lung_diameter(result["lung"][mask_type])
            ld_left_right_lung = evaluation_utils.lung_diameter(result["left_lung"][mask_type],
                                                                result["right_lung"][mask_type])

            lists[mask_type]["id_lung"].append(ld_lung / ground_truth["width"])
            lists[mask_type]["id_left_right_lung"].append(ld_left_right_lung / ground_truth["width"])

            #
            iou = dict()
            for cls in classes:
                res_bbox = result[cls][mask_type] if result[cls][mask_type] is not None else [0, 0, 0, 0]
                iou[cls] = evaluation_utils.iou_dice(gt[cls], res_bbox)[0]

            iou_lists[mask_type]["htq_lung"].append(
                (iou["heart"] + iou["lung"]) / 2
            )
            iou_lists[mask_type]["htq_left_right_lung"].append(
                (iou["heart"] + iou["left_lung"] + iou["right_lung"]) / 3
            )
            iou_lists[mask_type]["mld"].append(iou["heart"])
            iou_lists[mask_type]["mrd"].append(iou["heart"])
            iou_lists[mask_type]["id_lung"].append(iou["lung"])
            iou_lists[mask_type]["id_left_right_lung"].append(
                (iou["left_lung"] + iou["right_lung"]) / 2
            )

    output_diff = dict()
    output_corr = dict()

    for mask_type in mask_types:
        output_diff[mask_type] = dict()
        output_corr[mask_type] = dict()

        for eval_type in eval_types:
            output_diff[mask_type][eval_type] = dict()

            mean, std = evaluation_utils.difference(gt_lists[eval_type],
                                                    lists[mask_type][eval_type])

            output_diff[mask_type][eval_type]["mean"] = mean
            output_diff[mask_type][eval_type]["std"] = std

            #
            output_corr[mask_type][eval_type] = dict()

            correlation_gt = evaluation_utils.correlation_coefficient(gt_lists[eval_type],
                                                                      lists[mask_type][eval_type])

            output_corr[mask_type][eval_type]["ground_truth"] = correlation_gt

            correlation_iou = evaluation_utils.correlation_coefficient(iou_lists[mask_type][eval_type],
                                                                       lists[mask_type][eval_type])

            output_corr[mask_type][eval_type]["iou"] = correlation_iou

            #
            plt.plot([0, 1], [0, 1], color="black")

            plt.scatter(lists[mask_type][eval_type],
                        gt_lists[eval_type],
                        label=f"{mask_type}/{eval_type} - correlation")

            plt.legend()

            plt.show()
            plt.close()

    with open(f"{output_path}_diff.json", "w") as file:
        file.write(json.dumps(output_diff, indent=4))

    with open(f"{output_path}_corr.json", "w") as file:
        file.write(json.dumps(output_corr, indent=4))


def severity_groups(thresholds: list[float],
                    result_path: str,
                    ground_truth_path: str,
                    bbox_type: str,
                    lung_type: str,
                    output_path: str):
    """
    Generate a csv file with corresponding severity groups to the images.

    :param thresholds: severity group thresholds (sorted)
    :param result_path: path to the json file containing the results
    :param ground_truth_path: path to the ground truth dataset (detectron2 format)
    :param bbox_type: bbox or mask
    :param lung_type: lung or left_right_lung
    :param output_path: save csv to path (*.csv)
    """
    output = list()

    for idx, name, result, ground_truth in evaluation_utils.results_iterator(result_path, ground_truth_path):
        if lung_type == "lung":
            if (result["heart"][bbox_type] is None) \
                    or (result["lung"][bbox_type] is None):
                htq = 1
            else:
                htq, _ = evaluation_utils.htq_card(0.5,
                                                   result["heart"][bbox_type],
                                                   result["lung"][bbox_type])
        else:
            if (result["heart"][bbox_type] is None) \
                    or (result["left_lung"][bbox_type] is None) \
                    or (result["right_lung"][bbox_type] is None):
                htq = 1
            else:
                htq, _ = evaluation_utils.htq_card(0.5,
                                                   result["heart"][bbox_type],
                                                   result["left_lung"][bbox_type],
                                                   result["right_lung"][bbox_type])

        severity = 0
        for sev, threshold in enumerate(thresholds):
            if htq > threshold:
                severity = sev + 1

        output.append([name, severity])

    with open(f"{output_path}.csv", "w") as file:
        writer = csv.writer(file, delimiter=";")

        sorted_output = sorted(output, key=lambda x: x[0])
        sorted_output.insert(0, [""])
        sorted_output.insert(0, ["filename", "severity group"])

        writer.writerows(sorted_output)


def manual_automatic_htq_diff(ground_truth_path: str,
                              ground_truth_htq_path: str,
                              classes: list[str],
                              output_path: str):
    """
    Compares the manuel measured HTQs and the calcuated HTQs from the ground truth data.

    :param ground_truth_path: dataset in detectron2's format
    :param ground_truth_htq_path: json file with htq to an image id
    :param classes: classes in order of the class ids
    :param output_path: save comparison to (*.csv)
    """
    output = list()

    with open(ground_truth_htq_path, "r") as file:
        htq_s = json.loads(file.read())

    manual_list = list()
    automatic_list = list()

    for idx, name, result, ground_truth in evaluation_utils.results_iterator(None, ground_truth_path):
        manual = htq_s[str(idx)]["htq"]
        manual_list.append(manual)

        gt = dict()
        for category in ground_truth["annotations"]:
            gt[classes[category["category_id"]]] = category["bbox"]

        automatic, _ = evaluation_utils.htq_card(0.5,
                                                 gt["heart"],
                                                 gt["lung"])
        automatic_list.append(automatic)

        output.append([name, manual, automatic])

    mean, std = evaluation_utils.difference(automatic_list,
                                            manual_list)

    print(f"Mean: {mean}    |    Std: {std}")

    with open(f"{output_path}.csv", "w") as file:
        writer = csv.writer(file, delimiter=";")

        sorted_output = sorted(output, key=lambda x: x[0])
        sorted_output.insert(0, [""])
        sorted_output.insert(0, ["filename", "manual", "automatic"])
        sorted_output.insert(0, [""])
        sorted_output.insert(0, ["Mean", mean, "Std", std])

        writer.writerows(sorted_output)
