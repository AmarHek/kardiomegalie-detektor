import csv
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score


def apr_groups(data_path, output_path):
    """
    Evaluates the csv files from Lukas using sklearn.metrics

    :param data_path: csv file
    :param output_path: save metrics to
    """
    with open(data_path, "r") as file:
        reader = csv.reader(file, delimiter=";")

        data = list()
        for date in reader:
            data.append(date)

    net = list()
    eye = list()
    man = list()
    for date in data:
        if date[0].startswith("0"):
            net.append(int(date[1]))
            eye.append(int(date[3]))
            man.append(int(date[5]))

    output = dict()
    output["network"] = dict()

    output["network"]["accuracy"] = accuracy_score(man, net)

    for cls in [0, 1, 2]:
        output["network"][str(cls)] = dict()

        output["network"][str(cls)]["precision"] = precision_score(man, net, average=None, zero_division=0)[cls]
        output["network"][str(cls)]["recall"] = recall_score(man, net, average=None, zero_division=0)[cls]

    #
    output["eye"] = dict()

    output["eye"]["accuracy"] = accuracy_score(man, eye)

    for cls in [0, 1, 2]:
        output["eye"][str(cls)] = dict()

        output["eye"][str(cls)]["precision"] = precision_score(man, eye, average=None, zero_division=0)[cls]
        output["eye"][str(cls)]["recall"] = recall_score(man, eye, average=None, zero_division=0)[cls]

    with open(f"{output_path}.json", "w") as file:
        file.write(json.dumps(output, indent=4))
