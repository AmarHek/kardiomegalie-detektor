import os
import json

from PIL import Image


def convert_htq_csv(csv_path: str, dataset_path: str):
    """
    Convert the csv file with the HTQs from Lukas to an machine readable json file

    :param csv_path: path to the csv file
    :param dataset_path: path to the corresponding dataset
    """
    with open(dataset_path, "r") as file:
        ground_truth_raw = json.loads(file.read())

    names_ids = {
        image["file_name"]: image["image_id"]
        for image in ground_truth_raw
    }

    with open(csv_path, "r") as file:
        data = file.read()

    name_htq = list()
    for line in data.split("\n"):
        for item in line.split(";;;"):
            name_htq.append(item)

    output = dict()
    for item in name_htq:
        if item != "":
            item = item.removeprefix(";")
            name, htq = item.split(";")[:2]

            if names_ids.get(f"{name}.png") is not None:
                output[names_ids[f"{name}.png"]] = dict()

                output[names_ids[f"{name}.png"]]["file_name"] = f"{name}.png"
                output[names_ids[f"{name}.png"]]["htq"] = float(htq.replace(",", "."))
            else:
                print(f"error with {name}.png")

    with open("./ground_truth_htq.json", "w") as file:
        file.write(json.dumps(output, indent=4))


def convert_img(image_folder, save_folder):
    image_folder = os.path.abspath(image_folder)
    save_folder = os.path.abspath(save_folder)

    for file in os.listdir(image_folder):
        save_path = os.path.join(save_folder, file)
        save_path = save_path.removesuffix(f".{os.path.basename(save_path).split('.')[-1]}")

        image = Image.open(os.path.join(image_folder, file))
        image.save(f"{save_path}.png", format="png")

        print(f"converted {file}")
