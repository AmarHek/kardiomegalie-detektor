import os


def rename(folder):
    """
    Rename files

    :param folder: folder to the files
    """
    folder = os.path.abspath(folder)

    for file in os.listdir(folder):
        new_file = file.split(".")
        new_file = f"{new_file[0].removesuffix('_mask')}.{new_file[1]}"

        os.rename(os.path.join(folder, file), os.path.join(folder, new_file))

        print(f"renamed {file} to {new_file}")


def find_missing(image_folder, mask_folder):
    """
    Find missing masks

    :param image_folder: path to all images
    :param mask_folder: path to the masks
    """
    image_folder = os.path.abspath(image_folder)
    mask_folder = os.path.abspath(mask_folder)

    files = list()
    for file in os.listdir(image_folder):
        if not os.path.isfile(os.path.join(mask_folder, file)):
            files.append(file)

    with open("./missing.txt", "w") as file:
        file.write(f"Count: {len(files)}\n")
        for f in files:
            file.write(f"{f}\n")


def copy_bbox(org_folder, new_folder, files):
    files = [line.split(" ")[0] for line in files.split("\n")]

    for file in files:
        with open(os.path.join(org_folder, file), "r") as xml:
            data = xml.read()

        with open(os.path.join(new_folder, file), "w") as xml:
            xml.write(data)

        print(file)