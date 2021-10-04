import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def relu():
    plt.plot([-1, 0, 1], [0, 0, 1])

    plt.xlabel("z")
    plt.ylabel("g(z)=max{0,z}")

    plt.savefig("./relu.svg",
                bbox_inches="tight",
                format="svg")

    plt.show()
    plt.close()


def edge_detec(image_path):
    img = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
    output = np.empty(img.shape, dtype=np.float32)

    for y, row in enumerate(img):
        for x, px in enumerate(row):
            output[y, x] = img[y, x] - img[y, (x+1) % row.shape[0]]

    area = output[int(output.shape[0] / 4):output.shape[0] - int(output.shape[0] / 4),
                  int(output.shape[1] / 4):output.shape[1] - int(output.shape[1] / 4)]

    max_val = np.max(area)
    min_val = np.min(area)

    # max_val = np.max(output)
    # min_val = np.min(output)

    slope = 255 / (max_val - min_val)
    zero = slope * min_val

    for y, row in enumerate(img):
        for x, px in enumerate(row):
            if output[y][x] < min_val:
                output[y][x] = 0
            elif output[y][x] > max_val:
                output[y][x] = 255
            else:
                output[y][x] = slope * output[y][x] - zero

    output = Image.fromarray(output).convert("RGB")
    # output.save(os.path.join(os.path.dirname(image_path), "edge(contrast).png"), format="png")

    plt.imshow(output)
    plt.show()
    plt.close()
