import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def contrast_adjust():
    img = np.array(Image.open("./test_data/image.png").convert("RGB"))
    img2 = img.copy()

    minimum = np.min(img)
    maximum = np.max(img)

    area = img[int(915/4):915-int(915/4), int(914/4):914-int(914/4)]

    plt.imshow(area)
    plt.show()
    plt.close()

    min_val = np.min(area)
    max_val = np.max(area)

    slope = 255 / (max_val - min_val)
    zero = slope * min_val

    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            for k in range(0, 3):
                # if min_val <= img[i][j][k] <= max_val:

                if img[i][j][k] < min_val:
                    img[i][j][k] = 0
                elif img[i][j][k] > max_val:
                    img[i][j][k] = 255
                else:
                    img[i][j][k] = slope * img[i][j][k] - zero

    plt.imshow(img2)
    plt.show()
    plt.close()

    plt.imshow(img)
    plt.show()
    plt.close()
