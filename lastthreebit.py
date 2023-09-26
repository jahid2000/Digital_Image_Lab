import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_image(image, text, subplot):
    plt.subplot(1, 3, subplot)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(text)

original_image = cv2.resize(cv2.imread('Jack.jpg', cv2.IMREAD_GRAYSCALE), (512, 512))
plot_image(original_image, "Original Image", 1)

three_bit_image = (original_image >> 5) << 5
plot_image(three_bit_image, "Image using last 3 bits", 2)


image = cv2.absdiff(np.uint8(original_image), np.uint8(three_bit_image))
plot_image(image, "Difference Image", 3)

plt.show()