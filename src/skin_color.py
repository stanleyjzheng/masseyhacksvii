import cv2
import numpy as np
from deepface import DeepFace
import skimage
from skimage import color
import math

def display_image(image, name):
    window_name = name
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def segment_otsu(image_grayscale, img_BGR):
    threshold_value, threshold_image = cv2.threshold(image_grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #display_image(threshold_image, "otsu")
    threshold_image_binary = 1 - threshold_image / 255
    threshold_image_binary = np.repeat(threshold_image_binary[:, :, np.newaxis], 3, axis=2)
    img_face_only = np.multiply(threshold_image_binary, img_BGR)
    return img_face_only


def estimate_skin(image_path):
    img_BGR = cv2.imread(image_path, 3)

    img_grayscale = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)

    img_face_only = segment_otsu(img_grayscale, img_BGR)
    img_face_only = img_face_only.astype(np.uint8)

    img_HSV = cv2.cvtColor(img_face_only, cv2.COLOR_BGR2HSV)
    img_YCrCb = cv2.cvtColor(img_face_only, cv2.COLOR_BGR2YCrCb)

    blue = []
    green = []
    red = []

    height, width, channels = img_face_only.shape

    for i in range(height):
        for j in range(width):
            if ((img_HSV.item(i, j, 0) <= 170) and (140 <= img_YCrCb.item(i, j, 1) <= 170) and (
                    90 <= img_YCrCb.item(i, j, 2) <= 120)):
                blue.append(img_face_only[i, j].item(0))
                green.append(img_face_only[i, j].item(1))
                red.append(img_face_only[i, j].item(2))
            else:
                img_face_only[i, j] = [0, 0, 0]

    skin_tone_estimate_BGR = [np.mean(blue), np.mean(green), np.mean(red)]
    #hwc

    skin_tone_img = np.array([[skin_tone_estimate_BGR]*height]*width).astype(np.uint8)
    print(skin_tone_img.shape)
    cv2.imwrite('orginal.jpg', img_BGR)
    cv2.imwrite('segment.jpg', img_face_only)
    cv2.imwrite('color.jpg', skin_tone_img)

    obj = DeepFace.analyze(img_path=image_path, actions=['race'])

    return obj['dominant_race']


if __name__ == '__main__':
    print(estimate_skin('../input/img_1.jpg'))
