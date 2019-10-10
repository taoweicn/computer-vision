import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


def harris(images):
    fig = plt.figure()
    fig.suptitle('Detected Harris key points')

    for index, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)  # 图像膨胀
        img2 = img.copy()
        img2[dst > 0.001 * dst.max()] = [255, 0, 0]  # 标记特征点

        ax = fig.add_subplot(2, 3, index + 1)
        ax.set_title('img' + str(index + 1))
        ax.imshow(img2)
        ax.axis('off')


def FAST(images):
    fig = plt.figure()
    fig.suptitle('Detected FAST key points')

    for index, img in enumerate(images):
        fast = cv2.FastFeatureDetector_create()
        if index > 2:
            fast.setNonmaxSuppression(0)
        key_points = fast.detect(img, None)
        img2 = cv2.drawKeypoints(img, key_points, None, color=(0, 255, 0))

        ax = fig.add_subplot(2, 3, index + 1)
        ax.set_title('img' + str(index + 1))
        ax.imshow(img2)
        ax.axis('off')


def MSER(images):
    fig = plt.figure()
    fig.suptitle('Detected MSER key points')

    for index, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mser = cv2.MSER_create(_min_area=300)

        regions, boxes = mser.detectRegions(gray)

        img2 = img.copy()
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 0), 2)

        ax = fig.add_subplot(2, 3, index + 1)
        ax.set_title('img' + str(index + 1))
        ax.imshow(img2)
        ax.axis('off')


def main():
    # Viewpoint
    image_paths = glob.glob(os.path.join('./images/graf', '*.ppm'))
    image_paths.sort()
    images = [cv2.imread(path) for path in image_paths]

    harris(images)
    FAST(images)
    MSER(images)

    # Blur
    image_paths = glob.glob(os.path.join('./images/trees', '*.ppm'))
    image_paths.sort()
    images = [cv2.imread(path) for path in image_paths]

    harris(images)
    FAST(images)
    MSER(images)
    plt.show()


if __name__ == '__main__':
    main()
