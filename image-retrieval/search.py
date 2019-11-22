# python search.py -i dataset/train/ukbench00000.jpg

import argparse as ap
import cv2
import joblib
from scipy.cluster.vq import vq
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-i", "--image", help="Path to query image", required=True)
args = vars(parser.parse_args())

# Get query image path
image_path = args["image"]

# Load the classifier, class names, scaler, number of clusters and vocabulary 
im_features, image_paths, idf, numWords, voc = joblib.load("bof.pkl")

# Create feature extraction and keypoint detector objects
fea_det = cv2.xfeatures2d.SIFT_create()


def search(path, show_results=False):
    im = cv2.imread(path)
    kpts, des = fea_det.detectAndCompute(im, None)

    # rootsift
    # from rootsift import RootSIFT
    # rs = RootSIFT()
    # des = rs.compute(kpts, des)

    test_features = np.zeros((1, numWords), "float32")
    words, distance = vq(des, voc)
    for w in words:
        test_features[0][w] += 1

    # Perform Tf-Idf vectorization and L2 normalization
    test_features = test_features * idf
    test_features = preprocessing.normalize(test_features, norm='l2')

    score = np.dot(test_features, im_features.T)
    rank_ID = np.argsort(-score)

    if show_results:
        # Visualize the results
        plt.figure()
        plt.gray()
        plt.subplot(5, 4, 1)
        plt.imshow(im[:, :, ::-1])
        plt.axis('off')

        for i, ID in enumerate(rank_ID[0][0:16]):
            img = cv2.imread(image_paths[ID])
            plt.gray()
            plt.subplot(5, 4, i + 5)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')

        plt.show()

    return image_paths[rank_ID[0][0]]


def calc_accuracy():
    num = 0
    for path in image_paths:
        if path == search(path):
            num += 1
    return num / len(image_paths)


if __name__ == '__main__':
    search(image_path, show_results=True)
    print('The accuracy is %.3f' % calc_accuracy())
