# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:36:48 2021

@author: samgao1999
"""
import os
import struct
import numpy as np
import pandas as pd

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels



if __name__ == "__main__":
    train_images, train_labels = load_mnist(r"C:\Users\samgao1999\Desktop\机器学习\experiments\exp4", kind="train")
    test_images, test_labels = load_mnist(r"C:\Users\samgao1999\Desktop\机器学习\experiments\exp4", kind="t10k")
    X_train = pd.DataFrame(train_images, header=)