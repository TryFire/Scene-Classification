# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage import transform

problem_title = 'Scene Classification'
_target_column_name = 'type'
_prediction_label_names = [0, 1, 2, 3, 4, 5] 
name = ["building", "forest", "snow","mountain", "sea view", "street"]


# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.Accuracy(name='acc'),
    rw.score_types.BalancedAccuracy(name='balanced_acc')
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=57)
    return cv.split(X, y)

def get_data(df, dirname):
    

    transform_torch = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    images = []
    for filename in df['image_name']:
        
        if not filename.endswith('.jpg'):
            continue
        image_path = os.path.join(dirname, filename)

        image = plt.imread(image_path)
        image = image.astype(np.float32)
        if image.shape != (150,150,3):
            image = transform.resize(image, (150,150,3))

        image = transform_torch(image)
        image = image.numpy()
        images.append(image)
    
    labels = np.array(list(df['label']))
#     ll = np.zeros((len(labels), 6))
#     for i in range(len(labels)):
#         ll[i, labels[i]] = 1
    return np.array(images), labels

def _read_data(path, f_name):
    train_data_label = pd.read_csv(os.path.join('data', f_name+'.csv'))
    images_path = os.path.join('data', f_name)
    images, labels = get_data(train_data_label, images_path)

    return images, labels


def get_test_data(path='.'):
    f_name = 'test'
    images, labels = _read_data(path, f_name)
    
    return images, labels


def get_train_data(path='.'):
    f_name = 'train'
    return _read_data(path, f_name)
