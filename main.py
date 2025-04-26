import tensorflow as tf
from keras.src.utils import img_to_array
from keras.src.utils.image_dataset_utils import load_image
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import cv2
import kagglehub
import pandas as pd

import csv

dir = kagglehub.dataset_download("a2015003713/militaryaircraftdetectiondataset")

folds = os.listdir(dir)

labels = pd.read_csv(os.path.join(dir, folds.pop(len(folds)-1)))


trainData = labels[labels['split'] == 'train']
validationData = labels[labels['split'] == 'validation']
testData = labels[labels['split'] == 'test']

dataSet = os.path.join(dir, folds.pop(0))

classes = {}
classCount = 0

def parse_label(filename, labels_df):
    row = labels_df[labels_df['filename'] == filename]
    if not row.empty:
        xmin, ymin, xmax, ymax, classLabe = row.iloc[0][['xmin', 'ymin', 'xmax', 'ymax', 'class']]
        return tf.convert_to_tensor([xmin, ymin, xmax, ymax, classLabe], dtype=tf.float32)
    return tf.zeros((5,))

def loadData(data_df, batch_size=32):
    file_paths = [os.path.join(dataSet, f'{filename}.jpg') for filename in data_df['filename']]
    labes = [parse_label(filename, data_df) for filename in data_df['filename']]

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labes))

    for index, row in data.iterrows():
        imagePath = row['filename']
        width = row['width']
        height = row['height']
        image = load_image(os.path.join(dataSet, imagePath + '.jpg'), image_size=(224, 224), num_channels=3, interpolation='nearest', data_format='channels_last')
        img = img_to_array(image)
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        classLabel = row['class']
        if classLabel not in classes:
            classes[classLabel] = classCount
            classCount += 1
        imagesArr.append(img)
        labelArr.append([xmin, ymin, xmax, ymax, classLabel])

    return tf.data.Dataset.from_tensor_slices((imagesArr, labelArr))

trainDataSet = loadData(trainData)
# validationDataSet = loadData(validationData)
# testDataSet = loadData(testData)

print(trainDataSet)
print(folds)