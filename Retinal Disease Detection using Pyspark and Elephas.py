# GETTING SPARK

import pyspark

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
conf = SparkConf().setAppName('OCT Image Classification')
sc = SparkContext(conf = conf)
spark = SparkSession(sc)




# IMPORTS

import keras
import numpy as np
import os
import skimage.transform
import tensorflow as tf

from PIL import Image
from datetime import datetime
from elephas.ml_model import ElephasEstimator
from io import BytesIO
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Reshape
from keras.models import Sequential
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.linalg import VectorUDT, Vectors
from pyspark.sql.functions import rand
from pyspark.sql.types import DoubleType, StructField, StructType
from tensorflow.keras import optimizers
from time import time




# CONSTANTS

CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL'] # the classes
IMAGE_SIZE = (64, 64, 3) # input image size of the CNN
IMAGE_PRODUCT = IMAGE_SIZE[0] * IMAGE_SIZE[1] * IMAGE_SIZE[2] # used when the vector is flattened
IS_TEST = False # switching between (test, val) and (train, test)
OUTPUT_PATH = './'
PATH = '../input/octimages/OCT2017/'
TRAIN_PATH = PATH + ('test/' if IS_TEST else 'train/')
TEST_PATH = PATH + ('val/' if IS_TEST else 'test/')

DF_STRUCT = schema = StructType([
    StructField('features', VectorUDT(), True),
    StructField('label', DoubleType(), False)
])




# saving the outputs to a file

outputFile = open(OUTPUT_PATH + 'output-%s.log' % (time()), 'w+')
print('Writing the outputs of this run to: %s' % (outputFile.name))
completeStart = time()
def custom_print(content):
    print(content)
    outputFile.write(str(content) + '\n')




# UTILS

def convert_byte_array(rawdata, classNum):
    img = np.asarray(Image.open(BytesIO(rawdata)))
    img_file = skimage.transform.resize(img, IMAGE_SIZE)
    img_arr = np.asarray(img_file).reshape(IMAGE_PRODUCT)
    return (Vectors.dense(img_arr), float(classNum))

def generate_image_df(path):
    allImagesRdd = sc.parallelize([])

    minCount = None
    for className in CLASS_NAMES:
        count = len([name for name in os.listdir(path + className) if name.endswith('.jpeg')])
        if minCount is None or count < minCount:
            minCount = count

    for classNum in range(len(CLASS_NAMES)):
        imagesRdd = (sc.binaryFiles(path + CLASS_NAMES[classNum] + '/*.jpeg').values()
                     .map(lambda rawData: convert_byte_array(rawData , classNum)))
        allImagesRdd = allImagesRdd.union(imagesRdd.toDF().limit(minCount).rdd)
    return shuffle_and_get_df(allImagesRdd)

def shuffle_and_get_df(rdd, count=None):
    if count:
        rdd = rdd.zipWithIndex().map(lambda x: (x[1]%count, x[0])).groupByKey().mapValues(list).flatMap(lambda x: x[1])
        return spark.createDataFrame(rdd, schema=DF_STRUCT).cache()
    else:
        return spark.createDataFrame(rdd, schema=DF_STRUCT).orderBy(rand()).cache()




# Train dataset
startTime = time()
custom_print('Started reading and transforming training images at %s...' % (str(datetime.now())))
train_df = generate_image_df(TRAIN_PATH)
custom_print('...completed reading and transforming training images in %d seconds.' % (time() - startTime))




# Test dataset
custom_print('\nStarted reading and transforming testing images...')
test_df = generate_image_df(TEST_PATH)
custom_print('...completed reading and transforming testing images.')




# CNN

model = Sequential()
model.add(Dense(IMAGE_PRODUCT, input_dim=IMAGE_PRODUCT, activation='relu'))
model.add(Reshape(IMAGE_SIZE, input_dim=IMAGE_PRODUCT))

model.add(Conv2D(218, kernel_size=(3, 3), activation='relu', input_shape=IMAGE_SIZE))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))




# ESTIMATOR

sgd = optimizers.SGD(learning_rate=0.01)
optimizer_conf = optimizers.serialize(sgd)

estimator = ElephasEstimator()
estimator.set_keras_model_config(model.to_json())
estimator.set_optimizer_config(optimizer_conf)
estimator.set_epochs(25)
estimator.set_batch_size(64)
estimator.set_categorical_labels(True)
estimator.set_validation_split(0.15)
estimator.set_nb_classes(4)
estimator.set_mode('synchronous')
estimator.set_loss('categorical_crossentropy')
estimator.set_metrics(['accuracy'])




# Train the model
startTime = time()
custom_print('\nStarted training the model at %s...' % (str(datetime.now())))
pipeline = Pipeline(stages=[estimator])
pipeline_model = pipeline.fit(train_df)
custom_print('...completed training the model in %d seconds.' % (time() - startTime))




# Predicting the classes of test data

custom_print('\nStarted predicting the classes of Test images...')
prediction = pipeline_model.transform(test_df)
results = prediction.select('label', 'prediction')
custom_print('...completed predicting the classes of Test images.')




prediction_and_label= results.rdd.map(lambda row: (row.label, float(row.prediction.index(max(row.prediction)))))
metrics = MulticlassMetrics(prediction_and_label)




custom_print('\nWeighted Precision: %.2f\n' % (metrics.weightedPrecision))
custom_print('\nWeighted F-score: %.2f\n' % (metrics.weightedFMeasure))




custom_print('\n\n\n\n...completed the whole script in %d seconds.' % (time() - completeStart))
outputFile.close()