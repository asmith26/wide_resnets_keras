from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import os

import logging
logging.basicConfig(level=logging.DEBUG)
import sys
sys.stdout = sys.stderr
# Prevent reaching to maximum recursion depth in `theano.tensor.grad`
#sys.setrecursionlimit(2 ** 20)

import numpy as np
np.random.seed(2 ** 10)

from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ================================================
# DATA CONFIGURATION:
logging.debug("Loading data...")

nb_classes = 10
image_size = 32

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# convert class vectors to binary class matrices
Y_train =to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)
# ================================================

# ================================================
# NETWORK/TRAINING CONFIGURATION:
depth = 40
k = 4
batch_size = 128
sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
# ================================================


logging.debug("Loading pre-trained model...")
'''
# This will work for model saved with updated main.py
model = load_model('WRN-{0}-{1}.h5'.format(depth, k))
'''
model = model_from_json( open( 'WRN-{0}-{1}.json'.format(depth, k) ).read() )
model.load_weights( 'WRN-{0}-{1}.h5'.format(depth, k) )
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])


test_datagen = ImageDataGenerator(
                     featurewise_center=True,
                     featurewise_std_normalization=True,
                     zca_whitening=True)
test_datagen.fit(X_train)


logging.debug("Running testing...")
results = model.evaluate(test_datagen.flow(X_test, Y_test, batch_size=batch_size),
                                   val_samples=X_test.shape[0])

logging.info("Results:")
logging.info("Test loss: {0}".format(results[0]))
logging.info("Test accuracy: {0}".format(results[1]))
