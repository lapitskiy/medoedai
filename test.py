import time
print("Imported time")

import numpy
print("Imported numpy")

import scipy
print("Imported scipy")

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Is built with CUDA:", tf.test.is_built_with_cuda())
print("Is GPU available:", tf.config.list_physical_devices('GPU'))

import sklearn
print("Imported sklearn")