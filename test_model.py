import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from numpy import load

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

img_test = load('img_test.npy')
label_test = load('label_test.npy')

hypermodel = tf.keras.models.load_model('saved_model/my_model')

eval_result = hypermodel.evaluate(img_test, label_test)
print("[test loss, test accuracy]:", eval_result)


