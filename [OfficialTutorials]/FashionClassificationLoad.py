# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
#from keras.models import load_model

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2 as cv
from PIL import Image
import argmax

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

#print(test_images[0])

test_images = test_images / 255.0

model = keras.models.load_model("FashionClassification_20Steps")

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

#inputtest = cv.imread("imgtest.png")

#inputimg = keras.preprocessing.image.load_img("imgtest.png")
inputimg = np.asarray(Image.open("imgtest.png").convert('L'))

inputimg = 255-inputimg

inputimg = inputimg / 255.0

inputtest = np.array([inputimg,])
#inputtest = keras.preprocessing.image.img_to_array(inputimg)
#inputtest = int(inputtest / 255)

#print(test_images.shape)

#inputtest = np.expand_dims(inputtest, axis=0) 
print(model.predict(test_images)[0])
predict_test = model.predict(inputtest)
print(predict_test)

predict = np.argmax(predict_test,axis=1)  #axis = 1是取行的最大值的索引，0是列的最大值的索引
print(predict)
print("---------")
print("This Item is " + class_names[predict[0]])
#test_loss, test_acc = model.evaluate(inputtest,  test_labels, verbose=2)
#print('\nTest accuracy:', test_acc)