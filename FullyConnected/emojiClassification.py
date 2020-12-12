# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# System API
import os

# Input Config
Base_Dir   = './emoji'
Class_name = ['HuaJi','XiaoKongLong','XiongMaoTou','YingWu']
Source_Dir = ['HuaJi','XiaoKongLong','XiongMaoTou','YingWu']

# Image PreProcess
def ImagePreProcess(Path):
    imgtemp = cv.resize(cv.imread(Path),(512,512),)

    # BRG to GRAY
    #imgtemp = cv.cvtColor(imgtemp,cv.COLOR_RGB2GRAY)
    #imgtemp = 255 - imgtemp

    # BRG to RGB
    imgtemp = cv.cvtColor(imgtemp, cv.COLOR_BGR2RGB)

    #imgtemp = imgtemp / 255.0
    return imgtemp

# Load Images
def LoadImages(Dir,label):
    Input_image_list = []
    Input_label_list = []

    for Files in os.listdir(Dir):
        if(os.path.isfile(Dir + "/" + Files)):
            print("Reading " + Dir + "/" + Files)
            Input_image_list.append(ImagePreProcess(Dir + "/" + Files))
            Input_label_list.append(label)
    return Input_image_list,Input_label_list

# Load Train Images
image_list = []
label_list = []

for index in range(len(Class_name)):
    temp_image_list,temp_label_list = LoadImages(Base_Dir + "/" + Source_Dir[index] + '/Train',index)
    image_list = image_list + temp_image_list
    label_list = label_list + temp_label_list

train_images = np.array(image_list)
train_labels = np.array(label_list)
print(train_images.shape)
print(train_labels.shape)

# Load Test Images
image_list = []
label_list = []

for index in range(len(Class_name)):
    temp_image_list,temp_label_list = LoadImages(Base_Dir + "/" + Source_Dir[index] + '/Test',index)
    image_list = image_list + temp_image_list
    label_list = label_list + temp_label_list

test_images = np.array(image_list)
test_labels = np.array(label_list)

# Configure Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(512, 512, 3)),
    #keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64),
    keras.layers.Dense(128),
    keras.layers.Dense(256),
    keras.layers.Dense(128),
    keras.layers.Dense(128),
    keras.layers.Dense(128),
    keras.layers.Dense(128),
    keras.layers.Dense(128),
    keras.layers.Dense(64),
    keras.layers.Dense(4)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Start Training
history = model.fit(train_images, train_labels, epochs=100)

# Draw Training Message
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('accuracy / loss')
plt.plot(hist['epoch'], hist['accuracy'], 
           label='Train accuracy')
plt.twinx()
plt.plot(hist['epoch'], hist['loss'],'r', 
           label='Train loss')

plt.show()

# Test Model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(Class_name[predicted_label],
                                100*np.max(predictions_array),
                                Class_name[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(4))
  plt.yticks([])
  thisplot = plt.bar(range(4), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

model.save("emojiModel")
