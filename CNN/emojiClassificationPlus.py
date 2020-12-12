import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

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

Expansion_multiple = 5

# Image PreProcess
def ImagePreProcess(Path):
    imgtemp = cv.resize(cv.imread(Path),(256,256),)

    # BRG to GRAY
    #imgtemp = cv.cvtColor(imgtemp,cv.COLOR_RGB2GRAY)
    #imgtemp = 255 - imgtemp

    # BRG to RGB
    imgtemp = cv.cvtColor(imgtemp, cv.COLOR_BGR2RGB)

    #imgtemp = imgtemp / 255.0
    return imgtemp

# Change images
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(256, 
                                                              256,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

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

def LoadTrainImages(Dir,label,multiple):
    Input_image_list = []
    temp_image_list = []
    Output_image_list = []
    Output_label_list = []

    # 将所有图像加载到Input_image_list
    for Files in os.listdir(Dir):
        if(os.path.isfile(Dir + "/" + Files)):
            print("Reading " + Dir + "/" + Files)
            Input_image_list.append(ImagePreProcess(Dir + "/" + Files))

    for i in range(multiple):
      temp_image_list.extend(data_augmentation(np.array(Input_image_list)))
      
      for n in range(len(temp_image_list)):
        Output_image_list.append(temp_image_list[n].numpy().astype("uint8"))
        Output_label_list.append(label)
      
    print(np.array(Output_image_list).shape)
    print(np.array(Output_label_list).shape)
    return Output_image_list,Output_label_list

# Load Train Images
image_list = []
label_list = []

for index in range(len(Class_name)):
    temp_image_list,temp_label_list = LoadTrainImages(Base_Dir + "/" + Source_Dir[index] + '/Train',index,2)
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

print("Input image's Shape is: " + str(train_images.shape))

model = models.Sequential()
# 进行3x3卷积, 输出254*254矩阵
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(4,))

#model = models.Sequential([
#  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(256, 256, 3)),
#  layers.Conv2D(16, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Conv2D(32, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Conv2D(64, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Flatten(),
#  layers.Dense(128, activation='relu'),
#  layers.Dense(4)
#])

print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

with tf.device('/GPU:1'):
  history = model.fit(train_images, train_labels, epochs=8, 
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(test_acc)


# Test Model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
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

model.save("emojiModelPlus")
