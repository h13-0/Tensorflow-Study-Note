import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

# Helper libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# System API
import os

# Configure
# 以下内容请和 Train.py 中的匹配
## 源图片的总文件夹位置, 尽量保证输入图片的长宽比为1:1即可, 程序会自动缩放。
Base_Dir   = "../../[SourceImages]/meme"
## 图片的名称, 种类数必须和图片种类数相同, 且和下一个配置项顺序一致。
Class_name = ['HuaJi','XiaoKongLong','XiongMaoTou','YingWu']
## 每种名称对应的文件夹名,路径不能有中文 (文件夹应放到总文件夹下)。
Source_Dir = ['HuaJi','XiaoKongLong','XiongMaoTou','YingWu']
## 导入模型的图片大小, 没必要修改, 但是尽量和训练时设置的一致。
Image_Shape = (299,299)
## 权重文件的位置
Weight_File_Path = "./weights/299.102.hdf5"

# Check Configure
if(len(Class_name) != len(Source_Dir)):
	raise ValueError("'Class_name' and 'Source_Dir' have to correspond.")


# Load Images
def LoadImages(Dir,label):
	Input_image_list = []
	Source_image_list = []
	Input_label_list = []

	for Files in os.listdir(Dir):
		if(os.path.isfile(Dir + "/" + Files)):
			print("Loading " + Dir + "/" + Files)
			img_load = image.load_img(Dir + "/" + Files)
			img = image.img_to_array(img_load)
			img = tf.image.resize(img,Image_Shape)
			Source_image_list.append(img)
			img = preprocess_input(img)
			Input_image_list.append(img)
			Input_label_list.append(label)

	return Input_image_list,Input_label_list,Source_image_list
	
# Load Test Images
image_list = []
label_list = []
source_list = []
for index in range(len(Class_name)):
	temp_image_list,temp_label_list,temp_source_list = LoadImages(Base_Dir + "/" + Source_Dir[index] + '/Test',index)
	image_list = image_list + temp_image_list
	label_list = label_list + temp_label_list
	source_list = source_list + temp_source_list
	
test_images = np.array(image_list)
source_images = np.array(source_list)
test_labels = label_list

# Build Model
model = InceptionResNetV2(include_top=True, weights=Weight_File_Path, input_shape=(Image_Shape[0],Image_Shape[1],3), pooling='avg', classes=len(Class_name))

predictions = model.predict(np.array(test_images))

print(predictions)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(Image.fromarray(np.uint8(img)), cmap=plt.cm.binary)

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
    plot_image(i, predictions[i], test_labels, source_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()














