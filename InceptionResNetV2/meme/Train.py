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

# System API
import os

# Configure
# 以下内容请和 Valid.py 中的匹配
## 源图片的总文件夹位置, 尽量保证输入图片的长宽比为1:1即可, 程序会自动缩放。
Base_Dir   = '../../[SourceImages]/meme'
## 图片的名称, 种类数必须和图片种类数相同, 且和下一个配置项顺序一致。
Class_name = ['HuaJi','XiaoKongLong','XiongMaoTou','YingWu']
## 每种名称对应的文件夹名,路径不能有中文 (文件夹应放到总文件夹下)。
Source_Dir = ['HuaJi','XiaoKongLong','XiongMaoTou','YingWu']
## 导入模型的图片大小, 没必要修改, 但是尽量和使用时设置的一致。
Image_Shape = (299,299)
save_dir='./weights'
filepath="epoch-{epoch:02d}-val_acc-{val_accuracy:.4f}-acc_top3-{acc_top3:.4f}-acc_top5-{acc_top5:.4f}.hdf5"

# Check Configure
if(len(Class_name) != len(Source_Dir)):
	raise ValueError("'Class_name' and 'Source_Dir' have to correspond.")

# Load Images
def LoadImages(Dir,label):
	Input_image_list = []
	Input_label_list = []

	for Files in os.listdir(Dir):
		if(os.path.isfile(Dir + "/" + Files)):
			print("Loading " + Dir + "/" + Files)
			img_load = image.load_img(Dir + "/" + Files)
			img = image.img_to_array(img_load)
			img = tf.image.resize(img,Image_Shape)
			img = preprocess_input(img)
			Input_image_list.append(img)
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
train_labels = label_list

# Load Valid Images
image_list = []
label_list = []

for index in range(len(Class_name)):
    temp_image_list,temp_label_list = LoadImages(Base_Dir + "/" + Source_Dir[index] + '/Valid',index)
    image_list = image_list + temp_image_list
    label_list = label_list + temp_label_list

Valid_images = np.array(image_list)
Valid_labels = label_list

# Build Model
base_model = InceptionResNetV2(include_top=False,  input_shape=(Image_Shape[0],Image_Shape[1],3), weights="imagenet", pooling='avg')
outputs = layers.Dense(len(Class_name), activation='softmax')(base_model.output)
model = keras.Model(base_model.inputs, outputs)

checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(save_dir, filepath),
	monitor='val_acc',verbose=1, 
	save_best_only=False)

def acc_top3(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
  
def acc_top5(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', acc_top3, acc_top5])

# 模型训练
model.fit(train_images, keras.utils.to_categorical(train_labels),
          batch_size=8,
          epochs=10,
          shuffle=True,
          validation_data=(Valid_images, keras.utils.to_categorical(Valid_labels)),
          callbacks=[checkpoint])
