# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:09:22 2020

@author: Shaon
"""
#import tensorflow as tf
#from tensorflow.keras.applications.inception_v3 import InceptionV3
#
#pre_trained_model = InceptionV3(input_shape = (150,150,3),
#                                    include_top = False,
#                                    weights = 'imagenet')
##make all the layers non-trainable, (may lead to overfit though)                               )
#for layer in pre_trained_model.layers:
#    layer.trainable = False
##flatten the output layer to 1 dimention    
#x = tf.keras.layers.Flatten()(pre_trained_model.output)
##adding a fully connected layer with 1024 hidden units              )
#x = tf.keras.layers.Dense(1024, activation = 'relu') (x)
##adding a droupour rate of 0.2
#x = tf.keras.layers.Dropout(0.2) (x)
##adding a final softmax layer for classification
#x = tf.keras.layers.Dense(4, activation = 'softmax')(x)
#
#model = tf.keras.Model(pre_trained_model.input,x)
#model.compile(optimiser = tf.keras.optimizers.RMSprop(lr=0.0001),
#              loss = 'categorical_crossentropy',
#              metrics=['acc'])
#model.summary()
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import os
import keras
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
#from keras.applications import Xception
#from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
#import matplotlib.pyplot as plt
#import keras_metrics as km



# Resnet
#from keras.applications.resnet50 import ResNet50
#from keras.preprocessing import image
#from keras.applications.resnet50 import preprocess_input, decode_predictions

# Here's our 4 categories that we have to classify.
class_names = ['Bloodroot', 'Clubmoss', 'Dandelion', 'Lobelia']
# Traing Dataset
trainset = [class_names[0],class_names[1],class_names[2],class_names[3]] 
mypath = "F:/Akshay/Downloads"
for i in range(len(class_names)):
    trainset[i] = os.path.join(mypath,'Data/Train/',class_names[i])
print("Size of each class of Training Set: ")
for i in range(len(trainset)):
    print(class_names[i],' has ',len(os.listdir(trainset[i])), 'instances.')

#Validation Dataset
validationset = [class_names[0],class_names[1],class_names[2],class_names[3]] 
for i in range(len(class_names)):
    validationset[i] = os.path.join(mypath,'Data/Validation/',class_names[i])
print("\nSize of each class of Validation Set: ")
for i in range(len(validationset)):
    print(class_names[i],' has ',len(os.listdir(validationset[i])), 'instances.')

#Test Dataset
testset = [class_names[0],class_names[1],class_names[2],class_names[3]] 
for i in range(len(class_names)):
    testset[i] = os.path.join(mypath,'Data/Test/',class_names[i])
print("\nSize of each class of Test Set: ")
for i in range(len(testset)):
    print(class_names[i],' has ',len(os.listdir(validationset[i])), 'instances.')
    
#plotting the pie chart to check the proporstion of train, validation and test set .
sizes_trainset = []
for i in range(len(trainset)):
    sizes_trainset.append(len(os.listdir(trainset[i])))
explode = (0, 0, 0, 0)  
plt.pie(sizes_trainset, explode=explode, labels=class_names,
autopct='%1.1f%%', shadow=True, startangle=150)
plt.axis('equal')
plt.title('Proportion of each observed category of training set')
plt.show()

sizes_validationset = []
for i in range(len(trainset)):
    sizes_validationset.append(len(os.listdir(validationset[i])))
explode = (0, 0, 0, 0)  
plt.pie(sizes_validationset, explode=explode, labels=class_names,
autopct='%1.1f%%', shadow=True, startangle=150)
plt.axis('equal')
plt.title('Proportion of each observed category of validation set')
plt.show()

sizes_testset = []
for i in range(len(testset)):
    sizes_testset.append(len(os.listdir(testset[i])))
explode = (0, 0, 0, 0)  
plt.pie(sizes_testset, explode=explode, labels=class_names,
autopct='%1.1f%%', shadow=True, startangle=150)
plt.axis('equal')
plt.title('Proportion of each observed category of test set')
plt.show()

#Data Preprocessing
from keras.preprocessing.image import ImageDataGenerator

#Without Augmentation
#train_datagen = ImageDataGenerator(rescale = 1./255)

#Augmentation
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=40,
                                   width_shift_range =0.2,
                                   height_shift_range=0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)



                                   
validation_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

BATCH_SIZE = 25
IMG_HEIGHT = 150
IMG_WIDTH = 150
total_train_images = 0;
for i in range(len(trainset)):
    total_train_images += len(os.listdir(trainset[i]))

STEPS_PER_EPOCH = np.ceil(total_train_images/BATCH_SIZE)

training_set = train_datagen.flow_from_directory(directory = mypath+'/Data/Train', #directory=str(data_dir)
                                                 target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                 batch_size = BATCH_SIZE,
                                                 class_mode = 'categorical')


validation_set = validation_datagen.flow_from_directory(directory = mypath+'/Data/Validation',
                                                 target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                 batch_size = BATCH_SIZE,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(directory = mypath+'/Data/Test',
                                            target_size = (IMG_HEIGHT, IMG_WIDTH),
                                            batch_size = BATCH_SIZE,
                                            class_mode = 'categorical')

#displaying a portion of random batch images from the dataset
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for i in range(25):
      plt.subplot(5,5,i+1)
      plt.imshow(image_batch[i])
      for j in range(len(class_names)):
          if(label_batch[i][j]==1):
              plt.title(class_names[j])
              break;
      plt.axis('off')

image_batch, label_batch = next(training_set)
show_batch(image_batch, label_batch)



#-----------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------


# create the base pre-trained model
#using InceptV3
base_model = InceptionV3(weights='imagenet', include_top=False)

#Temporary
#base_model = keras.applications.xception.Xception(include_top=False, weights='imagenet')
#base_model = ResNet50(weights='imagenet', include_top=False)
#base_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')

# =============================================================================
# from keras.applications.vgg19 import VGG19, preprocess_input
# vgg19_model = VGG19(weights = 'imagenet', include_top = False)
# x = vgg19_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
# predictions = Dense(4, activation = 'softmax')(x)
# model = Model(input = vgg19_model.input, output = predictions)
# 
# =============================================================================
#using Mobilenet V2
#base_model = keras.applications.mobilenet_v2.MobileNetV2(include_top=True, weights='imagenet')
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(4, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

#model.summary()
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
#for layer in base_model.layers:
#    layer.trainable = False
#
## compile the model (should be done *after* setting layers to non-trainable)
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
# let's visualize layer names and layer indices to see how many layers
# we should freeze:

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True
# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate

from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.0001), 
              loss='categorical_crossentropy', 
              metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),'accuracy'])




from keras import optimizers
# =============================================================================
# sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),'accuracy'])
# 
# 
# 
adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),'accuracy'])
# 
# ==========================================================-------------===================


#Traing the model 


#early_stopping = tf.keras.callbacks.EarlyStopping(patience =2) # import from tensorflow
early_stopping = keras.callbacks.callbacks.EarlyStopping(patience =3)  #import from keras
history = model.fit_generator(training_set,
                         steps_per_epoch = STEPS_PER_EPOCH, #total_training_images/batch_size
                         epochs = 15,
                         validation_data = validation_set,
                         callbacks=[early_stopping])


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
#plot training and validation accuracy values
axes[0].set_ylim(0,1)
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
#plot training and validation loss values
#axes[1].set_ylim(0,1)
axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
plt.tight_layout()
plt.show()

fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
#plot training and validation precision values
axes1[0].set_ylim(0,1)
axes1[0].plot(history.history['precision'], label='Train')
#axes1[0].plot(history.history['val_precision_1'], label='Validation')
axes1[0].plot(history.history['val_precision'], label='Validation')
axes1[0].set_title('Model Precision')
axes1[0].set_xlabel('Epoch')
axes1[0].set_ylabel('Precision')
axes1[0].legend()
#plot training and validation recall values
axes1[0].set_ylim(0,1)
axes1[1].plot(history.history['recall'], label='Train')
#axes1[1].plot(history.history['val_recall_1'], label='Validation')
axes1[1].plot(history.history['val_recall'], label='Validation')
axes1[1].set_title('Model Recall')
axes1[1].set_xlabel('Epoch')
axes1[1].set_ylabel('Recall')
axes1[1].legend()
plt.tight_layout()
plt.show()
print("Evaluate on test data")
test_loss, test_precision, test_recall,  test_accuracy  = model.evaluate(test_set, verbose=0)
print("Test Loss: {0:.2f}, Test Precision: {1:.2f}, Test Recall: {2:.2f}, Test Accuracy: {3:.2f}%".format(test_loss,test_precision,test_recall, test_accuracy*100))

#from sklearn import metrics
#import numpy as np
#predictions = model.predict_generator(test_set)
#val_preds = np.argmax(predictions, axis=-1)
#val_trues = test_set.classes
#cm = metrics.confusion_matrix(val_trues, val_preds)
#labels = test_set.class_indices.keys()
##print(labels)
##precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(val_trues, val_preds)
#print(metrics.classification_report(val_trues, val_preds))
#i=0
#for key in labels:
#    print('Class {0} : {1}'.format(i, key))
#    i+=1
    

#path = 'CanadianMedicinalPlantDataset/bloodroot'
#from glob import glob
#import os
#
#def clean_directory(dir_path, ext=".jpg"):
#    files = glob(os.path.join(dir_path, ".*" + ext))  # this line find all files witch starts with . and ends with given extension
#    for file_path in files:
#        print(file_path)
#        os.remove(file_path)
#        
#clean_directory('CanadianMedicinalPlantDataset/Train/bloodroot')







# =============================================================================
# #for adam
# fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
# #plot training and validation precision values
# axes1[0].set_ylim(0,1)
# axes1[0].plot(history.history['precision_2'], label='Train')
# #axes1[0].plot(history.history['val_precision_1'], label='Validation')
# axes1[0].plot(history.history['val_precision_2'], label='Validation')
# axes1[0].set_title('Model Precision')
# axes1[0].set_xlabel('Epoch')
# axes1[0].set_ylabel('Precision')
# axes1[0].legend()
# #plot training and validation recall values
# axes1[0].set_ylim(0,1)
# axes1[1].plot(history.history['recall_2'], label='Train')
# #axes1[1].plot(history.history['val_recall_1'], label='Validation')
# axes1[1].plot(history.history['val_recall_2'], label='Validation')
# axes1[1].set_title('Model Recall')
# axes1[1].set_xlabel('Epoch')
# axes1[1].set_ylabel('Recall')
# axes1[1].legend()
# plt.tight_layout()
# plt.show()
# print("Evaluate on test data")
# test_loss, test_precision, test_recall,  test_accuracy  = model.evaluate(test_set, verbose=0)
# print("Test Loss: {0:.2f}, Test Precision: {1:.2f}, Test Recall: {2:.2f}, Test Accuracy: {3:.2f}%".format(test_loss,test_precision,test_recall, test_accuracy*100))
# 
# =============================================================================




fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
#plot training and validation precision values
axes1[0].set_ylim(0,1)
axes1[0].plot(history.history['precision_3'], label='Train')
#axes1[0].plot(history.history['val_precision_1'], label='Validation')
axes1[0].plot(history.history['val_precision_3'], label='Validation')
axes1[0].set_title('Model Precision')
axes1[0].set_xlabel('Epoch')
axes1[0].set_ylabel('Precision')
axes1[0].legend()
#plot training and validation recall values
axes1[0].set_ylim(0,1)
axes1[1].plot(history.history['recall_3'], label='Train')
#axes1[1].plot(history.history['val_recall_1'], label='Validation')
axes1[1].plot(history.history['val_recall_3'], label='Validation')
axes1[1].set_title('Model Recall')
axes1[1].set_xlabel('Epoch')
axes1[1].set_ylabel('Recall')
axes1[1].legend()
plt.tight_layout()
plt.show()
print("Evaluate on test data")
test_loss, test_precision, test_recall,  test_accuracy  = model.evaluate(test_set, verbose=0)
#print("Test Loss: {0:.2f}, Test Precision: {1:.2f}, Test Recall: {2:.2f}, Test Accuracy: {3:.2f}%"
print("Test Loss: {0:.2f}, Test Precision: {1:.2f}, Test Recall: {2:.2f}, Test Accuracy: {3:.2f}%".format(test_loss,test_precision,test_recall, test_accuracy*100))
# 