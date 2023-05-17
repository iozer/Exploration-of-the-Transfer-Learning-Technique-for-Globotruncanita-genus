# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:54:06 2022

@author: iozer
"""

from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
#from tf2_resnets import models
from classification_models.resnet import ResNet18 as rn

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, GRU
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,MaxPool2D
from tensorflow.keras.layers import TimeDistributed,Bidirectional,GlobalAveragePooling2D
from tensorflow.keras.layers import LSTM,GRU,SimpleRNN,BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score



img_height = 100
img_width = 100



def  get_images(path): 
   image_list = []
   class_list = []    
   for dirnames in os.listdir(path):
    # print path to all subdirectories first.
    for subdirname in dirnames:
        new_path =os.path.join(path,subdirname)
        for dirname, dirnames1,filenames in os.walk(new_path):
            for filename in filenames:
                img = image_to_vector(os.path.join(new_path,filename))
                image_list.append(img)
                class_list.append(subdirname)
   return np.array(image_list),class_list

def image_to_vector(img_file_path):

    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_file_path, target_size=(img_height, img_width))
    # convert PIL.Image.Image type to 3D tensor with shape (120, 160, 3)
    x = image.img_to_array(img)
    #y = x.reshape(img_height*img_width*3,1)
    #y= y[0 : img_height*img_width]
    #y= y[2*img_height*img_width : 3*img_height*img_width]
    #y = y.reshape(img_height,img_width)
    return x

def create_cnn_model():
    model = Sequential() 
    
    model.add(Conv2D(12, (3, 3), input_shape = (img_height,img_width,3) )) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size =(2, 2))) 
   
    model.add(Flatten())
    model.add(Dense(units=100))

    model.add(Dropout(0.2)) 
    model.add(Dense(3)) 
    model.add(Activation('softmax')) 

    model.compile(loss ='categorical_crossentropy', 
                     optimizer ='adam', 
                   metrics =['accuracy']) 
    return model

def create_cnn_model1():
    model = Sequential() 
    
    model.add(Conv2D(32, (3, 3), input_shape = (img_height,img_width,3) ))
    model.add(Conv2D(32, (3, 3) ))  
    model.add(BatchNormalization())
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
    
    model.add(Conv2D(64, (3, 3) ))
    model.add(Conv2D(64, (3, 3) )) 
    model.add(BatchNormalization())
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
    
    model.add(Conv2D(128, (3, 3) )) 
    model.add(Conv2D(128, (3, 3) )) 
    model.add(BatchNormalization())
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
   
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5)) 

    model.add(Flatten())
    model.add(Dense(units=100))


    model.add(Dense()) 
    model.add(Activation('softmax')) 

    model.compile(loss ='categorical_crossentropy', 
                     optimizer ='adam', 
                   metrics =['accuracy']) 
    return model

def pretrained_resnet18():
    base_model = rn(input_shape=(80,80,3), weights='imagenet', include_top=False)
    flatten_layer = Flatten()
    dense_layer_1 = Dense(256, activation='relu')
    prediction_layer = Dense(3, activation='softmax')
    
    model1 = Sequential([
      base_model,
      flatten_layer,
      dense_layer_1,
      prediction_layer
      ])
    
    model1.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        )
    
    return model1

def pretrained_vgg16():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape = (img_height,img_width,3))
    base_model.trainable = False ## Not trainable weights
    flatten_layer = Flatten()
    dense_layer_1 = Dense(256, activation='relu')
    prediction_layer = Dense(3, activation='softmax')
    
    model = Sequential([
      base_model,
      flatten_layer,
      dense_layer_1,
      prediction_layer
      ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        )
    
    return model

def pretrained_resnet():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape = (img_height,img_width,3))
    base_model.trainable = True ## Not trainable weights
    flatten_layer = Flatten()
    dense_layer_1 = Dense(256, activation='relu')
    prediction_layer = Dense(3, activation='softmax')
    
    model = Sequential([
      base_model,
      flatten_layer,
      dense_layer_1,
      prediction_layer
      ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        )
    
    return model

    
def create_vgg16():
    model = Sequential()
    model.add(Conv2D(input_shape=(80,80,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=3, activation="softmax"))
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def create_cnn_lstm_model():
    model = Sequential() 
   
    model.add(Conv2D(12, (3, 3), input_shape = (img_height,img_width,3) )) 
    model.add(Activation('relu'))
 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
   
    model.add(TimeDistributed(Flatten()))
    #model.add(LSTM(units=100))
    model.add(Bidirectional(LSTM(units=100)))

    model.add(Dropout(0.2)) 
    model.add(Dense(3)) 
    model.add(Activation('softmax')) 

    model.compile(loss ='categorical_crossentropy', 
                     optimizer ='adam', 
                   metrics =['accuracy']) 
    return model



def create_lstm_model():
    model = Sequential() 
   
    model.add(LSTM(units=100)) 
    #model.add(Bidirectional(LSTM(units=100)))

    model.add(Dropout(0.2)) 
    model.add(Dense(2)) 
    model.add(Activation('softmax')) 

    model.compile(loss ='categorical_crossentropy', 
                     optimizer ='adam', 
                   metrics =['accuracy']) 
    return model
    
def create_checkpoint():
    filepath = "best.hdf5"
    checkpoint = ModelCheckpoint(filepath,
                            monitor='val_accuracy',
                            verbose=2,
                            save_best_only=True,
                            mode='max')
    return checkpoint



filePath = 'D:/bilimsel/veri/fosil/all/' 

features,classes2 = get_images(filePath)


features = features / 255
#features = features.reshape(features.shape[0],80,240)
encoder = LabelEncoder()
classes = encoder.fit_transform(classes2)
rnd_seed = 103

classes1 = to_categorical(classes)
x_train,x_test,y_train,y_test = sp(features, classes1,random_state=rnd_seed, test_size=.1)


filepath = "best.hdf5"
np.random.seed(146)



#model = create_lstm_model()
model = create_cnn_model1()
#model = pretrained_resnet()
#model = create_vgg16()
#model = pretrained_vgg16()
#model = pretrained_resnet18()
#model = create_cnn_lstm_model()
checkpoint = create_checkpoint()
print('Train... Model')


history = model.fit(x_train , y_train, epochs=400, batch_size=100, validation_data=(x_test,y_test),verbose=2,callbacks=[checkpoint])


#plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['accuracy'], label='train')
plt.legend()
plt.show()


t = time.time()
model.load_weights("best.hdf5")

y_pred = model.predict(x_test)
elapsed = time.time() - t
print( "Elapsed time: %f seconds.\n" %elapsed )    


conf_mat = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    
roc_auc = roc_auc_score(y_test, y_pred.round(),multi_class="ovr")
acc = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))

print('Accuracy Score: ', acc)
print('Roc_AUC Score :' , roc_auc )























       



print(conf_mat)