# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:02:53 2022

@author: iozer
"""

from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import os

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, GRU
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import TimeDistributed,Bidirectional,GlobalAveragePooling2D
from tensorflow.keras.layers import LSTM,GRU,SimpleRNN,BatchNormalization
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
   for dirname in os.listdir(path):
    # print path to all subdirectories first.
        new_path =os.path.join(path,dirname)
        for dirname, dirnames1,filenames in os.walk(new_path):
            for filename in filenames:
                img = image_to_vector(os.path.join(new_path,filename))
                image_list.append(img)
                class_list.append(dirname)
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
    #model.add(Dropout(0.2))  
    model.add(MaxPooling2D(pool_size =(2, 2))) 
   
    #model.add(TimeDistributed(Flatten()))
    model.add(Flatten())
    model.add(Dense(units=100))

    model.add(Dropout(0.2)) 
    model.add(Dense(3)) 
    model.add(Activation('softmax')) 

    model.compile(loss ='categorical_crossentropy', 
                     optimizer ='adam', 
                   metrics =['accuracy']) 
    return model

def create_cnn_lstm_model():
    model = Sequential() 
   
    model.add(Conv2D(12, (3, 3), input_shape = (img_height,img_width,3) )) 
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
   
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(units=100))
    #model.add(Bidirectional(LSTM(units=100)))

    model.add(Dropout(0.2)) 
    model.add(Dense(3)) 
    model.add(Activation('softmax')) 

    model.compile(loss ='categorical_crossentropy', 
                     optimizer ='adam', 
                   metrics =['accuracy']) 
    return model

def create_cnn_model1():
    model = Sequential() 
    
    model.add(Conv2D(32, (3, 3), input_shape = (img_height,img_width,3),kernel_initializer='VarianceScaling'))
    model.add(Conv2D(32, (3, 3),kernel_initializer='VarianceScaling' ))  
    model.add(BatchNormalization())
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
    
    model.add(Conv2D(64, (3, 3),kernel_initializer='VarianceScaling' ))
    model.add(Conv2D(64, (3, 3),kernel_initializer='VarianceScaling' )) 
    model.add(BatchNormalization())
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
    
    model.add(Conv2D(128, (3, 3),kernel_initializer='VarianceScaling' )) 
    model.add(Conv2D(128, (3, 3),kernel_initializer='VarianceScaling' )) 
    model.add(BatchNormalization())
    model.add(Activation('relu')) 
    model.add(MaxPooling2D(pool_size =(2, 2))) 
   
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5)) 
    

    model.add(Flatten())

    model.add(Dense(units=100, activation='sigmoid'))


    model.add(Dense(3)) 
    model.add(Activation('softmax')) 

    model.compile(loss ='categorical_crossentropy', 
                     optimizer ='adam', 
                   metrics =['accuracy']) 
    return model


def pretrained_Xception():
    base_model = Xception(weights=None, include_top=False, input_shape = (img_height,img_width,3))
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

def pretrained_InceptionV3():
    base_model = InceptionV3(weights="imagenet", include_top=False, input_shape = (img_height,img_width,3))
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

def  pretrained_MobileNet():
    base_model = MobileNet(weights=None, include_top=False, input_shape = (img_height,img_width,3))
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

def create_lstm_model():
    model = Sequential() 
   
    #model.add(LSTM(units=100)) 
    model.add(Bidirectional(LSTM(units=100)))

    model.add(Dropout(0.2)) 
    model.add(Dense(3)) 
    model.add(Activation('softmax')) 

    model.compile(loss ='categorical_crossentropy', 
                     optimizer ='adam', 
                   metrics =['accuracy']) 
    return model
    
def create_checkpoint():
    filepath = "best.hdf5"
    checkpoint = ModelCheckpoint(filepath,
                            monitor='val_loss',
                            verbose=2,
                            save_best_only=True,
                            mode='min')
    return checkpoint



filePath = r'D:\bilimsel\veri\fosil\Formis-Stuarti-Elevata'  


features,classes = get_images(filePath)

features = features / 255

#features = features.reshape(features.shape[0],80,240)
encoder = LabelEncoder()
classes = encoder.fit_transform(classes)


rnd_seed = 116
np.random.seed(117)
kf =StratifiedKFold(n_splits=10, random_state=rnd_seed, shuffle =True)



scores = []
scores_auc = []
confussionMatrix = []
total = np.zeros((3,3))
filepath = "best.hdf5"


for train_index, test_index in kf.split(features,classes): 
    #result = next(kf.split(df), None)
    classes1 = to_categorical(classes)
    x_train_val =features[train_index]
    x_test = features[test_index]
    y_train_val = classes1[train_index]
    y_test = classes1[test_index]
    
    x_train,x_val,y_train,y_val = sp(x_train_val, y_train_val,random_state=5, test_size=.1)
    
    #model = create_lstm_model()
    #model = create_cnn_model1()
    #model = pretrained_resnet()
    #model = pretrained_vgg16()
    #model = pretrained_Xception()
    #model = pretrained_InceptionV3()
    model = pretrained_MobileNet()
    #model = pretrained_vgg16()
    #model = create_cnn_lstm_model()
    checkpoint = create_checkpoint()
    print('Train... Model')
    model.fit(x_train , y_train, epochs=200, batch_size=10, validation_data=(x_val,y_val),verbose=2,callbacks=[checkpoint])
    model.load_weights("best.hdf5")
    y_pred = model.predict(x_test)
    

    conf_mat = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    
    scores_auc.append(roc_auc_score(y_test, y_pred.round(),multi_class="ovr"))
    scores.append(accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
    total = total + conf_mat
    confussionMatrix.append(conf_mat)
print('Scores from each Iteration: ', scores)
print('Average K-Fold Score :' , np.mean(scores))

print(conf_mat)
