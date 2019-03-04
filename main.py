# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:35:38 2019

@author: tony_
"""

import os, cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils,to_categorical

ALTO = 32
ANCHO = 32
CHANNELS = 3

#cargar imagenes y labels serializadas
images_in=open("images.pickle","rb")
images = pickle.load(images_in)
labels_in=open("labels.pickle","rb")
labels = pickle.load(labels_in)

#parametros modelo
optimizer = RMSprop(lr=1e-4)
objective = 'sparse_categorical_crossentropy'

#creacion modelo de red neuronal
modelo = Sequential()
modelo.add(Conv2D(32, (3, 3), padding='same', input_shape=(ALTO, ANCHO,3), activation='relu'))
modelo.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
modelo.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))

modelo.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
modelo.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
modelo.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))

modelo.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
modelo.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
modelo.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))

modelo.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
modelo.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

modelo.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))
modelo.add(Flatten())
modelo.add(Dense(256, activation='relu'))
modelo.add(Dropout(0.5))

modelo.add(Dense(256, activation='relu'))
modelo.add(Dropout(0.5))

modelo.add(Dense(4))
modelo.add(Activation('softmax'))

#compilacion modelo
modelo.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

#parametros entrenamiento
nb_epoch = 20
batch_size = 16

modelo.fit(images, labels, batch_size=batch_size, epochs=nb_epoch,validation_split=0.25, verbose=1, shuffle=True)

#guardar el modelo
modelo.save('traffic_model.h5') 

prediccion=modelo.predict(np.array(images[0]).reshape(-1,32,32,3), verbose=1)
print(np.argmax(prediccion))