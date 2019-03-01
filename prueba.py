import os, cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils

ALTO = 64
ANCHO = 64
CHANNELS = 3

train_dir="./train/"
test_dir="./test/"

train_images = [train_dir+i for i in os.listdir(train_dir)]
test_images =  [test_dir+i for i in os.listdir(test_dir)]

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ALTO, ANCHO), interpolation=cv2.INTER_CUBIC)



def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ALTO, ANCHO), dtype=np.uint8)
    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i%5 == 0: print('Processed {} of {}'.format(i, count))
    return data


train = prep_data(train_images)
test = prep_data(test_images)

labels = []
for i in train_images:
    if 'pare' in i:
        labels.append(1)
    else:
        labels.append(0)
    

optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'

def catdog():
    modelo = Sequential()

    modelo.add(Conv2D(32, (3, 3), padding='same', input_shape=(3, ALTO, ANCHO), activation='relu'))
    modelo.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    modelo.add(MaxPooling2D(data_format="channels_first", pool_size=(2, 2)))

    modelo.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    modelo.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    modelo.add(MaxPooling2D(data_format="channels_first", pool_size=(2, 2)))
    
    modelo.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    modelo.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    modelo.add(MaxPooling2D(data_format="channels_first", pool_size=(2, 2)))
    
    modelo.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    modelo.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

    modelo.add(MaxPooling2D(data_format="channels_first", pool_size=(2, 2)))
    modelo.add(Flatten())
    modelo.add(Dense(256, activation='relu'))
    modelo.add(Dropout(0.5))
    
    modelo.add(Dense(256, activation='relu'))
    modelo.add(Dropout(0.5))

    modelo.add(Dense(1))
    modelo.add(Activation('sigmoid'))

    modelo.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return modelo

model = catdog()
    
nb_epoch = 20
batch_size = 16

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

def entrenar():
    
    history = LossHistory()
    model.fit(train, labels, batch_size=batch_size, epochs=nb_epoch,
              validation_split=0.25, verbose=1, shuffle=True, callbacks=[history, early_stopping])
    predicciones = model.predict(test, verbose=0)
    return predicciones, history

predicciones, history = entrenar()

for i in range(0,5):
    if predicciones[i, 0] >= 0.5: 
        print('I am {:.2%} esto es un pare'.format(predicciones[i][0]))
    else: 
        print('I am {:.2%} esto es un prohibido'.format(1-predicciones[i][0]))
        
    plt.imshow(test[i].T)
    plt.show()