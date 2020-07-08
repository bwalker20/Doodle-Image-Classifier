import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.backend.tensorflow_backend import set_session
import os
import io

#load data after running merge_data------------------------

X_train = np.load("./X_train_data.npy", mmap_mode = 'r+')
Y_train = np.load("./Y_train_data.npy", mmap_mode = 'r+')


#define model----------------------------------------------

model = Sequential()

model.add(Convolution2D(32, (3,3), activation='relu', input_shape=(28,28,1), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(40, activation = 'softmax'))

#compile model--------------------------------------------

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit model -----------------------------------------------

print("training model")
model.fit(X_train, Y_train, batch_size = 256, epochs=15, verbose=1)

model.save("./final_model.h5")
