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
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Preprocess Data-----------------------
#categories = ["cup" , "fish", "fork", "ladder", "tree", "airplane", "donut", "face", "house", "saw", "tent", "sun", "moon", "dog", "table"]
#"cup" , "fish", "fork", "ladder", "tree", 
#"airplane", "donut", "face", "house", "saw",
#"tent", "sun", "moon", "dog", "table"
data_cup = np.load("../google_data/cup.npy", mmap_mode = 'r+')
data_fish = np.load("../google_data/fish.npy", mmap_mode = 'r+')
data_fork = np.load("../google_data/fork.npy", mmap_mode = 'r+')
data_ladder = np.load("../google_data/ladder.npy", mmap_mode = 'r+')
data_tree = np.load("../google_data/tree.npy", mmap_mode = 'r+')
data_airplane = np.load("../google_data/airplane.npy", mmap_mode = 'r+')
data_donut = np.load("../google_data/donut.npy", mmap_mode = 'r+')
data_face = np.load("../google_data/face.npy", mmap_mode = 'r+')
data_house = np.load("../google_data/house.npy", mmap_mode = 'r+')
data_saw = np.load("../google_data/saw.npy", mmap_mode = 'r+')
data_tent = np.load("../google_data/tent.npy", mmap_mode = 'r+')
data_sun = np.load("../google_data/sun.npy", mmap_mode = 'r+')
data_moon = np.load("../google_data/moon.npy", mmap_mode = 'r+')
data_dog = np.load("../google_data/dog.npy", mmap_mode = 'r+')
data_table = np.load("../google_data/table.npy", mmap_mode = 'r+')
#15 from here
data_eye = np.load("../google_data/eye.npy", mmap_mode = 'r+')
data_pear = np.load("../google_data/pear.npy", mmap_mode = 'r+')
data_sword = np.load("../google_data/sword.npy", mmap_mode = 'r+')
data_telephone = np.load("../google_data/telephone.npy", mmap_mode = 'r+')
data_tornado = np.load("../google_data/tornado.npy", mmap_mode = 'r+')
#20----------
data_pool = np.load("../google_data/pool.npy", mmap_mode = 'r+')
data_stopsign = np.load("../google_data/stop_sign.npy", mmap_mode = 'r+')
data_oven = np.load("../google_data/oven.npy", mmap_mode = 'r+')
data_bicycle = np.load("../google_data/bicycle.npy", mmap_mode = 'r+')
data_fan = np.load("../google_data/fan.npy", mmap_mode = 'r+')
#25
data_line = np.load("../google_data/line.npy", mmap_mode = 'r+')
data_key = np.load("../google_data/key.npy", mmap_mode = 'r+')
data_waterslide = np.load("../google_data/waterslide.npy", mmap_mode = 'r+')
data_tshirt = np.load("../google_data/t-shirt.npy", mmap_mode = 'r+')
data_purse = np.load("../google_data/purse.npy", mmap_mode = 'r+')
#30
data_axe = np.load("../google_data/axe.npy", mmap_mode = 'r+')
data_nose = np.load("../google_data/nose.npy", mmap_mode = 'r+')
data_belt = np.load("../google_data/belt.npy", mmap_mode = 'r+')
data_steak = np.load("../google_data/steak.npy", mmap_mode = 'r+')
data_beach = np.load("../google_data/beach.npy", mmap_mode = 'r+')
#35
data_mushroom = np.load("../google_data/mushroom.npy", mmap_mode = 'r+')
data_shovel = np.load("../google_data/shovel.npy", mmap_mode = 'r+')
data_spoon = np.load("../google_data/spoon.npy", mmap_mode = 'r+')
data_eiffeltower = np.load("../google_data/The_Eiffel_Tower.npy", mmap_mode = 'r+')
data_zigzag = np.load("../google_data/zigzag.npy", mmap_mode = 'r+')
#40
#Check min size
min_size = min(len(data_cup), len(data_fish), len(data_fork), len(data_ladder), len(data_tree),
  len(data_airplane), len(data_donut), len(data_face), len(data_house), len(data_saw),
  len(data_tent), len(data_sun), len(data_moon), len(data_dog), len(data_table), len(data_eye),
  len(data_pear), len(data_sword), len(data_telephone), len(data_tornado), len(data_pool),
  len(data_stopsign), len(data_oven), len(data_bicycle), len(data_fan), len(data_line),
  len(data_key), len(data_waterslide), len(data_tshirt), len(data_purse), len(data_axe), 
  len(data_nose), len(data_belt), len(data_steak), len(data_beach), len(data_mushroom), 
  len(data_shovel), len(data_spoon), len(data_eiffeltower), len(data_zigzag))

print(min_size)

data_cup = np.delete(data_cup, range(min_size, len(data_cup)), axis = 0)
data_fish = np.delete(data_fish, range(min_size, len(data_fish)), axis = 0)
data_fork = np.delete(data_fork, range(min_size, len(data_fork)), axis = 0)
data_ladder = np.delete(data_ladder, range(min_size, len(data_ladder)), axis = 0)
data_tree = np.delete(data_tree, range(min_size, len(data_tree)), axis = 0)
data_airplane = np.delete(data_airplane, range(min_size, len(data_airplane)), axis = 0)
data_donut = np.delete(data_donut, range(min_size, len(data_donut)), axis = 0)
data_face = np.delete(data_face, range(min_size, len(data_face)), axis = 0)
data_house = np.delete(data_house, range(min_size, len(data_house)), axis = 0)
data_saw = np.delete(data_saw, range(min_size, len(data_saw)), axis = 0)
data_tent = np.delete(data_tent, range(min_size, len(data_tent)), axis = 0)
data_sun = np.delete(data_sun, range(min_size, len(data_sun)), axis = 0)
data_moon = np.delete(data_moon, range(min_size, len(data_moon)), axis = 0)
data_dog = np.delete(data_dog, range(min_size, len(data_dog)), axis = 0)
data_table = np.delete(data_table, range(min_size, len(data_table)), axis = 0)
data_eye = np.delete(data_eye, range(min_size, len(data_eye)), axis = 0)
data_pear = np.delete(data_pear, range(min_size, len(data_pear)), axis = 0)
data_sword = np.delete(data_sword, range(min_size, len(data_sword)), axis = 0)
data_telephone = np.delete(data_telephone, range(min_size, len(data_telephone)), axis = 0)
data_tornado = np.delete(data_tornado, range(min_size, len(data_tornado)), axis = 0)
data_pool = np.delete(data_pool, range(min_size, len(data_pool)), axis = 0)
data_stopsign = np.delete(data_stopsign, range(min_size, len(data_stopsign)), axis = 0)
data_oven = np.delete(data_oven, range(min_size, len(data_oven)), axis = 0)
data_bicycle = np.delete(data_bicycle, range(min_size, len(data_bicycle)), axis = 0)
data_fan = np.delete(data_fan, range(min_size, len(data_fan)), axis = 0)
data_line = np.delete(data_line, range(min_size, len(data_line)), axis = 0)
data_key = np.delete(data_key, range(min_size, len(data_key)), axis = 0)
data_waterslide = np.delete(data_waterslide, range(min_size, len(data_waterslide)), axis = 0)
data_tshirt = np.delete(data_tshirt, range(min_size, len(data_tshirt)), axis = 0)
data_purse = np.delete(data_purse, range(min_size, len(data_purse)), axis = 0)
data_axe = np.delete(data_axe, range(min_size, len(data_axe)), axis = 0)
data_nose = np.delete(data_nose, range(min_size, len(data_nose)), axis = 0)
data_belt = np.delete(data_belt, range(min_size, len(data_belt)), axis = 0)
data_steak = np.delete(data_steak, range(min_size, len(data_steak)), axis = 0)
data_beach = np.delete(data_beach, range(min_size, len(data_beach)), axis = 0)
data_mushroom = np.delete(data_mushroom, range(min_size, len(data_mushroom)), axis = 0)
data_shovel = np.delete(data_shovel, range(min_size, len(data_shovel)), axis = 0)
data_spoon = np.delete(data_spoon, range(min_size, len(data_spoon)), axis = 0)
data_eiffeltower = np.delete(data_eiffeltower, range(min_size, len(data_eiffeltower)), axis = 0)
data_zigzag = np.delete(data_zigzag, range(min_size, len(data_zigzag)), axis = 0)

X_train = np.concatenate((data_cup, data_fish, data_fork, data_ladder, data_tree, data_airplane,
  data_donut, data_face, data_house, data_saw, data_tent, data_sun, data_moon, data_dog, 
  data_table, data_eye, data_pear, data_sword, data_telephone, data_tornado, data_pool,
  data_stopsign, data_oven, data_bicycle, data_fan, data_line, data_key, data_waterslide,
  data_tshirt, data_purse, data_axe, data_nose, data_belt, data_steak, data_beach, data_mushroom,
  data_shovel, data_spoon, data_eiffeltower, data_zigzag), axis = 0)

l = min_size
base = [0] * 5
y_cup = 0
y_fish = 1
y_fork = 2
y_ladder = 3
y_tree = 4
y_airplane = 5
y_donut = 6
y_face = 7
y_house = 8
y_saw = 9
y_tent = 10
y_sun = 11
y_moon = 12
y_dog = 13
y_table = 14
y_eye = 15
y_pear = 16
y_sword = 17
y_telephone = 18
y_tornado = 19
y_pool = 20
y_stopsign = 21
y_oven = 22
y_bicycle = 23
y_fan = 24
y_line = 25
y_key = 26
y_waterslide = 27
y_tshirt = 28
y_purse = 29
y_axe = 30
y_nose = 31
y_belt = 32
y_steak = 33
y_beach = 34
y_mushroom = 35
y_shovel = 36
y_spoon = 37
y_eiffeltower = 38
y_zigzag = 39

Y_train = [y_cup] * l + [y_fish] * l + [y_fork] * l + [y_ladder] * l + [y_tree] * l + [y_airplane]*l + [y_donut]*l + [y_face]*l + [y_house]*l + [y_saw]*l + [y_tent]*l + [y_sun]*l + [y_moon]*l + [y_dog]*l + [y_table]*l + [y_eye]*l + [y_pear]*l + [y_sword]*l + [y_telephone]*l + [y_tornado]*l + [y_pool]*l + [y_stopsign]*l + [y_oven]*l + [y_bicycle]*l + [y_fan]*l + [y_line]*l + [y_key]*l + [y_waterslide]*l + [y_tshirt]*l + [y_purse]*l + [y_axe]*l + [y_nose]*l + [y_belt]*l + [y_steak]*l + [y_beach]*l + [y_mushroom]*l + [y_shovel]*l + [y_spoon]*l + [y_eiffeltower]*l + [y_zigzag]*l
print(len(Y_train))
Y_train = np.asarray(Y_train)
Y_train = np_utils.to_categorical(Y_train)
print(Y_train.shape)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
print(X_train.shape)
for i in range(0, X_train.shape[0]):
  X_train[i] = np.divide(X_train[i], 255)

np.save("./X_train_data.npy", X_train)
np.save("./Y_train_data.npy", Y_train)
