import os
import numpy as np

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import img_to_array, load_img
from keras import backend as K

DATASET_BASE_PATH = 'd:/datasets/dogcat'

base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)

x = Dense(1, activation='sigmoid')(x)

model = Model(input=base_model.input, output=x)

# set each layer of InceptionV3 as frozen, 
# so that we don't end up training InceptionV3 model (!!!)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# utility method
def preprocess_input(x, dim_ordering='default'):
	# source: https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x

# utility method to generate training data
def get_dataset():
	folder = DATASET_BASE_PATH + '/cat'
	y = np.array([0])
	for name in os.listdir(folder):
		fpath = os.path.join(folder, name)
		x = img_to_array(load_img(fpath, target_size=(299, 299)))
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		yield (x,y)

	folder = DATASET_BASE_PATH + '/dog'
	y = np.array([1])
	for name in os.listdir(folder):
		fpath = os.path.join(folder, name)
		x = img_to_array(load_img(fpath, target_size=(299, 299)))
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		yield (x,y)

# utility method to generate test data
def get_test_data():
	folder = DATASET_BASE_PATH + '/test'
	for name in os.listdir(folder):
		fpath = os.path.join(folder, name)
		x = img_to_array(load_img(fpath, target_size=(299, 299)))
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		yield (x, name)

# reshape training data
data = list(get_dataset())
img_shape = data[0][0].shape
total_sample = len(data)
X = np.zeros((total_sample, img_shape[1], img_shape[2], img_shape[3]))
Y = np.zeros((total_sample, 1))
index = 0
for item in data:
	X[index, :, :, :] = item[0]
	Y[index, :] = item[1]
	index += 1

# train our model
model.fit(X, Y, batch_size=20, nb_epoch=3, verbose=1)

# note, i have tried with model.fit_generator(...). however, it was throwing 
# threading exception in my mahcine. try it in your.
# comment the whole reshape training data section and uncomment the following:
#
# model.fit_generator(get_dataset(), samples_per_epoch=20, nb_epoch=3, verbose=1)

# save the learned weights so that we can load them for later use (not shown in the code)
model.save_weights('inception_dog_cat.h5')


# prediction time
data = list(get_test_data())
img_shape = data[0][0].shape
total_sample = len(data)
X = np.zeros((total_sample, img_shape[1], img_shape[2], img_shape[3]))
filenames = []
index = 0
for item in data:
	X[index, :, :, :] = item[0]
	filenames.append(item[1])
	index += 1

Y = model.predict(X)

for pred in zip(filenames, Y.flatten()):
	print("file:", pred[0], "prediction:", 'cat' if pred[1]<0.5 else 'dog')
