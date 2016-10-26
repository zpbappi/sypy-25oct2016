import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense

# load the data
# image size 28x28 px, output 1 int
# 60,000 training data, 10,000 test data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape the data
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize the pixel values
X_train /= 255
X_test /= 255

# reshape the output
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


# let there be model
model = Sequential()
model.add(Dense(10, input_shape=(784, ), activation='softmax'))


# define criteria
model.compile(
	loss='categorical_crossentropy',
	optimizer='sgd',
	metrics=['accuracy'])

# train
batch_size = 128
epochs = 10
model.fit(
	X_train, Y_train, 
	batch_size=batch_size, nb_epoch=epochs,
	validation_data=(X_test, Y_test),
	verbose=1)

# evaludate
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test accuracy: ', score[1])


# visualizing weights
print("Enter numbers (0-9) to visualize weights:")
import matplotlib.pyplot as plt
plt.gray()
weights = model.layers[0].get_weights()[0]

def normalize_weight(w):
	max_val = abs(w.max())
	min_val = abs(w.min())

	max_val = min_val if min_val > max_val else max_val
	min_val = -max_val

	img = w.reshape(28,28)
	return (img - min_val) / (max_val - min_val)


import sys
while True:
	digit = int(sys.stdin.readline())
	if digit < 0 or digit > 9:
		break
	w = normalize_weight(weights[:,digit])
	#plt.matshow(w, fignum=False)
	plt.imshow(w)
	plt.show()