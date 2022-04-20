import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

X_train = np.load('X_train_split.npy')
y_train = np.load('y_train_split.npy')

X_test = np.load('X_test_split.npy')
y_test = np.load('y_test_split.npy')

print(str(X_train.shape) + " " +  str(X_test.shape) + " " + str(y_train.shape) + " " + str(y_test.shape))

class_weights = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
d_class_weights = dict(enumerate(class_weights))

filepath = "saved-model-final-final-2-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max')

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape = (1024, 1024, 3)))
model.add(Activation(LeakyReLU(alpha=0.001)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Convolution2D(32, 3, 3))
model.add(Activation(LeakyReLU(alpha=0.001)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Convolution2D(64, 3, 3))
model.add(Activation(LeakyReLU(alpha=0.001)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.0000001)
model.add(Flatten())
model.add(Dense(64, activation=LeakyReLU(alpha=0.001)))
model.add(Dropout(0.8))
model.add(Dense(4, activation='softmax'))


model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=32, epochs=30, class_weight=d_class_weights, callbacks=[checkpoint, reduce_lr], validation_data=(X_test, y_test))
