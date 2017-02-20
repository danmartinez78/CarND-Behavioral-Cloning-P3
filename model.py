from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D, Lambda
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import pickle
from sklearn.model_selection import train_test_split
import cv2


# import data
with open('./data/data.p', mode='rb') as f:
    data = pickle.load(f)

# # split
X_train, X_test, y_train, y_test = train_test_split(data['features'], data['labels'], test_size=0.2)

# generators
gen_train = ImageDataGenerator()
gen_test = ImageDataGenerator()

cv2.imshow("test", X_train[99])
cv2.waitKey(2)
print(y_train[99])

# model
model = Sequential()
# input layer w/ cropping
model.add(Cropping2D(cropping=((40,20), (0,0)), input_shape=(160,320,3)))
# normalize
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
# 5x5 conv
model.add(Convolution2D(3, 5, 5, border_mode='valid'))
model.add(Activation('relu'))
# 5x5 conv
model.add(Convolution2D(3, 5, 5, border_mode='valid'))
model.add(Activation('relu'))
# 5x5 conv
model.add(Convolution2D(6, 5, 5, border_mode='valid'))
model.add(Activation('relu'))
# 3x3 conv
model.add(Convolution2D(8, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Dropout(0.6))
# 3x3 conv
model.add(Convolution2D(12, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
# flatten
model.add(Flatten())
# fcn1
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# fcn2
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# fcn3
model.add(Dense(10))
model.add(Activation('relu'))
# output
model.add(Dense(1))

# compile
model.compile('adam', 'mse')
checkpointer = ModelCheckpoint("./model.hdf5", verbose=1, save_best_only=True)
#history = model.fit(X, y, batch_size=128, nb_epoch=2, validation_split=0.2)
history = model.fit_generator(gen_train.flow(X_train, y_train, batch_size=32),
                samples_per_epoch=len(X_train),
                nb_epoch=1,
                validation_data=gen_test.flow(X_test, y_test, batch_size=32),
                nb_val_samples=len(X_train),
                callbacks=[checkpointer],
                verbose = 1
                )

### print the keys contained in the history object
print(history.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()