from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D, Lambda, MaxPooling2D
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import pickle
from sklearn.model_selection import train_test_split

# import data
with open('./data/data.p', mode='rb') as f:
    data = pickle.load(f)

# # split
X_train, X_test, y_train, y_test = train_test_split(data['features'], data['labels'], test_size=0.2)

# generators
datagen = ImageDataGenerator()

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)

testgen = ImageDataGenerator()

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
testgen.fit(X_test)

# model
model = Sequential()
# input layer w/ cropping
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# normalize to -0.5:0.5
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
# 5x5 conv
model.add(Convolution2D(24, 5, 5, border_mode='valid'))
model.add(Activation('relu'))
# 5x5 conv
model.add(Convolution2D(36, 5, 5, border_mode='valid'))
model.add(Activation('relu'))
# 5x5 conv
model.add(Convolution2D(48, 5, 5, border_mode='valid'))
model.add(Activation('relu'))
# 3x3 conv
model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
# 3x3 conv
model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
# flatten
model.add(Flatten())
# fcn1
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.50))
# fcn2
model.add(Dense(50))
model.add(Activation('relu'))
# fcn3
model.add(Dense(10))
model.add(Activation('relu'))
# output
model.add(Dense(1))

# compile
model.compile('adam', 'mse')
check = ModelCheckpoint("./model.h5", verbose=1, save_best_only=True)
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),
                samples_per_epoch=len(X_train),
                nb_epoch=6,
                validation_data=testgen.flow(X_test, y_test, batch_size=128),
                nb_val_samples=len(X_train),
                callbacks=[check],
                verbose=1
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

print("DONE!")