from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3),input_shape = (32, 32, 3), activation = 'relu'))

classifier.add(Conv2D(32, (3, 3),input_shape = (32, 32, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units = 512, activation = 'relu'))

classifier.add(Dense(units = 9, activation = 'softmax'))

classifier.summary()
""" 
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_3 (Conv2D)            (None, 30, 30, 32)        896
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 28, 28, 32)        9248
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 14, 14, 32)        0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 12, 12, 32)        9248
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 6, 6, 32)          0
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0
_________________________________________________________________
dense_2 (Dense)              (None, 512)               590336
_________________________________________________________________
dense_3 (Dense)              (None, 9)                 4617
=================================================================
Total params: 614,345
Trainable params: 614,345
Non-trainable params: 0
_________________________________________________________________"""

# Compiling the CNN
from tensorflow.keras.optimizers import RMSprop, SGD

classifier.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])

#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
# <tensorflow.python.keras.preprocessing.image.ImageDataGenerator object at 0x000001C7198D17C8>

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('17flowers\\train',
                                                 target_size = (32, 32),
                                                 batch_size = 16,
                                                 class_mode = 'categorical',
                                                 shuffle=True)
# Found 463 images belonging to 9 classes.

test_set = test_datagen.flow_from_directory('17flowers\\test',
                                            target_size = (32, 32),
                                            batch_size = 16,
                                            class_mode = 'categorical',
                                            shuffle=False)
# Found 127 images belonging to 9 classes.

nb_train_samples=463
nb_validation_samples=127
batch_size=32

# https://stackoverflow.com/questions/51591550/logits-and-labels-must-be-broadcastable-error-in-tensorflow-rnn
classifier.fit_generator(training_set,
                         steps_per_epoch = nb_train_samples // batch_size,
                         epochs = 20,
                         validation_data = test_set,
                         validation_steps = nb_validation_samples // batch_size)
""" 
Epoch 1/20
14/14 [==============================] - 1s 59ms/step - loss: 1.8244 - accuracy: 0.2857 - val_loss: 3.2733 - val_accuracy: 0.2708
Epoch 2/20
14/14 [==============================] - 1s 59ms/step - loss: 1.7100 - accuracy: 0.3304 - val_loss: 3.0721 - val_accuracy: 0.4375
Epoch 3/20
14/14 [==============================] - 1s 56ms/step - loss: 1.5725 - accuracy: 0.4036 - val_loss: 3.8662 - val_accuracy: 0.5208
Epoch 4/20
14/14 [==============================] - 1s 56ms/step - loss: 1.4782 - accuracy: 0.4821 - val_loss: 3.3428 - val_accuracy: 0.5208
Epoch 5/20
14/14 [==============================] - 1s 57ms/step - loss: 1.2991 - accuracy: 0.5625 - val_loss: 4.1388 - val_accuracy: 0.5417
Epoch 6/20
14/14 [==============================] - 1s 56ms/step - loss: 1.2777 - accuracy: 0.5446 - val_loss: 3.7384 - val_accuracy: 0.5000
Epoch 7/20
14/14 [==============================] - 1s 54ms/step - loss: 1.1609 - accuracy: 0.6009 - val_loss: 3.6140 - val_accuracy: 0.5417
Epoch 8/20
14/14 [==============================] - 1s 55ms/step - loss: 1.1251 - accuracy: 0.6054 - val_loss: 3.1463 - val_accuracy: 0.5417
Epoch 9/20
14/14 [==============================] - 1s 55ms/step - loss: 1.0709 - accuracy: 0.6099 - val_loss: 3.8672 - val_accuracy: 0.5833
Epoch 10/20
14/14 [==============================] - 1s 57ms/step - loss: 1.0338 - accuracy: 0.6295 - val_loss: 3.3396 - val_accuracy: 0.5417
Epoch 11/20
14/14 [==============================] - 1s 56ms/step - loss: 1.0261 - accuracy: 0.6682 - val_loss: 2.8374 - val_accuracy: 0.5208
Epoch 12/20
14/14 [==============================] - 1s 59ms/step - loss: 0.9372 - accuracy: 0.6830 - val_loss: 2.3818 - val_accuracy: 0.5417
Epoch 13/20
14/14 [==============================] - 1s 63ms/step - loss: 0.9587 - accuracy: 0.6682 - val_loss: 2.4026 - val_accuracy: 0.5833
Epoch 14/20
14/14 [==============================] - 1s 63ms/step - loss: 0.9150 - accuracy: 0.6875 - val_loss: 3.1476 - val_accuracy: 0.5625
Epoch 15/20
14/14 [==============================] - 1s 55ms/step - loss: 0.8236 - accuracy: 0.7130 - val_loss: 3.7638 - val_accuracy: 0.5833
Epoch 16/20
14/14 [==============================] - 1s 62ms/step - loss: 0.8473 - accuracy: 0.6830 - val_loss: 3.0478 - val_accuracy: 0.5000
Epoch 17/20
14/14 [==============================] - 1s 62ms/step - loss: 0.7217 - accuracy: 0.7489 - val_loss: 3.0338 - val_accuracy: 0.5417
Epoch 18/20
14/14 [==============================] - 1s 59ms/step - loss: 0.8553 - accuracy: 0.7175 - val_loss: 2.5646 - val_accuracy: 0.5625
Epoch 19/20
14/14 [==============================] - 1s 57ms/step - loss: 0.6399 - accuracy: 0.7803 - val_loss: 3.4857 - val_accuracy: 0.2292
Epoch 20/20
14/14 [==============================] - 1s 58ms/step - loss: 0.7311 - accuracy: 0.7411 - val_loss: 3.2573 - val_accuracy: 0.4792
<tensorflow.python.keras.callbacks.History object at 0x000001C721053808> """


#result: 
#loss: 0.1170 - acc: 0.9617 - val_loss: 1.2301 - val_acc: 0.7568

