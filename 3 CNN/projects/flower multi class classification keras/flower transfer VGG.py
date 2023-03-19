from tensorflow.keras.applications import VGG16

# VGG16 was designed to work on 224 x 224 pixel input images sizes
img_rows = 64
img_cols = 64 

#Loads the VGG16 model 
vgg16 = VGG16(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))

# Let's print our layers 
for (i,layer) in enumerate(vgg16.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)
""" 
0 InputLayer True
1 Conv2D True
2 Conv2D True
3 MaxPooling2D True
4 Conv2D True
5 Conv2D True
6 MaxPooling2D True
7 Conv2D True
8 Conv2D True
9 Conv2D True
10 MaxPooling2D True
11 Conv2D True
12 Conv2D True
13 Conv2D True
14 MaxPooling2D True
15 Conv2D True
16 Conv2D True
17 Conv2D True
18 MaxPooling2D True """

# Here we freeze the last 4 layers 
# Layers are set to trainable as True by default
for layer in vgg16.layers:
    layer.trainable = False
    
# Let's print our layers 
for (i,layer) in enumerate(vgg16.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)
""" 
0 InputLayer False
1 Conv2D False
2 Conv2D False
3 MaxPooling2D False
4 Conv2D False
5 Conv2D False
6 MaxPooling2D False
7 Conv2D False
8 Conv2D False
9 Conv2D False
10 MaxPooling2D False
11 Conv2D False
12 Conv2D False
13 Conv2D False
14 MaxPooling2D False
15 Conv2D False
16 Conv2D False
17 Conv2D False
18 MaxPooling2D False """

def addTopModel(bottom_model, num_classes, D=256):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""
    top_model = bottom_model.output
    top_model = Flatten(name = "flatten")(top_model)
    top_model = Dense(D, activation = "relu")(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(num_classes, activation = "softmax")(top_model)
    return top_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

num_classes = 9
FC_Head = addTopModel(vgg16, num_classes)
model = Model(inputs=vgg16.input, outputs=FC_Head)
print(model.summary())
""" 
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 64, 64, 3)]       0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 64, 64, 64)        1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 64, 64, 64)        36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 32, 32, 64)        0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 32, 32, 128)       73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 32, 32, 128)       147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 16, 16, 128)       0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 16, 16, 256)       295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 16, 16, 256)       590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 16, 16, 256)       590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 8, 8, 256)         0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 8, 8, 512)         1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 8, 8, 512)         2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 8, 8, 512)         2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 4, 4, 512)         0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 4, 4, 512)         2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 4, 4, 512)         2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 4, 4, 512)         2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 2, 2, 512)         0
_________________________________________________________________
flatten (Flatten)            (None, 2048)              0
_________________________________________________________________
dense_4 (Dense)              (None, 256)               524544
_________________________________________________________________
dropout (Dropout)            (None, 256)               0
_________________________________________________________________
dense_5 (Dense)              (None, 9)                 2313
=================================================================
Total params: 15,241,545
Trainable params: 526,857
Non-trainable params: 14,714,688
_________________________________________________________________ """

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

train_data_dir = '17flowers/train'
validation_data_dir = '17flowers/test'

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# Change the batchsize according to your system RAM
train_batchsize = 32
val_batchsize = 10
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=train_batchsize,
        class_mode='categorical')
# Found 463 images belonging to 9 classes.

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)
# Found 127 images belonging to 9 classes.

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
                   
checkpoint = ModelCheckpoint("flowers_vgg_64.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 5,
                          verbose = 1,
                          restore_best_weights = True)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.2,
                              patience = 3,
                              verbose = 1,
                              min_delta = 0.00001)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint, reduce_lr]

# Note we use a very small learning rate 
model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.0001),
              metrics = ['accuracy'])

nb_train_samples = 463
nb_validation_samples = 127
epochs = 25
batch_size = 32

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)

""" 
Epoch 1/25
14/14 [==============================] - ETA: 0s - loss: 2.3318 - accuracy: 0.1230
Epoch 00001: val_loss improved from inf to 2.24400, saving model to flowers_vgg_64.h5
14/14 [==============================] - 3s 244ms/step - loss: 2.3318 - accuracy: 0.1230 - val_loss: 2.2440 - val_accuracy: 0.3000
Epoch 2/25
14/14 [==============================] - ETA: 0s - loss: 2.0532 - accuracy: 0.2274
Epoch 00002: val_loss did not improve from 2.24400
14/14 [==============================] - 3s 243ms/step - loss: 2.0532 - accuracy: 0.2274 - val_loss: 2.2968 - val_accuracy: 0.3333
Epoch 3/25
14/14 [==============================] - ETA: 0s - loss: 1.8761 - accuracy: 0.3063
Epoch 00003: val_loss did not improve from 2.24400
14/14 [==============================] - 3s 247ms/step - loss: 1.8761 - accuracy: 0.3063 - val_loss: 2.3720 - val_accuracy: 0.3667
Epoch 4/25
14/14 [==============================] - ETA: 0s - loss: 1.7502 - accuracy: 0.3480
Epoch 00004: val_loss did not improve from 2.24400

Epoch 00004: ReduceLROnPlateau reducing learning rate to 1.9999999494757503e-05.
14/14 [==============================] - 3s 217ms/step - loss: 1.7502 - accuracy: 0.3480 - val_loss: 2.3928 - val_accuracy: 0.3667
Epoch 5/25
14/14 [==============================] - ETA: 0s - loss: 1.6386 - accuracy: 0.4223
Epoch 00005: val_loss did not improve from 2.24400
14/14 [==============================] - 3s 226ms/step - loss: 1.6386 - accuracy: 0.4223 - val_loss: 2.3540 - val_accuracy: 0.4333
Epoch 6/25
14/14 [==============================] - ETA: 0s - loss: 1.6551 - accuracy: 0.3991Restoring model weights from the end of the best epoch.

Epoch 00006: val_loss did not improve from 2.24400
14/14 [==============================] - 3s 226ms/step - loss: 1.6551 - accuracy: 0.3991 - val_loss: 2.3567 - val_accuracy: 0.4333
Epoch 00006: early stopping """

model.save("flowers_vgg.h5")
    
    
