from keras.models import load_model
mymodel = load_model('my_model.h5')

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', 
                            target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = mymodel.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
    print(" The test image is")
    print(prediction)
else:
    prediction = 'cat'
    print(" The test image is")
    print(prediction)