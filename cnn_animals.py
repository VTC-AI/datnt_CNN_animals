# Import keras lib & packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Initial CNN
classifier = Sequential()

# Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Add 2nd Convolution layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flaten
classifier.add(Flatten())

# Full connection
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(2, activation = 'softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit CNN2images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
  rescale=1./255,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
  'dataset/training_set',
  target_size=(64, 64),
  batch_size=16,
  class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
  'dataset/test_set',
  target_size=(64, 64),
  batch_size=16,
  class_mode='categorical'
)

classifier.fit_generator(
  training_set,
  steps_per_epoch=(4000 / 16),
  epochs=50,
  validation_data=test_set,
  validation_steps=(2000 / 16)
)

# Make new predictions
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices

# if result[0][0] == 1:
#   prediction = 'dog'
# elif result[0][1] == 1:
#   prediction = 'cat'
# else:
#   prediction = 'none'

print(result)