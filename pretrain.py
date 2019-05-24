from __future__ import absolute_import, division, print_function
import keras
import os
import tensorflow as tf
import PIL
import numpy as np
import sys
import time
from keras import callbacks
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import random
import cv2
import pickle
print("Tensorflow version is ", tf.__version__)


# set the matplotlib backend so figures can be saved in the background
# matplotlib.use("Agg")

# Initialize the data and labels
start = time.time()
print("[INFO] loading images...")
data = []
labels = []
image_size = 160
args = {"dataset": "data/train2", "model": "models/net2.model", "label-bin": "models/net2_lb.pickle",
        "plot": "models/net2_plot.png"}

num_classes = 2
# grab the image paths and randomly shuffle them
image_paths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(image_paths)

# loop over the input image
for image_path in image_paths:
    # load the image, resize it to 160x160 pixels (the required input
    # spatial dimensions of MobileNet) and store the image in the
    # data list
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size))
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = image_path.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(train_x, test_x, train_y, test_y) = train_test_split(data,
                                                      labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode='nearest')


# zip_file = keras.utils.get_file(origin="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
#                                fname="cats_and_dogs_filtered.zip", extract=True)
# base_dir, _ = os.path.splitext(zip_file)
# print(base_dir)

# train_dir = os.path.join(base_dir, "train")
# validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat pictures
# train_cats_dir = os.path.join(train_dir, 'cats')
# print('Total training cat images: ', len(os.listdir(train_cats_dir)))

# Directory with our training dog pictures
# train_dogs_dir = os.path.join(train_dir, 'dogs')
# print('Total training dog images: ', len(os.listdir(train_dogs_dir)))

# Directory with our validation cat pictures
# validation_cats_dir = os.path.join(validation_dir, 'cats')
# print('Total validation cat images: ', len(os.listdir(validation_cats_dir)))

# Directory with our validation dog pictures
# validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# print('Total validation dog images: ', len(os.listdir(validation_dogs_dir)))

INIT_LR = 0.0001  # 0.01
epochs = 30
batch_size = 32

# Rescale all images by 1./255 and apply image augmentation
# train_datagen = keras.preprocessing.image.ImageDataGenerator(
#    rotation_range=30, width_shift_range=0.1, shear_range=0.2,
#    zoom_range=0.2, horizontal_flip=True, fill_mode="nearest", rescale=1./255)
# validation_datagen = keras.preprocessing.image.ImageDataGenerator(
#    rotation_range=30, width_shift_range=0.1, shear_range=0.2,
#    zoom_range=0.2, horizontal_flip=True, fill_mode="nearest", rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
# train_generator = train_datagen.flow_from_directory(
#    train_dir,
#    target_size=(image_size, image_size),
#    batch_size=batch_size,
#    class_mode='categorical')

# Flow validation images in batches of 20 using test_datagen generator
# validation_generator = validation_datagen.flow_from_directory(
#    validation_dir,
#    target_size=(image_size, image_size),
#    batch_size=batch_size,
#    class_mode='categorical')

IMG_SHAPE = (image_size, image_size, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                            include_top=False,
                                            weights='imagenet')
base_model.trainable = False

# Let's take a look at the base model architecture
base_model.summary()

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=keras.optimizers.RMSprop(lr=INIT_LR), # , decay=INIT_LR / epochs),  # 0.0001
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

print(len(model.trainable_weights))

# epochs = 10
# steps_per_epoch = train_generator.n // batch_size
# validation_steps =validation_generator.n // batch_size

"""
Tensorboard log
"""
log_dir = './tf-log'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb]

history = model.fit_generator(aug.flow(train_x, train_y, batch_size=batch_size),  # train_generator
                              steps_per_epoch=len(train_x) // batch_size,  # steps_per_epoch,
                              epochs=epochs,
                              workers=0,
                              callbacks=cbks,
                              validation_data=(test_x, test_y),  # validation_generator,
                              validation_steps=len(test_x) // batch_size)  # validation_steps)
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, max(plt.ylim())])
plt.title('Training and Validation Loss')

plt.show()

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the 'fine_tune_at' layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model.summary()

print(len(model.trainable_weights))

history_fine = model.fit_generator(aug.flow(train_x, train_y, batch_size=batch_size),  # train_generator,
                                   steps_per_epoch=len(train_x) // batch_size,  # steps_per_epoch,
                                   epochs=epochs,
                                   workers=0,
                                   callbacks=cbks,
                                   validation_data=(test_x, test_y),  # validation_generator,
                                   validation_steps=len(test_x) // batch_size)  # validation_steps)

acc += history_fine.history['acc']
val_acc += history_fine.history['val_acc']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

dur_epochs = range(len(acc))

plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)
plt.plot(dur_epochs, acc, label='Training Accuracy')
plt.plot(dur_epochs, val_acc, label='Validation Accuracy')
plt.ylim([0.9, 1])
plt.plot([epochs-1, epochs-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(dur_epochs, loss, label='Training Loss')
plt.plot(dur_epochs, val_loss, label='Validation Loss')
plt.ylim([0, 0.2])
plt.plot([epochs-1, epochs-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
# saved_model_path = tf.contrib.saved_model.save_keras_model(model, './saved_models')
target_dir = './models'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
model.save(args["model"])  # './models/model.h5')
model.save_weights('./models/weights.h5')

f = open(args["label-bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
# Calculate execution time
end = time.time()
duration = end - start

if duration < 60:
    print('Execution Time: ', duration, "seconds")
elif 3600 > duration > 60:
    print('Execution Time: ', duration / 60, 'minutes')
else:
    duration = duration / 3600
    print('Execution Time: ', duration, 'hours')
