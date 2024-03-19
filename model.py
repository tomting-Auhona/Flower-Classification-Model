# Importing necessary libraries
import numpy as np # For numerical computations
import pandas as pd # For data manipulation
import matplotlib.pyplot as plt # For data visualization
import seaborn as sns # For enhanced data visualization
import keras # For building neural networks
from keras.models import Sequential, Model # For creating sequential and functional API models
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, MaxPooling2D # Different layers for the neural network
from keras.preprocessing.image import ImageDataGenerator # For image data preprocessing
from sklearn.model_selection import train_test_split # For splitting data into training and testing sets
from sklearn.metrics import classification_report, confusion_matrix # For evaluating model performance
from keras.applications import VGG19 # Pre-trained VGG19 model
import cv2 # For image processing tasks
import os # For interacting with the operating system
import random # For generating random numbers
import tensorflow as tf # TensorFlow library for machine learning tasks

# Defining the labels for different flower classes
labels = ['dandelion', 'daisy', 'tulip', 'sunflower', 'rose']
img_size = 224 # Desired image size for input to the model

# Function to retrieve and preprocess image data
def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR) # Reading image
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Resizing image to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return data

# Loading and preprocessing image data
data = get_data("D:/petals/flowers")

x = [] # List to store features (images)
y = [] # List to store labels

for feature, label in data:
    x.append(feature)
    y.append(label)

# Normalize the data (scaling pixel values to range [0, 1])
x = np.array(x) / 255

# Reshaping the data from 1-D to 3-D as required input by CNNs
x = x.reshape(-1, img_size, img_size, 3)
y = np.array(y)

# Converting labels to one-hot encoded format
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(y)

# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

# Cleaning up memory by deleting unnecessary variables
del x, y, data

# Detecting hardware and configuring distribution strategy for training
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is set (Kaggle).
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in TensorFlow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

# Building the model using transfer learning (VGG19)
with strategy.scope():
    pre_trained_model = VGG19(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

    # Freezing layers of the pre-trained model
    for layer in pre_trained_model.layers[:19]:
        layer.trainable = False

    # Creating the final classification layers
    model = Sequential([
        pre_trained_model,
        MaxPool2D((2, 2), strides=2),
        Flatten(),
        Dense(5, activation='softmax')])

    # Compiling the model
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Displaying model summary
model.summary()

# Callback for reducing learning rate if validation accuracy plateaus
from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3, min_lr=0.000001)

# Training the model
history = model.fit(x_train, y_train, batch_size=64, epochs=12, validation_data=(x_test, y_test), callbacks=[learning_rate_reduction])

# Evaluating model performance
print("Loss of the model is - ", model.evaluate(x_test, y_test)[0])
print("Accuracy of the model is - ", model.evaluate(x_test, y_test)[1] * 100, "%")

# Saving the trained model
saved_model_path = 'flower_v2.h5'
model.save(saved_model_path)
