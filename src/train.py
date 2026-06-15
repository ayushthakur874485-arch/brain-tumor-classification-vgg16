import os
import numpy as np
import random
from PIL import Image, ImageEnhance
from sklearn.utils import shuffle

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16

# Dataset paths
train_dir = "dataset/Training"
test_dir = "dataset/Testing"

IMAGE_SIZE = 128
BATCH_SIZE = 20
EPOCHS = 5

# Load paths
train_paths = []
train_labels = []

for label in os.listdir(train_dir):
    label_path = os.path.join(train_dir, label)

    for image in os.listdir(label_path):
        train_paths.append(os.path.join(label_path, image))
        train_labels.append(label)

train_paths, train_labels = shuffle(train_paths, train_labels)

# Label encoding
classes = ['glioma', 'notumor', 'meningioma', 'pituitary']

def encode_label(labels):
    return np.array([classes.index(label) for label in labels])

# Augmentation
def augment_image(image):
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    image = np.array(image) / 255.0
    return image

# Open images
def open_images(paths):
    images = []

    for path in paths:
        img = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img = augment_image(img)
        images.append(img)

    return np.array(images)

# Generator
def datagen(paths, labels, batch_size=12):

    while True:

        for i in range(0, len(paths), batch_size):

            batch_paths = paths[i:i+batch_size]

            batch_images = open_images(batch_paths)

            batch_labels = encode_label(labels[i:i+batch_size])

            yield batch_images, batch_labels

# VGG16
base_model = VGG16(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

for layer in base_model.layers:
    layer.trainable = False

base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True

# Model
model = Sequential([
    Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    base_model,
    Flatten(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

steps = len(train_paths) // BATCH_SIZE

history = model.fit(
    datagen(train_paths, train_labels, BATCH_SIZE),
    epochs=EPOCHS,
    steps_per_epoch=steps
)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/model.h5")

print("Model saved successfully.")# Move your training code here
