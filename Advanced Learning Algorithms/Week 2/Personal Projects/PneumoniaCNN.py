# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os

##########################################################
# Title: PneumoniaCNN
# Model Type: Convolutional Neural Network for Binary Classification
# Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

# Goal: Analyse over 5000 xrays of lungs to determine whether they are of a person with pneumonia or of a healthy person.

##########################################################

data_dir = "C:/Users/Hami/Downloads/archive(13)"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255, rotation_range=30,zoom_range=0.2, horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, fill_mode='nearest')

train_generator = datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

val_generator = datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)

def train_cnn():
    # # CNN
    model = models.Sequential([
        tf.keras.Input(shape=(224, 224, 3)),
        
        layers.Conv2D(32, (3,3), activation='relu'),  
        layers.MaxPooling2D((2,2)), 
        layers.Conv2D(64, (3,3), activation='relu'), 
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(256, (3,3), activation='relu'), 
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(512, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
    
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=[lr_scheduler]
    )

    loss, acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    best_acc = 0.92
    if acc > best_acc:
        model.save('pneumonia_model.keras')

def predict():
    model = tf.keras.models.load_model('pneumonia_model.keras')
    
    pneumonia_folder = os.path.join(test_dir, "opacity")
    normal_folder = os.path.join(test_dir, "normal")

    pneumonia_files = os.listdir(pneumonia_folder)
    normal_files = os.listdir(normal_folder)

    pneumonia_img_path = os.path.join(pneumonia_folder, random.choice(pneumonia_files))
    normal_img_path = os.path.join(normal_folder, random.choice(normal_files))

    def load_and_preprocess(img_path):
        img = image.load_img(img_path, target_size=(224, 224))  
        img_array = image.img_to_array(img) / 255.0 
        img_array = np.expand_dims(img_array, axis=0)  
        return img, img_array

    pneumonia_img, pneumonia_array = load_and_preprocess(pneumonia_img_path)
    normal_img, normal_array = load_and_preprocess(normal_img_path)

    pneumonia_pred = model.predict(pneumonia_array)[0][0]
    normal_pred = model.predict(normal_array)[0][0]
    
    pneumonia_label = "Pneumonia" if pneumonia_pred > 0.5 else "Normal"
    normal_label = "Pneumonia" if normal_pred > 0.5 else "Normal"

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pneumonia_img)
    axes[0].set_title(f"Pred: {pneumonia_label} ({pneumonia_pred:.2f})")
    axes[0].axis("off")

    axes[1].imshow(normal_img)
    axes[1].set_title(f"Pred: {normal_label} ({normal_pred:.2f})")
    axes[1].axis("off")

    plt.show()
    
     
train_cnn() # Uncomment to train 
predict()
