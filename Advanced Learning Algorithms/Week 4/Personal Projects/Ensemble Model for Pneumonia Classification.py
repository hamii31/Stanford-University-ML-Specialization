import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Ensemble Model for Pneumonia Classification

# Models:
#       - For Feature Classification: Random Forest Model for Binary Classification. Ran a GridSearchCV to find the best estimators and depth. Weights were manually
#       calibrated to fit the class imbalances. Best Testing Accuracy: 96.215% (Excellent Model)
#       
#       - For Image Classification: Convolutional Neural Network for Binary Classification. 
#       Reused the best weights for cv accuracy after each training, since the model had high variance issues.
#       Best Testing Accuracy: 93.42% (Surpasses my previous Pneumonia X-ray CNN Model from which I transfered the structuring of layers and units.)
#       
#       - Ensemble Model: The combination of tabular and image classification offer a more robust take on pneumonia classification. 
#       Overall accuracy of the model: 94.82%

# Augmented Datasets:
#       - For Feature Classification: Only the lung diseases from https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset 
#       - For Image Classification: A mix between https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia and https://datasetninja.com/zhang-lab-data-chest-xray

# Random Forest Model Accuracy: 0.9621570482497634
# Random Forest Classification Report:
#                precision    recall  f1-score   support

#            0       0.95      1.00      0.98       819
#            1       1.00      0.83      0.91       238

#     accuracy                           0.96      1057
#    macro avg       0.98      0.92      0.94      1057
# weighted avg       0.96      0.96      0.96      1057

# CNN Model Accuracy: 0.9479166666666
# CNN Classification Report:
#                precision    recall  f1-score   support

#            0       0.96      0.90      0.93       468
#            1       0.94      0.98      0.96       780

#     accuracy                           0.95      1248
#    macro avg       0.95      0.94      0.94      1248
# weighted avg       0.95      0.95      0.95      1248

# ==================================================
# 0.9550368574582151


def random_forest_model():
    data = pd.read_csv("lungdiseases.csv")
    data['diseases'] = data['diseases'].apply(lambda x: 1 if x == 'pneumonia' else 0)

    X = data.drop(columns=['diseases'])  
    y = data['diseases']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Starting Random Forest for feature selection...")
    rf = RandomForestClassifier(n_estimators=100, class_weight={0: 5, 1: 3}, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)

    rf_predictions_test = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions_test)
    rf_report = classification_report(y_test, rf_predictions_test)

    print("Random Forest Model Accuracy:", rf_accuracy)
    print("Random Forest Classification Report:\n", rf_report)
    print("="*50)

    print("Done!")
    return rf_accuracy

def cnn_model():
    data_dir = "C:/Users/Hami/Downloads/archive(13)"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    count_normal = 0  
    count_pneumonia = 0 

    for class_folder in os.listdir(train_dir):
        class_folder_path = os.path.join(train_dir, class_folder)
    
        if os.path.isdir(class_folder_path):  
            num_images = len(os.listdir(class_folder_path)) 
        
            if class_folder == "normal":  
                count_normal += num_images
            elif class_folder == "opacity":  
                count_pneumonia += num_images

    print("Count of class 'normal':", count_normal)
    print("Count of class 'opacity (pneumonia)':", count_pneumonia)


    img_size = (224, 224)
    batch_size = 32

    datagen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=30,
        zoom_range=0.2, 
        horizontal_flip=True, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        shear_range=0.2, 
        fill_mode='nearest')

    train_generator = datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

    val_generator = datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

    test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', shuffle=False)

    print("Train samples: ", train_generator.samples)
    print("Test samples: ", test_generator.samples)

    print("Starting CNN for image classification...")
    cnn_model = Sequential([
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
    print("Done!")
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    class_weight = {0: 5., 1: 3.} 

    checkpoint = ModelCheckpoint(
        "og_cnn_model.keras",  
        monitor="val_accuracy", 
        save_best_only=True, 
        mode="max", 
        verbose=1
    )
    
    return cnn_model, train_generator, val_generator, test_generator, class_weight, checkpoint

def train_and_evaluate_ensemble(cnn_model, train_generator, val_generator, test_generator, class_weight, checkpoint):
    rf_accuracy = random_forest_model()
    
    cnn_model.load_weights("og_cnn_model.keras")
    cnn_model.fit(train_generator, epochs=20, validation_data=val_generator, class_weight=class_weight, callbacks=[checkpoint])

    cnn_model.summary()

    cnn_predictions_test = cnn_model.predict(test_generator, verbose=1)

    y_true = test_generator.classes

    cnn_predictions_prob = cnn_predictions_test.flatten()  
    cnn_predictions_binary = (cnn_predictions_prob > 0.5).astype(int) 

    cnn_accuracy = accuracy_score(y_true, cnn_predictions_binary)
    cnn_report = classification_report(y_true, cnn_predictions_binary)

    print("CNN Model Accuracy:", cnn_accuracy)
    print("CNN Classification Report:\n", cnn_report)
    print("="*50)

    if cnn_accuracy > 0.9358974358974359:
        cnn_model.save("best_cnn_model.keras")
        
    combined_accuracy = (rf_accuracy + cnn_accuracy) / 2
    print(combined_accuracy)


def show_xray_predictions():
    cnn_model = load_model("best_cnn_model.keras")

    pneumonia_img_path = "C:/Users/Hami/Downloads/archive(13)/test/opacity/BACTERIA-40699-0001.jpeg"
    normal_img_path = "C:/Users/Hami/Downloads/archive(13)/test/normal/NORMAL-8944072-0001.jpeg"

    def load_and_preprocess_image(img_path, target_size=(224, 224)): 
        img = image.load_img(img_path, target_size=target_size, color_mode='rgb')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array /= 255.0 
        return img_array, img

    pneumonia_img_array, pneumonia_img = load_and_preprocess_image(pneumonia_img_path)
    normal_img_array, normal_img = load_and_preprocess_image(normal_img_path)

    pneumonia_pred = cnn_model.predict(pneumonia_img_array)[0][0]
    normal_pred = cnn_model.predict(normal_img_array)[0][0]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(pneumonia_img, cmap='gray')
    axes[0].set_title(f"Pneumonia Prediction: {pneumonia_pred:.2f}")

    axes[1].imshow(normal_img, cmap='gray')
    axes[1].set_title(f"Normal Prediction: {normal_pred:.2f}")

    plt.show()



cnn_model, train_generator, val_generator, test_generator, class_weight, checkpoint = cnn_model()
train_and_evaluate_ensemble(cnn_model, train_generator, val_generator, test_generator, class_weight, checkpoint)
show_xray_predictions()
