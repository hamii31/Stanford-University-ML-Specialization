import os
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Ensemble Model for Pneumonia Classification


# Random Forest Model Accuracy: 0.9583727530747398
# Random Forest Classification Report:
#                precision    recall  f1-score   support

#            0       0.96      0.99      0.97       819
#            1       0.94      0.87      0.90       238

#     accuracy                           0.96      1057
#    macro avg       0.95      0.93      0.94      1057
# weighted avg       0.96      0.96      0.96      1057

# CNN Model Accuracy: 0.9190705128205128
# CNN Classification Report:
#                precision    recall  f1-score   support

#            0       0.94      0.84      0.89       468
#            1       0.91      0.97      0.94       780

#     accuracy                           0.92      1248
#    macro avg       0.92      0.90      0.91      1248
# weighted avg       0.92      0.92      0.92      1248

# ==================================================
# Overal Accuracy of the Ensemble Model: 0.9387216329476262


data = pd.read_csv("lungdiseases.csv")
data['diseases'] = data['diseases'].apply(lambda x: 1 if x == 'pneumonia' else 0)

X = data.drop(columns=['diseases'])  
y = data['diseases']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10, 20, 40],
    'learning_rate': [0.01, 0.1, 0.2],
    'scale_pos_weight': [1, 2, 4, 8],  # Test different weights
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, scoring='f1', cv=3)
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)
print("Starting Random Forest for feature selection...")
rf = RandomForestClassifier(n_estimators=100, class_weight={0: 5, 1: 3}, random_state=42)
rf.fit(X_train, y_train)

rf_predictions_test = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions_test)
rf_report = classification_report(y_test, rf_predictions_test)

print("Random Forest Model Accuracy:", rf_accuracy)
print("Random Forest Classification Report:\n", rf_report)
print("="*50)

print("Done!")

data_dir = "C:/Users/Hami/Downloads/archive(13)"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

count_normal = 0  
count_pneumonia = 0 

# Counts the instances of the classes
for class_folder in os.listdir(train_dir):
    class_folder_path = os.path.join(train_dir, class_folder)
    
    if os.path.isdir(class_folder_path):  
        num_images = len(os.listdir(class_folder_path)) 
        
        if class_folder == "normal":  
            count_normal += num_images
        elif class_folder == "opacity":  
            count_pneumonia += num_images

# Print the counts
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

# cnn_model = load_model("best_cnn_model.keras")
cnn_model.load_weights("og_cnn_model.keras")
cnn_model.fit(train_generator, epochs=20, validation_data=val_generator, class_weight=class_weight, callbacks=[checkpoint])

cnn_predictions_train = cnn_model.predict(train_generator, verbose=1)
cnn_predictions_test = cnn_model.predict(test_generator, verbose=1)

y_true = test_generator.classes

cnn_predictions_prob = cnn_predictions_test.flatten()  
cnn_predictions_binary = (cnn_predictions_prob > 0.5).astype(int) 

cnn_accuracy = accuracy_score(y_true, cnn_predictions_binary)
cnn_report = classification_report(y_true, cnn_predictions_binary)

print("CNN Model Accuracy:", cnn_accuracy)
print("CNN Classification Report:\n", cnn_report)
print("="*50)

if cnn_accuracy > 0.917:
    cnn_model.save("best_cnn_model.keras")


combined_accuracy = (rf_accuracy + cnn_accuracy) / 2
print(combined_accuracy)
