import os
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load tabular data (patient history, symptoms)
data = pd.read_csv("lungdiseases.csv")
data['diseases'] = data['diseases'].apply(lambda x: 1 if x == 'pneumonia' else 0)

X = data.drop(columns=['diseases'])  
y = data['diseases']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Starting Random Forest for feature selection...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
feature_importances = rf.feature_importances_
selected_features = X_train.columns[np.argsort(feature_importances)[-10:]]  # Gets top 10 features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
print("Done!")

data_dir = "C:/Users/Hami/Downloads/archive(13)"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

image_size = (128, 128)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(train_dir, target_size=image_size, color_mode='grayscale', batch_size=batch_size, class_mode='binary')
val_generator = datagen.flow_from_directory(val_dir, target_size=image_size, color_mode='grayscale', batch_size=batch_size, class_mode='binary')
test_generator = datagen.flow_from_directory(test_dir, target_size=image_size, color_mode='grayscale', batch_size=batch_size, class_mode='binary', shuffle=False)

print(train_generator.samples)
print(test_generator.samples)

print("Starting CNN for image classification...")
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
print("Done!")
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(train_generator, epochs=10, validation_data=val_generator)

# Extracts the same amount of samples 
cnn_predictions_train = cnn_model.predict(train_generator, verbose=1)
cnn_predictions_test = cnn_model.predict(test_generator, verbose=1)

train_num_samples = cnn_predictions_train.shape[0]
test_num_samples = cnn_predictions_test.shape[0]
X_train_selected = X_train_selected.sample(n=train_num_samples, random_state=42)
X_test_selected = X_test_selected.sample(n=test_num_samples, random_state=42)

indices = X_train_selected.sample(n=train_num_samples, random_state=42).index
y_train = y_train.loc[indices].values

print("Combining predictions in Boosted Tree...")
xgb_model = xgb.XGBClassifier()
meta_train = np.hstack((X_train_selected, cnn_predictions_train))
meta_test = np.hstack((X_test_selected, cnn_predictions_test))
xgb_model.fit(meta_train, y_train)

final_predictions = xgb_model.predict(meta_test)

min_length = min(len(final_predictions), len(y_test))
accuracy = accuracy_score(y_test[:min_length], final_predictions[:min_length])
print("Final Hybrid Model Accuracy:", accuracy)
