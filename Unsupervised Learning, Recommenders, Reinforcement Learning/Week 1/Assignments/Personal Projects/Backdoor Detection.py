import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve


data = pd.read_csv('C:/Users/Hami/Downloads/UNSW_NB15_traintest_backdoor/UNSW_NB15_traintest_backdoor.csv')
data_features = data.drop(columns=['class'])
actual_labels = data['class'].values

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data_features)

train_data, test_data, train_labels, test_labels = train_test_split(data_normalized, actual_labels, test_size=0.2, random_state=42)

print(f'Training data shape: {train_data.shape}')
print(f'Testing data shape: {test_data.shape}')

# Autoencoder Model
input_dim = train_data.shape[1]
input_layer = keras.layers.Input(shape=(input_dim,))
encoded = keras.layers.Dense(16, activation='relu')(input_layer)
encoded = keras.layers.Dense(8, activation='relu')(encoded)
decoded = keras.layers.Dense(16, activation='relu')(encoded)
decoded = keras.layers.Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = keras.Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.load_weights('autoencoder_weights.weights.h5')

autoencoder.fit(train_data, train_data, epochs=50, batch_size=32, shuffle=True, validation_data=(test_data, test_data))

autoencoder.save_weights('autoencoder_weights.weights.h5')

# Reconstruction errors
reconstructed = autoencoder.predict(test_data)
mse = np.mean(np.power(test_data - reconstructed, 2), axis=1)

# Calculate precision and recall for different thresholds
precision, recall, thresholds = precision_recall_curve(test_labels, mse)

plt.figure()
plt.plot(recall, precision, marker='o')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.show()

# Optimal threshold
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f'Optimal Threshold: {optimal_threshold}')

anomalies = mse > optimal_threshold

for i, is_anomaly in enumerate(anomalies):
    if is_anomaly:
        print(f"Sample {i} is an anomaly (MSE: {mse[i]:.6f})")
    else:
        print(f"Sample {i} is normal (MSE: {mse[i]:.6f})")
        
predicted_labels = anomalies.astype(int)

accuracy = accuracy_score(test_labels, predicted_labels)

print(f'Accuracy: {accuracy:.2f}')
