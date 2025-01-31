import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
np.set_printoptions(precision=2)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


##########################################################
# Title: Lung Doctor
# Model Type: Neural Network for Multiclass Classificaiton with Softmax 
# Average Precision (for 10 runs with shuffeled training and testing sets on each run): 85.897% 
# Best Precision: 88.64%
# Worst Precision: 82.75%
# Dataset: https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset

# Goal: Analyze over 377 symptoms to determine which lung disease the patient has out of 7 possible options.

##########################################################

def remove_zero_variation(X_train, X_test):
    combined = np.vstack((X_train, X_test))
    variances = np.var(combined, axis=0)

    zero_variance_features = np.where(variances == 0)[0]
    
    X_train_cleaned = np.delete(X_train, zero_variance_features, axis=1)
    X_test_cleaned = np.delete(X_test, zero_variance_features, axis=1)
    
    return X_train_cleaned, X_test_cleaned, zero_variance_features
   

file_path = 'data/lungdiseases.csv'
df = pd.read_csv(file_path)

train_ratio = 0.9

df = shuffle(df)

split_index = int(len(df) * train_ratio)

train_df = df.iloc[:split_index] 
test_df = df.iloc[split_index:]  

test_df["diseases"].to_csv("data/lungdiseases_test.csv", index=False)

X = train_df.drop(columns=['diseases']).to_numpy()
X_test = test_df.drop(columns=['diseases']).to_numpy()

# X, X_test, features = remove_zero_variation(X, X_test) 

print("Training set: ", X.shape[1])
print("Testing set: ", X_test.shape[1])

target_mapper = {
    'lung cancer':1,
    'asthma':2,
    'pneumonia':3,
    'chronic obstructive pulmonary disease (copd)':4,
    'acute bronchitis':5,
    'pulmonary embolism':6,
    'pulmonary fibrosis':7
    }

train_df['diseases'] = train_df['diseases'].map(target_mapper)
Y = train_df['diseases'].to_numpy()

norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  
Xn = norm_l(X)

tf.random.set_seed(1234)

print("Feature count: ", X.shape[1])

model = Sequential(
    [
        Dense(X.shape[1], activation = 'relu'),
        Dense(256, activation = 'relu'),
        Dense(128, activation = 'relu'),
        Dense(64, activation = 'relu'),
        Dense(32, activation = 'relu'),
        Dense(16, activation = 'relu'),
        Dense(8, activation = 'linear')
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3), # Adaptive Moment Estimation = 0.003
)

early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True) # overfitting prevention

model.fit(
    X,Y,
    epochs=20,
    batch_size = 32,
    callbacks=early_stopping
)


# TESTING
file_path = 'data/lungdiseases_test.csv'
check_test_df = pd.read_csv(file_path)

check_test_df['diseases'] = check_test_df['diseases'].map(target_mapper)
y_test = check_test_df.to_numpy()

logits = model.predict(X_test)
f_x = tf.nn.softmax(logits).numpy()

y_pred = np.argmax(f_x, axis=1)  # Get predicted class labels
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)  # Weighted for multi-class

print(f"Precision Score:  {precision * 100:0.2f}%") 
# curr best 377 features, 20 epochs, 0.003 ADAM, 7 dense layers total = 87%, 88.64%, 83.36%, 86.60%, 84.29%, 87.08%, 85.36%, 88.36%, 82.75%, 85.53%

def check_accuracy(row, category):
    if check_test_df.at[row, "diseases"] == category:
        return 1.0
    else:
        return 0.0

def reverse_mapping(disease):
    match disease:
        case 1:
            return "lung cancer"
        case 2:
            return "asthma"
        case 3:
            return "pneumonia"
        case 4:
            return "chronic obstructive pulmonary disease (copd)"
        case 5:
            return "acute bronchitis"
        case 6:
            return "pulmonary embolism"
        case 7:
            return "pulmonary fibrosis"
        case _:
            return "Unknown disease"
        
for i in range(len(X_test)):
    print( f"{f_x[i]}, category: {reverse_mapping(np.argmax(f_x[i]))}")
