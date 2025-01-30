import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
np.set_printoptions(precision=2)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


# 433 Test Cases, 82% Accuracy
# Multiclass Classification Neural Network 
# Lung Doctor


def remove_zero_variation(X):
    std_dev = np.std(X, axis=0)
    
    mask = (std_dev != 0)
    X_filtered = X[:, mask]
    
    column_variances = np.var(X, axis=0)

    non_zero_variance_columns = column_variances != 0
    
    return X_filtered, non_zero_variance_columns
   

file_path = 'data/lungdiseases.csv'
df = pd.read_csv(file_path)

train_ratio = 0.9

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

split_index = int(len(df) * train_ratio)

train_df = df.iloc[:split_index] 
test_df = df.iloc[split_index:]  

test_df["diseases"].to_csv("data/lungdiseases_test.csv", index=False)

print(df.shape[1])

X = train_df.drop(columns=['diseases']).to_numpy()
X_test = test_df.drop(columns=['diseases']).to_numpy()

print(X.shape[1])
print(X_test.shape[1])

X, features = remove_zero_variation(X)
X_test, features = remove_zero_variation(X_test)

print(len(features))

print(X.shape[1])
print(X_test.shape[1])

disease_mapper = {
    'lung cancer':1,
    'asthma':2,
    'pneumonia':3,
    'chronic obstructive pulmonary disease (copd)':4,
    'acute bronchitis':5,
    'pulmonary embolism':6,
    'pulmonary fibrosis':7
    }

train_df['diseases'] = train_df['diseases'].map(disease_mapper)
Y = train_df['diseases'].to_numpy()

norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  
Xn = norm_l(X)

Xt = np.tile(Xn,(100,1))
Yt= np.tile(Y,(100,))   

tf.random.set_seed(1234)

model = Sequential(
    [
        Dense(len(features), activation = 'relu'),
        Dense(len(features) - 10, activation = 'relu'),
        Dense(8, activation = 'linear')
    ]
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.0003),
)

# save best weights
save_model = "my_model.keras"
checkpoint = ModelCheckpoint(save_model, monitor="loss", save_best_only=True, mode="min")

model.fit(
    Xt,Yt,
    epochs=50,
    callbacks=[checkpoint]
)

# 10, 100 epochs


# TESTING
file_path = 'data/lungdiseases_test.csv'
check_test_df = pd.read_csv(file_path)

# Load the saved best weights (Uncomment)
# model.load_weights("save_model")

logits = model.predict(X_test)
f_x = tf.nn.softmax(logits).numpy()

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
        

def check_accuracy(i, label): 
    column_name = "diseases"

    if check_test_df.at[i, column_name] == label:
        return "1.0"
    else:
        return "0.0"
   
correct_labels = 0

for i in range(len(X_test)):
    label = reverse_mapping(np.argmax(logits[i]))
    acc = check_accuracy(i, label)
    print( f"predicting for: {i}, category: {label}, accuracy: {acc}")
    
    if acc == "1.0":
        correct_labels += 1
    
print(correct_labels)
print(f"Average accuracy of the model: {(correct_labels / len(X_test)):0.2f}")
