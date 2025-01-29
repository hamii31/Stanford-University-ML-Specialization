import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


##########################################################
# Title: Neural Network Classification Model for Breast Cancer 
# Model Type: Multilayered Perceptron
# Dataset: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

# Goal: To determine whether the breast tumor is malignant or benign based on these real-valued cell nucleus features:
# 	a) radius (mean of distances from center to points on the perimeter)
# 	b) texture (standard deviation of gray-scale values)
# 	c) perimeter
# 	d) area
# 	e) smoothness (local variation in radius lengths)
# 	f) compactness (perimeter^2 / area - 1.0)
# 	g) concavity (severity of concave portions of the contour)
# 	h) concave points (number of concave portions of the contour)
# 	i) symmetry 
# 	j) fractal dimension ("coastline approximation" - 1)

##########################################################


X_columns = [' Radius1', ' Texture1', ' Perimeter1', ' Area1', ' Smoothness1', ' Compactness1', ' Concavity1', ' Concave_Points1', ' Symmetry1', ' Fractal_Dimension1', 
             ' Radius2', ' Texture2', ' Perimeter2', ' Area2', ' Smoothness2', ' Compactness2', ' Concavity2', ' Concave_Points2', ' Symmetry2', ' Fractal_Dimension2', 
             ' Radius3', ' Texture3', ' Perimeter3', ' Area3', ' Smoothness3', ' Compactness3', ' Concavity3', ' Concave_Points3', ' Symmetry3', ' Fractal_Dimension3']

file_path = 'data/wdbc.csv'
df = pd.read_csv(file_path)

# PREPROCESSING
train_ratio = 0.9 

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

split_index = int(len(df) * train_ratio)

 # 90% training, 10% testing
train_df = df.iloc[:split_index] 
test_df = df.iloc[split_index:]  

test_df.to_csv("data/wdbc_test.csv", index=False)

print("Initial size of Dataframe: ",df.shape[1])

X = train_df[X_columns].to_numpy()
X_test = test_df[X_columns].to_numpy()

print("Initial size of Training set: ", X.shape[1])
print("Initial size of Testing set: ", X_test.shape[1])

# mappings
target_mapping = {
        'M':1,
        'B':0
    }

train_df[' Target'] = train_df[' Target'].map(target_mapping)
Y = train_df[' Target'].to_numpy()

print("Sizes of X: ", X.shape, " and Y: ", Y.shape)

print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")


Xt = np.tile(Xn,(1000,1))
Yt= np.tile(Y,(1000,))   
print("Normalized and copied sizes of X: ", Xt.shape, " and Y: ", Yt.shape) 

# MODEL FITING AND TESTING
tf.random.set_seed(1234) 
model = Sequential(
    [
        tf.keras.Input(shape=(30,)),
        Dense(3, activation='sigmoid', name = 'layer1'),
        Dense(1, activation='sigmoid', name = 'layer2')
     ]
)

model.summary()

L1_num_params = len(X_columns) * 3 + 3   # W1 parameters  + b1 parameters
L2_num_params = 3 * 1 + 1   # W2 parameters  + b2 parameters
print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params  )

W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(
    Xt,Yt,            
    epochs=10,
)

W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

# TESTING

def predict(X_test):
    X_testn = norm_l(X_test)
    predictions = model.predict(X_testn)
    yhat = (predictions >= 0.5).astype(int)
    print(f"decisions = \n{yhat}")
    if yhat >= 0.5:
        print("The tumor is malignant.")
    else:
        print("The tumor is benign.")



print(f"Testing {len(X_test)} cases...")
for i in range(len(X_test)):
    predict(X_test[i])

# Check the test csv file to make sure that the predictions are correct
