import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# A simple neural network


X_columns = [' Radius1', ' Texture1', ' Perimeter1', ' Area1', ' Smoothness1', ' Compactness1', ' Concavity1', ' Concave_Points1', ' Symmetry1', ' Fractal_Dimension1', 
             ' Radius2', ' Texture2', ' Perimeter2', ' Area2', ' Smoothness2', ' Compactness2', ' Concavity2', ' Concave_Points2', ' Symmetry2', ' Fractal_Dimension2', 
             ' Radius3', ' Texture3', ' Perimeter3', ' Area3', ' Smoothness3', ' Compactness3', ' Concavity3', ' Concave_Points3', ' Symmetry3', ' Fractal_Dimension3']

file_path = 'data/wdbc.csv'
df = pd.read_csv(file_path)


X = df[X_columns].to_numpy()

# mappings
target_mapping = {
        'M':1,
        'B':0
    }

df[' Target'] = df[' Target'].map(target_mapping)
Y = df[' Target'].to_numpy()

print(X.shape, Y.shape)

print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")


Xt = np.tile(Xn,(1000,1))
Yt= np.tile(Y,(1000,))   
print(Xt.shape, Yt.shape) 


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

# Testing

def predict(X_test):
    X_testn = norm_l(X_test)
    predictions = model.predict(X_testn)
    yhat = (predictions >= 0.5).astype(int)
    print(f"decisions = \n{yhat}")
    if yhat >= 0.5:
        print("The tumor is malignant.")
    else:
        print("The tumor is benign.")

# Benign samples
X_test = np.array([
    13.54,	15.71,	87.46,	566.3,	0.09779,	0.06492,	0.06664,	0.04781,	0.1967,	0.05766,
    0.2699,	0.9768,	2.058,	23.56,	0.008462,	0.0146,0.01985,	0.01315,	0.0198,	0.0023,	15.11,	
    19.26,	99.7,	711.2,	0.144,	0.1773,	0.239,	0.1288,	0.2977,	0.07259
])
predict(X_test)


X_test = np.array([
    14.26, 19.65, 97.83,	629.9,	0.07837,	0.2233,	0.3003,	0.07798,	0.1704,	0.07769,
    0.3628,	1.49,	3.399,	29.25,	0.005298,	0.07446,	0.1435,	0.02292,	0.02566,	0.01298,
    15.3,	23.73, 107,	709,	0.08949,	0.4193,	0.6783,	0.1505,	0.2398, 0.1082
])
predict(X_test)

# Malignant samples
X_test = np.array([
    15.34,	14.26,	102.5,	704.4,	0.1073,	0.2135,	0.2077,	0.09756, 0.2521,	0.07032,	
    0.4388,	0.7096,	3.384,	44.91,	0.006789,	0.05328,	0.06446,	0.02252, 0.03672,
    0.004394,	18.07,	19.08,	125.1,	980.9,	0.139,	0.5954,	0.6305,	0.2393,	0.4667,	0.09946
])
predict(X_test)

X_test = np.array([
    14.78, 23.94,	97.4,	668.3,	0.1172,	0.1479,	0.1267,	0.09029,	0.1953,	0.06654,	
    0.3577,	1.281,	2.45,	35.24,	0.006703,	0.0231,	0.02315,	0.01184,	0.019,	0.003224,
    17.31,	33.39,	114.6,	925.1,	0.1648,	0.3416,	0.3024,	0.1614,	0.3321,	0.08911
])
predict(X_test)
