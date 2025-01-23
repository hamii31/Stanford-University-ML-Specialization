from os import remove
from re import sub
import numpy as np
import math, copy, time, sys
import matplotlib.pyplot as plt
import pandas as pd
import csv

np.set_printoptions(precision=2)
plt.style.use('ggplot')

##########################################################
# Title:  Lung Cancer Identification  
# Model: Regularized Logistic Regression
# Algorithms: Regularized Logistic Cost Function, Regularized Logistic Gradient Descent, Z-Score Normalization (Standard Deviation, Mean), Zero-Variation Dataset Filtration
# Dataset: https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset

# Goal: To identify whether the 377 symptoms are of a lung cancer, or those of other lung diseases such as COPD, pulmonary ebmolism, bronchitis, etc.

##########################################################
 
##########################################################
# Algorithms
##########################################################


# SIGMOID FUNCTION
def sigmoid(z):

    z = np.clip(z, -500, 500)
    g = 1/(1+np.exp(-z))
   
    return g

#REGULARIZED LOGISTIC COST FUNCTION
def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):

    m,n  = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b                                      
        f_wb_i = sigmoid(z_i)                                          
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)      
             
    cost = cost/m                                                     

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          
    reg_cost = (lambda_/(2*m)) * reg_cost                             
    
    total_cost = cost + reg_cost                                       
    return total_cost                                                  

# COMPUTE REGULARIZED LOGISTIC GRADIENT DESCENT
def compute_gradient_logistic_reg(X, y, w, b, lambda_): 
    m,n = X.shape
    dj_dw = np.zeros((n,))                           
    dj_db = 0.0                                      

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          
        err_i  = f_wb_i  - y[i]                       
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]    
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                 
    dj_db = dj_db/m                                   

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw  

# GRADIENT DESCENT
def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    w = copy.deepcopy(w_in) 
    b = b_in
    lambda_ = 0.7
    
    for i in range(num_iters):
        loading_bar(i, num_iters)
        dj_db, dj_dw = compute_gradient_logistic_reg(X, y, w, b, lambda_)   
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
    

    return w, b       

# ZSCORE NORMALIZATION
def zscore_normalize_features(X,rtn_ms=False):
    mu     = np.mean(X,axis=0)  
    sigma  = np.std(X,axis=0)
    
    X_norm = (X - mu)/sigma      
    
    print("Mean (mu):", mu)
    print("Standard Deviation (sigma):", sigma)
    print("Normalized X:", X_norm)

    if rtn_ms:
        return(X_norm, mu, sigma)
    else:
        return(X_norm)
    

def remove_zero_variation(X):
    # Calculate the standard deviation for each column
    std_dev = np.std(X, axis=0)
    
    mask = (std_dev != 0)
    X_filtered = X[:, mask]
    
    # Calculate variance for each column
    column_variances = np.var(X, axis=0)

    # Identify columns with non-zero variance
    non_zero_variance_columns = column_variances != 0
    
    return X_filtered, non_zero_variance_columns
   
# PREDICTION
def make_prediction(symptoms, X_mu, X_sigma, iteration):
    symptoms_normalized = (symptoms - X_mu) / X_sigma 
    z_i = np.dot(symptoms_normalized,w_out) + b_out 
    predict = sigmoid(z_i)
    
    if predict > 0.5:
        # Check the lungdiseases_test.csv file to make sure the results are correct
        percentage = predict * 100
        print(f"{iteration + 2}:{percentage:0.2f}% lung cancer")
        

# LOADING BAR
def loading_bar(iteration, total, length=30):
    progress = int((iteration + 1) / total * length)
    bar = f"[{'#' * progress}{'.' * (length - progress)}]"
    percent = f"{((iteration + 1) / total) * 100:.2f}%"
    print(f"\r{bar} {percent} ({iteration + 1}/{total})", end="", flush=True)


file_path = 'data/lungdiseases.csv'
df = pd.read_csv(file_path)

# Set the split ratio
train_ratio = 0.9  # 90% training, 10% testing

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Compute the split index
split_index = int(len(df) * train_ratio)

# Split the DataFrame
train_df = df.iloc[:split_index]  # First 70% for training
test_df = df.iloc[split_index:]  # Remaining 30% for testing

test_df.to_csv("data/lungdiseases_test.csv", index=False)

print(df.shape[1])

# get all columns except the target
X_train = train_df.drop(columns=['diseases']).to_numpy()
X_test = test_df.drop(columns=['diseases']).to_numpy()

print(X_train.shape[1])
print(X_test.shape[1])

# Remove zero variation to avoid errors
X_train, columns = remove_zero_variation(X_train)
X_test, columns = remove_zero_variation(X_test)

print(columns)


print(X_train.shape[1])
print(X_test.shape[1])


# map for panic disorder
df['diseases'] = df['diseases'].map(lambda x: 1 if x == 'lung cancer' else 0)
y_train = df['diseases'].to_numpy()

# normalize 
print("Normalizing data set...")
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train, True)

#UNCOMMENT TO DISPLAY GRAPHICAL DATA
Column_names = df.columns.tolist()

# RUN GRADIENT DESCENT 
w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

print("Running gradient descent...")
w_out, b_out = gradient_descent(X_norm, y_train, w_tmp, b_tmp, alph, iters) 
print(f"Best parameters: w:{w_out}, b:{b_out}") # Last values will be the best parameters because the cost is the lowest

# TESTING
print(f"Testing {len(X_test)} cases...")
for i in range(len(X_test)):
    make_prediction(X_test[i], X_mu, X_sigma, i)
