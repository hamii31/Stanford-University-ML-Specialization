import numpy as np
import math, copy
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)
plt.style.use('ggplot')

##########################################################
# Title:  Medical Cost Prediction 
# Model: Multiple Linear Regression
# Algorithms: Cost Function, Gradient Descent, Z-Score Normalization (Standard Deviation, Mean)
# Dataset: https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv
# Goal: To predict medical expenses based on their demographic, lifestyle, and health-related factors.
##########################################################
 
##########################################################
# Regression Routines
##########################################################

def compute_gradient_matrix(X, y, w, b): 
    m,n = X.shape
    f_wb = X @ w + b              
    e   = f_wb - y                
    dj_dw  = (1/m) * (X.T @ e)    
    dj_db  = (1/m) * np.sum(e)    
        
    return dj_db,dj_dw

def compute_cost(X, y, w, b): 
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i],w) + b       
        cost = cost + (f_wb_i - y[i])**2              
    cost = cost/(2*m)                                 
    return(np.squeeze(cost)) 

def compute_gradient(X, y, w, b): 
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i,j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw/m                                
    dj_db = dj_db/m                                
        
    return dj_db,dj_dw

def run_gradient_descent(X,y,iterations=1000, alpha = 1e-6):

    m,n = X.shape
    initial_w = np.zeros(n)
    initial_b = 0
    
    w_out, b_out = gradient_descent(X ,y, initial_w, initial_b,
                                               compute_cost, compute_gradient_matrix, alpha, iterations)
    
    return(w_out, b_out)

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    m = len(X)
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db,dj_dw = gradient_function(X, y, w, b)   
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
       
    return w, b

def zscore_normalize_features(X,rtn_ms=False):
    mu     = np.mean(X,axis=0)  
    sigma  = np.std(X,axis=0)
    X_norm = (X - mu)/sigma      

    if rtn_ms:
        return(X_norm, mu, sigma)
    else:
        return(X_norm)
    
X_train = np.array([[19, 0, 27.9, 0, 1, 0], [18, 1, 33.77, 1, 0, 1], [28, 1, 33, 3, 0, 1], [33, 1, 22.705, 0, 0, 3], [32, 1, 28.88, 0, 0, 3],
                    [31, 0, 25.74, 0, 0, 1], [46, 0, 33.44, 1, 0, 1], [37, 0, 27.74, 3, 0, 3], [37, 1, 29.83, 2, 0, 2], [60, 0, 25.84, 0, 0, 3],
                    [25, 1, 26.22, 0, 0, 2], [62, 0, 26.29, 0, 1, 2], [23, 1, 34.4, 0, 0, 0]]) # age, sex, bmi ratio, children count, smoker, region

y_train = np.array([16884.924, 1725.5523, 4449.462, 21984.47061, 3866.8552, 3756.6216, 8240.5896, 7281.5056, 6406.4107, 28923.13692,
                    2721.3208, 27808.7251, 1826.843]) # insurance cost in dollars

X_features = ['age','sex','bmi','children', 'smoker', 'region'] # sex: 1 = male, 0 = female, smoker: 1 = yes, 0 = no, region: 0 = southwest, 1 = southeast, 2 = northeast, 3 = northwest


# normalize the original features
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train, True)

# Run gradient descent algorithm with normalized data. Note the vastly larger value of alpha. This will speed up gradient descent.
w_norm, b_norm = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1, )


# Plot
fig,ax=plt.subplots(1, 6, figsize=(12, 3), sharey=True)

for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Outcome")
plt.show()

# Evaluate 

# Men
x_patient = np.array([20, 1, 22.5, 0, 0, 0])
x_patient_normalized = (x_patient - X_mu) / X_sigma
x_patient_predict = np.dot(x_patient_normalized, w_norm) + b_norm
print(f"Predicted insurance cost for a man in his 20s, BMI 22.5, no kids, doesn't smoke, from the southwest: ${x_patient_predict:0.2f}")

x_patient = np.array([20, 1, 22.5, 0, 0, 3])
x_patient_normalized = (x_patient - X_mu) / X_sigma
x_patient_predict = np.dot(x_patient_normalized, w_norm) + b_norm
print(f"Predicted insurance cost for a man in his 20s, BMI 22.5, no kids, doesn't smoke, from the northwest: ${x_patient_predict:0.2f}")

# Women
x_patient = np.array([20, 0, 22.5, 0, 0, 1])
x_patient_normalized = (x_patient - X_mu) / X_sigma
x_patient_predict = np.dot(x_patient_normalized, w_norm) + b_norm
print(f"Predicted insurance cost for a woman in her 20s, BMI 22.5, no kids, doesn't smoke, from the southeast: ${x_patient_predict:0.2f}")

x_patient = np.array([20, 0, 22.5, 0, 0, 2])
x_patient_normalized = (x_patient - X_mu) / X_sigma
x_patient_predict = np.dot(x_patient_normalized, w_norm) + b_norm
print(f"Predicted insurance cost for a woman in her 20s, BMI 22.5, no kids, doesn't smoke, from the northeast: ${x_patient_predict:0.2f}")



