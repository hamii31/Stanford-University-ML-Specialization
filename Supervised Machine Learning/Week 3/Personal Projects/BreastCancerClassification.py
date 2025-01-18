from re import sub
import numpy as np
import math, copy
import matplotlib.pyplot as plt
import pandas as pd
import csv

np.set_printoptions(precision=2)
plt.style.use('ggplot')

##########################################################
# Title:  Breast Cancer Classification  
# Model: Logistic Regression
# Algorithms: Logistic Cost Function, Logistic Gradient Descent, Z-Score Normalization (Standard Deviation, Mean)
# Dataset: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
# Additional Information about the Dataset:
# Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.  
# They describe characteristics of the cell nuclei present in the image. A few of the images can be found at http://www.cs.wisc.edu/~street/images/
# Separating plane described above was obtained using Multisurface Method-Tree (MSM-T) 
# [K. P. Bennett, "Decision Tree Construction Via Linear Programming." Proceedings of the 4th Midwest Artificial Intelligence and 
# Cognitive Science Society, pp. 97-101, 1992], a classification method which uses linear programming to construct a decision tree.  
# Relevant features were selected using an exhaustive search in the space of 1-4 features and 1-3 separating planes.

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
 
##########################################################
# Algorithms
##########################################################


# SIGMOID FUNCTION
def sigmoid(z):

    z = np.clip(z, -500, 500)
    g = 1/(1+np.exp(-z))
   
    return g

#LOGISTIC COST FUNCTION
def compute_cost_logistic(X, y, w, b):

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost

# GRADIENT DESCENT
def compute_gradient_logistic(X, y, w, b): 
    m,n = X.shape
    dj_dw = np.zeros((n,))                           
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          
        err_i  = f_wb_i  - y[i]                     
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   
    dj_db = dj_db/m                                   
        
    return dj_db, dj_dw  


def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    w = copy.deepcopy(w_in) 
    b = b_in
    
    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)   
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
   
def make_prediction(tumor, X_mu, X_sigma):
    tumor_normalized = (tumor - X_mu) / X_sigma 

    z_i = np.dot(tumor_normalized,w_out) + b_out 
    predict = sigmoid(z_i)
    if predict <= 0.5:
        percentage = 100 - (predict * 100)
        print(f"{percentage:0.2f}% Benign")
    else:
        percentage = predict * 100
        print(f"{percentage:0.2f}% Malignant")


def data_to_csv():

    input_file = "data/wdbc.data"
    output_file = "data/wdbc.csv"
    # Read and convert the .data file
    with open(input_file, "r") as data_file:
        lines = data_file.readlines()

    with open(output_file, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
    
        writer.writerow(X_columns)
        for line in lines:
            # Split the line into columns (adjust delimiter if needed)
            columns = line.strip().split(",")
            writer.writerow(columns)

# PREPARE THE DATASET

# Uncomment if in need to convert a .data file to a .csv file
# data_to_csv()

X_columns = [' Radius1', ' Texture1', ' Perimeter1', ' Area1', ' Smoothness1', ' Compactness1', ' Concavity1', ' Concave_Points1', ' Symmetry1', ' Fractal_Dimension1', 
             ' Radius2', ' Texture2', ' Perimeter2', ' Area2', ' Smoothness2', ' Compactness2', ' Concavity2', ' Concave_Points2', ' Symmetry2', ' Fractal_Dimension2', 
             ' Radius3', ' Texture3', ' Perimeter3', ' Area3', ' Smoothness3', ' Compactness3', ' Concavity3', ' Concave_Points3', ' Symmetry3', ' Fractal_Dimension3']

file_path = 'data/wdbc.csv'
df = pd.read_csv(file_path)


X_train = df[X_columns].to_numpy()

# mappings
target_mapping = {
        'M':1,
        'B':0
    }

df[' Target'] = df[' Target'].map(target_mapping)
y_train = df[' Target'].to_numpy()

# normalize 
print("Normalizing data set...")
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train, True)

# RUN GRADIENT DESCENT 
w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

print("Running gradient descent...")
w_out, b_out = gradient_descent(X_norm, y_train, w_tmp, b_tmp, alph, iters) 
print(f"Best parameters: w:{w_out}, b:{b_out}") # Last values will be the best parameters because the cost is the lowest

# TESTING
#Malignant Tumors (1):

#     Higher values for mean, standard error, and worst-case features such as radius, texture, perimeter, area, and compactness.
#     Lower symmetry and smoothness compared to benign tumors.
#     More irregularity in features like concavity and concave points.

# Benign Tumors (0):

#     Lower values for the majority of features, particularly radius, perimeter, area, and compactness.
#     Features tend to be more uniform and smooth.

print("Making predictions...")

# Benign samples
benign_data = np.array([
    13.54,	15.71,	87.46,	566.3,	0.09779,	0.06492,	0.06664,	0.04781,	0.1967,	0.05766,
    0.2699,	0.9768,	2.058,	23.56,	0.008462,	0.0146,0.01985,	0.01315,	0.0198,	0.0023,	15.11,	
    19.26,	99.7,	711.2,	0.144,	0.1773,	0.239,	0.1288,	0.2977,	0.07259
])
make_prediction(benign_data, X_mu, X_sigma)
print()

benign_data = np.array([
    14.26, 19.65, 97.83,	629.9,	0.07837,	0.2233,	0.3003,	0.07798,	0.1704,	0.07769,
    0.3628,	1.49,	3.399,	29.25,	0.005298,	0.07446,	0.1435,	0.02292,	0.02566,	0.01298,
    15.3,	23.73, 107,	709,	0.08949,	0.4193,	0.6783,	0.1505,	0.2398, 0.1082
])
make_prediction(benign_data, X_mu, X_sigma)
print()

# Malignant samples
malignant_data = np.array([
    15.34,	14.26,	102.5,	704.4,	0.1073,	0.2135,	0.2077,	0.09756, 0.2521,	0.07032,	
    0.4388,	0.7096,	3.384,	44.91,	0.006789,	0.05328,	0.06446,	0.02252, 0.03672,
    0.004394,	18.07,	19.08,	125.1,	980.9,	0.139,	0.5954,	0.6305,	0.2393,	0.4667,	0.09946
])
make_prediction(malignant_data, X_mu, X_sigma)
print()

malignant_data = np.array([
    14.78, 23.94,	97.4,	668.3,	0.1172,	0.1479,	0.1267,	0.09029,	0.1953,	0.06654,	
    0.3577,	1.281,	2.45,	35.24,	0.006703,	0.0231,	0.02315,	0.01184,	0.019,	0.003224,
    17.31,	33.39,	114.6,	925.1,	0.1648,	0.3416,	0.3024,	0.1614,	0.3321,	0.08911
])
make_prediction(malignant_data, X_mu, X_sigma)
print()
