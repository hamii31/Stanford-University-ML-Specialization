from re import X
import numpy as np
import math, copy
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(precision=2)
plt.style.use('ggplot')

##########################################################
# Title:  Medical Cost Prediction 
# Model: Multiple Linear Regression, Segmented
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
    
def custom_train_and_predict(X, y, subject):
    X_norm, X_mu, X_sigma = zscore_normalize_features(X, True)
    w_norm, b_norm = run_gradient_descent(X_norm, y, 1000, 1.0e-1)
    X_features = ['age','gender','bmi','children', 'smoker', 'region']

    fig,ax=plt.subplots(1, 6, figsize=(12, 3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X[:,i],y)
        ax[i].set_xlabel(X_features[i])
    ax[0].set_ylabel("Outcome for Students")
    plt.show()

    x_subj = subject
    x_subj_norm = (x_subj - X_mu) / X_sigma
    x_subj_predict = np.dot(x_subj_norm, w_norm) + b_norm
    print(f"Predicted medical cost: ${x_subj_predict:0.2f}")
    
def create_avg_test_subj(data, age_group):
    avg_age = math.ceil(data['age'].mean())
    leading_gender = data['gender'].value_counts().idxmax()
    avg_bmi = math.ceil(data['bmi'].mean())
    avg_children = math.ceil(data['children'].mean())
    smoking_status = data['smoker'].value_counts().idxmax()
    avg_region = math.ceil(data['region'].mean())
        
    X = data[X_columns].to_numpy()
    y = data['charges'].to_numpy()
         
    print(f"Computing average test subject from {len(X)} cases in the {age_group} group.")
    print(f"Age: {avg_age}, gender: {leading_gender}, BMI: {avg_bmi}, children: {avg_children}, smoker: {smoking_status}, region: {avg_region}")
        
    patient = np.array([avg_age, leading_gender, avg_bmi, avg_children, smoking_status, avg_region])       
    custom_train_and_predict(X, y, patient)  


file_path = 'data/insurance.csv'
df = pd.read_csv(file_path)

# mappings
gender_mapping = {
        'male':1,
        'female':0
    }

smoker_mapping = {
        'yes':1,
        'no':0
    }

region_mapping = {
        'southwest':0,
        'southeast':1,
        'northwest':2,
        'northeast':3
    }

# Map the columns 
df['gender'] = df['gender'].map(gender_mapping)
df['smoker'] = df['smoker'].map(smoker_mapping)
df['region'] = df['region'].map(region_mapping)

# Define age ranges
age_ranges = {
    'students': df[(df['age'] >= 18) & (df['age'] <= 24)],
    'young adults': df[(df['age'] >= 25) & (df['age'] <= 35)],
    'adults': df[(df['age'] >= 36) & (df['age'] <= 49)],
    'senior adults': df[(df['age'] >= 49) & (df['age'] <= 59)],
    'seniors': df[df['age'] >= 60]
}

X_columns = ['age', 'gender', 'bmi', 'children', 'smoker', 'region']  

##########################################################
# Model Segmentation
##########################################################
for age_group, data in age_ranges.items():
    if age_group == 'students':
        create_avg_test_subj(data, age_group)
        
    if age_group == 'young adults':
        create_avg_test_subj(data, age_group)
        
    if age_group == 'adults':
        create_avg_test_subj(data, age_group)
     
    if age_group == 'senior adults':
        create_avg_test_subj(data, age_group)
     
    if age_group == 'seniors':
        create_avg_test_subj(data, age_group)
     

# Now for the entire dataset
create_avg_test_subj(df, "global")
