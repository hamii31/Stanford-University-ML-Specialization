from re import sub
import numpy as np
import math, copy
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(precision=2)
plt.style.use('ggplot')


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
    dj_dw = np.zeros((n,))                           #(n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar
        
    return dj_db, dj_dw  


def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
       
        
    return w, b        #return final w,b and J history for graphing

def zscore_normalize_features(X,rtn_ms=False):
    mu     = np.mean(X,axis=0)  
    sigma  = np.std(X,axis=0)
    X_norm = (X - mu)/sigma      

    if rtn_ms:
        return(X_norm, mu, sigma)
    else:
        return(X_norm)
   
def make_prediction(subject, X_mu, X_sigma):
    subject_norm = (subject - X_mu) / X_sigma # normalize subject

    z_i = np.dot(subject_norm,w_out) + b_out 
    predict = sigmoid(z_i)
    print(f'Predicted: {predict:0.2f}')
    if predict <= 0.5:
        print("The subject does not have diabetes")
    else:
        print("The subject has diabetes")



file_path = 'data/diabetes.csv'
df = pd.read_csv(file_path)

X_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
# Diabetes Pedigree Function (DPF) - The genetic likelihood of developing diabetes. The higher the number, the more likely to develop diabetes. 
# Example: You have one person with diabetes in your family. The DPF number might be around 0.5 and 1.
# Example 2: You have more than one person with diabetes in your family. The DPF number will be or above 1. 
# Example 3: You have no diabetes family history. The DPF number might be around 0 and 0.5.

X_train = df[X_columns].to_numpy()
y_train = df['Outcome'].to_numpy()

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
print('Original subject')
subject = np.array([6, 148, 72, 35, 0, 33.6, 0.627, 50]) # 0.72
make_prediction(subject, X_mu, X_sigma)
print()

print('How does only lowering the number of pregnancies affect the result of the original subject?')
subject = np.array([0, 148, 72, 35, 0, 33.6, 0.627, 50]) # 0.55
make_prediction(subject, X_mu, X_sigma)
print()
# The number of pregnancies definitely affects the outcome.

print('How does only lowering the age affect the result of the original subject?')
subject = np.array([6, 148, 72, 35, 0, 33.6, 0.627, 23]) # 0.63
make_prediction(subject, X_mu, X_sigma)
print()
# Although the age is also important to determine the likelihood of diabetes, it is not as important as the number of pregnancies.

print('How does only lowering glucose levels affect the result of the original subject?')
subject = np.array([6, 84, 72, 35, 0, 33.6, 0.627, 50]) # 0.21
make_prediction(subject, X_mu, X_sigma)
print()
# Glucose levels turn out to be the most improtant factor in determining if someone has diabetes, no matter the age, number of pregnancies, BMI, 
# blood pressure, insulin levels, DPF or skin thickness.

print('How does only lowering the BMI affect the result of the original subject?')
subject = np.array([6, 148, 72, 35, 0, 23, 0.627, 50]) # 0.50
make_prediction(subject, X_mu, X_sigma)
print()
# The BMI is also a cornerstone for diabetes. 

print('How does only lowering the blood pressure affect the result of the original subject?')
subject = np.array([6, 148, 50, 35, 0, 33.6, 0.627, 50]) # 0.78 
make_prediction(subject, X_mu, X_sigma)
print()
# An interesting outcome. Lowering the blood pressure made the outcome worse. What will happen if we raise it?

print('How does bringing the blood pressure up affect the result of the original subject?')
subject = np.array([6, 148, 90, 35, 0, 33.6, 0.627, 50]) # 
make_prediction(subject, X_mu, X_sigma)
print()
