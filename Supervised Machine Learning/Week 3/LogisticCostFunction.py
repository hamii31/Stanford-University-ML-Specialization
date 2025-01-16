import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Dataset
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])                                           #(m,)

# SIGMOID FUNCTION
def sigmoid(z):

    g = 1/(1+np.exp(-z))
   
    return g


# LOGISTIC COST FUNCTION
def compute_cost_logistic(X, y, w, b):

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost

w_tmp = np.array([1,1])
b_tmp = -3
print("Cost for b = -3 : ", compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))


# Example for a bad cost
w_tmp = np.array([1,1])
b_tmp = -4

print("Cost for b = -4 : ", compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))
