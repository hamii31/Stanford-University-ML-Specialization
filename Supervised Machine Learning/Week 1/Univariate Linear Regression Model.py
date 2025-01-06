import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Add gradient descent to the linear regression model
# First compute the gradient (The derivative terms)
def compute_gradient(x, y, w ,b):
    m = x.shape[0];
    dj_dw = 0   # d * J(w,b) / dw
    dj_db = 0   # d * J(w,b) / db
    
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db

def gradient_descent(x, y, w_initial, b_initial, alpha, iterations):
    b = b_initial
    w = w_initial
    J_history = []
    p_history = []
    
    for i in range(iterations):
        # calculate gradient
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        
        # update params simultaneously (because the derivatives are already calculated)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
       

    return w, b


# Add a cost function to compute cost
#                 m
# J(w,b) = 1/2m * ∑(ŷ^(i) - y^(i))^2
#                i=1
# OR
#                 m
# J(w,b) = 1/2m * ∑(f(w,b)(x(i)) - y^(i))^2
#                i=1

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    
    for i in range(m):
        f_wb = w * x[i] + b     # f(w,b)(x(i))
        cost = (f_wb - y[i])**2
        cost_sum = cost_sum + cost
        
    total_cost = ( 1 / (2 * m)) * cost_sum
    
    return total_cost   # You want the cost to be as minimal as possible

def model_compute_prediction(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb


def predict(w, b):
    x_i = 1.2  # in 1000s sqft
    predicted_house_cost = w * x_i + b    

    return predicted_house_cost, x_i * 1000

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2]) # 1000s sqft 
y_train = np.array([250, 300, 480,  430,   630, 730,]) # 1000s dollars
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")


print(f"x_train.shape: {x_train.shape}")
m = len(x_train)
print(f"Number of training examples is: {m}")

i = 0

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# Using the cost function I have determined that b = 23.5 and w = 200 leads to the minimum cost
w = 200
b = 23.5
print(f"w: {w}")
print(f"b: {b}")

# use this to fit the model to the dataset
cost = compute_cost(x_train, y_train, w, b)
print(f"Current cost: {cost}")

prediction, sqft = predict(w, b)
print(f"The prediction is ${prediction:.0f} thousand dollars for a {sqft} ft house BEFORE gradient descent.")
tmp_prediction = prediction

# GRADIENT DESCENT
# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w, b = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, iterations)

print("The best w and b params found by the gradient descent algorithm are as follows:")
print(f"w = {w:8.4f}")
print(f"b = {b:8.4f}")

# compute the cost of w and b after the gradient descent
cost = compute_cost(x_train, y_train, w, b)
print(f"Cost after gradient descent: {cost}")

tmp_f_wb = model_compute_prediction(x_train, w, b)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

prediction, sqft = predict(w, b)
print(f"The prediction is ${prediction:.0f} thousand dollars for a {sqft} ft house AFTER gradient descent.")
print(f"That's about ${tmp_prediction - prediction:.0f} thousand dollar difference in cost ;)")

# CONCLUSION: 
# The gradient descent algorithm is a strong algorithm in Machine Learning that helps with minimizing the cost,
# thus leading to better results in prediction. 
