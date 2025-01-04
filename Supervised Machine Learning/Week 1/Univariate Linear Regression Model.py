import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

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
    
    print(f"The cost is: {total_cost}")     # You want the cost to be as minimal as possible

def model_compute_prediction(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb


def predict():
    w = 200                         
    b = 23.5
    x_i = 1.2
    cost_1200sqft = w * x_i + b    

    print(f"${cost_1200sqft:.0f} thousand dollars for a {x_i * 1000} ft house")

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

# The cost function determined that b = 23.5 and w = 200 leads to the minimum cost -> more accurate model
w = 200
b = 23.5
print(f"w: {w}")
print(f"b: {b}")

# use this to fit the model to the dataset
compute_cost(x_train, y_train, w, b)

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


    

predict()
