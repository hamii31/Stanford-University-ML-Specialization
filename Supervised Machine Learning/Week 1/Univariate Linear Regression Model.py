import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def model_compute_prediction(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb


def predict():
    w = 200                         
    b = 100    
    x_i = 1.2
    cost_1200sqft = w * x_i + b    

    print(f"${cost_1200sqft:.0f} thousand dollars")

x_train = np.array([1.0, 2.0]) # in 1000s square feet
y_train = np.array([300.0, 500.0]) # in 1000s dollars
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

print(f"x_train.shape: {x_train.shape}")
m = len(x_train)
print(f"Number of training examples is: {m}")

i = 0

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")
  
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
