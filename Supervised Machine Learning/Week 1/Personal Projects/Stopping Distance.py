import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


# What I learned by doing this project:
# - Handling overflows
# - Having a verbose training set is crucial for the model to work right, even if it's a basic model
# - Knowing how to fit your model around the training set is very important

# How to improve the model?
# - Use a curved line function (Will learn that in week 2 and implement it here)
# - Make the training set even more verbose


def compute_gradient(x, y, w ,b):
    m = x.shape[0];
    dj_dw = 0   
    dj_db = 0   
    
    for i in range(m):
        f_wb = w * (x[i]/10) + b
        dj_dw_i = (f_wb - y[i]) * (x[i]/10)
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db

def gradient_descent(x, y, w_initial, b_initial, alpha, iterations):
    b = b_initial
    w = w_initial
    
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
       

    return w, b

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    
    for i in range(m):
        f_wb = w * (x[i]/10) + b    
        cost = (f_wb - y[i])**2
        cost_sum = cost_sum + cost
        
    total_cost = ( 1 / (2 * m)) * cost_sum
    
    return total_cost

def model_compute_prediction(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * (x[i]/10) + b
        
    return f_wb


def predict(w, b, n):
    x_i = n 
    distance_to_stop = w * (x_i/10) + b    

    return distance_to_stop, x_i

# Training set built around Forensic Dynamics Inc.'s Braking Calculator - http://www.forensicdynamics.com/stopping-braking-distance-calculator

x_train = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 
                   120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200]).astype(np.float64) # speed (km/h)

y_train = np.array([0.5, 1.26, 2.24, 3.51, 5.06, 6.88, 8.99, 11.38, 14.05, 17.01, 20.24, 23.76, 27.55, 31.66, 
                    35.99, 40.66, 45.55, 50.75, 56.23, 62, 68.04, 74.37, 80.98, 87.87, 95.04, 102.49, 110.22,
                    118.24, 126.53, 135.11, 143.97, 153.11, 162.53, 172.23, 182.21, 192.47, 203.02, 213.85,
                    224.95]).astype(np.float64) # stopping distance on dry asphalt (in meters)

m = len(x_train)
print(f"Number of training examples is: {m}")

w_init = 0
b_init = 0
iterations = 100000
tmp_alpha = 1.0e-2
w, b = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, iterations)

print("The best w and b params found by the gradient descent algorithm are as follows:")
print(f"w = {w:8.4f}")
print(f"b = {b:8.4f}")

cost = compute_cost(x_train, y_train, w, b)
print(f"J = {cost:0.4f}")

tmp_f_wb = model_compute_prediction(x_train, w, b)

plt.plot(x_train, tmp_f_wb, c='b',label='f(w,b)')
plt.scatter(x_train, y_train, marker='x', c='r',label='Values')
plt.title("Stopping Distance vs Speed")
plt.ylabel('Stopping Distance (in meters)')
plt.xlabel('Speed (in km/h)')
plt.legend()
plt.show()

n_test = np.array([50, 70, 90, 110, 130, 150, 170, 190, 210]) # test different speeds in km/h
n = len(n_test)

for i in range(n):
    distance, speed = predict(w, b, n_test[i])
    print(f"If a car is moving at {speed}km/h it will need {distance:0.2f}m to fully stop.")

