import numpy as np
import matplotlib.pyplot as plt

# Input is an array. 
input_array = np.array([1,2,3])
exp_array = np.exp(input_array) # NumPy has a function called exp(), which offers a convenient way to calculate the exponential 
                                # (e^z) of all elements in the input array (z).

print("Input to exp:", input_array)
print("Output of exp:", exp_array)

# Input is a single number
input_val = 1  
exp_val = np.exp(input_val)

print("Input to exp:", input_val)
print("Output of exp:", exp_val)

# SIGMOID FUNCTION
def sigmoid(z):

    g = 1/(1+np.exp(-z))
   
    return g


# Generate an array of evenly spaced values between -10 and 10
z_tmp = np.arange(-10,11)

# Use the function implemented above to get the sigmoid values
y = sigmoid(z_tmp)

# Code for pretty printing the two arrays next to each other
np.set_printoptions(precision=3) 
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])

#The values in the left column are z, and the values in the right column are sigmoid(z). 
#As you can see, the input values to the sigmoid range from -10 to 10, and the output values range from 0 to 1.
#Now, let's try to plot this function using the matplotlib library.

# Plot z vs sigmoid(z)
fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(z_tmp, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
plt.show()

# As you can see, the sigmoid function approaches 0 as z goes to large negative values and approaches 1 as z goes to large positive values.
