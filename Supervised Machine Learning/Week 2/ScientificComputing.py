import numpy as np
import time

# VECTOR CREATION
print("VECTOR CREATION")
# NumPy routines which allocate memory and fill arrays with value
print("NumPy routines which allocate memory and fill arrays with value")
a = np.zeros(4);                print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,));             print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
print("NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument")
a = np.arange(4.);              print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4);          print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# NumPy routines which allocate memory and fill with user specified values
print("NumPy routines which allocate memory and fill with user specified values")
a = np.array([5,4,3,2]);  print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

print()
# INDEXING
print("INDEXING")
#vector indexing operations on 1-D vectors
print("vector indexing operations on 1-D vectors")
a = np.arange(10)
print(a)

#access an element
print("access an element")
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# access the last element, negative indexes count from the end
print("access the last element, negative indexes count from the end")
print(f"a[-1] = {a[-1]}")

#indexs must be within the range of the vector or they will produce an error
print("indexes must be within the range of the vector or they will produce an error")
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)
    


print()
# SLICING
print("SLICING")    
#vector slicing operations
a = np.arange(10)
print(f"a         = {a}")

#access 5 consecutive elements (start:stop:step)
print("access 5 consecutive elements (start:stop:step)")
c = a[2:7:1];     print("a[2:7:1] = ", c)

# access 3 elements separated by two 
print("access 3 elements separated by two")
c = a[2:7:2];     print("a[2:7:2] = ", c)

# access all elements index 3 and above
print("access all elements index 3 and above")
c = a[3:];        print("a[3:]    = ", c)

# access all elements below index 3
print("access all elements below index 3")
c = a[:3];        print("a[:3]    = ", c)

# access all elements
print("access all elements")
c = a[:];         print("a[:]     = ", c)    

print()
# SINGLE VECTOR OPERATIONS
print("SINGLE VECTOR OPERATIONS")
a = np.array([1,2,3,4])
print(f"a             : {a}")
# negate elements of a
print("negate elements of a")
b = -a 
print(f"b = -a        : {b}")

# sum all elements of a, returns a scalar
print("sum all elements of a, return scalar")
b = np.sum(a) 
print(f"b = np.sum(a) : {b}")

b = np.mean(a)
print(f"b = np.mean(a): {b}")

b = a**2
print(f"b = a**2      : {b}")

print()
# ELEMENT-WISE VECTOR OPERATIONS
print("ELEMENT-WISE VECTOR OPERATIONS")
a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
print(f"Binary operators work element wise: {a + b}") # 1 + (-1)    2 + (-2)    3 + 3   4 + 4 = 0 0 6 8

#try a mismatched vector operation
print("mismatched vector operation")
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)
    
print()

# SLACAR VECTOR OPERATIONS
print("SCALAR VECTOR OPERATIONS")
a = np.array([1, 2, 3, 4])

# multiply a by a scalar
print("multply a by a scalar")
b = 5 * a 
print(f"b = 5 * a : {b}")

print()
# VECTOR DOT PRODUCT
print("VECTOR DOT PRODUCT")
def my_dot(a, b): 
    x=0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x

# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"my_dot(a, b) = {my_dot(a, b)}")


# test 1-D
print("Let's try the same operators using np.dot")
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ") 
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")

print()

# NEED FOR SPEED: Vectors vs For-Loop
print("NEED FOR SPEED: Vectors vs For-Loops")
np.random.seed(1)
a = np.random.rand(10000000)  # very large arrays
b = np.random.rand(10000000)

tic = time.time()  # capture start time
c = np.dot(a, b)
toc = time.time()  # capture end time

print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

tic = time.time()  # capture start time
c = my_dot(a,b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del(a);del(b)  #remove these big arrays from memory

print()
# VECTOR OPERATIONS IN COURSE 1
print("VECTOR OPERATIONS IN COURSE 1")
# show common Course 1 example
X = np.array([[1],[2],[3],[4]])
w = np.array([2])
c = np.dot(X[1], w)

print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")

print()

# MATRICES
print("MATRICES: 2-D arrays")

# matrix creation
print("matrix creation")
a = np.zeros((1, 5))                                       
print(f"a shape = {a.shape}, a = {a}")                     

a = np.zeros((2, 1))                                                                   
print(f"a shape = {a.shape}, a = {a}") 

a = np.random.random_sample((1, 1))  
print(f"a shape = {a.shape}, a = {a}") 

# NumPy routines which allocate memory and fill with user specified values
print("NumPy routines which allocate memory and fill with user specified values")
a = np.array([[5], [4], [3]]);   print(f" a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],   # One can also
              [4],   # separate values
              [3]]); #into separate rows
print(f" a shape = {a.shape}, np.array: a = {a}")

print()

# INDEXING MATRICES
print("INDEXING MATRICES")
#vector indexing operations on matrices
print("vector indexing operations on matrices")
a = np.arange(6).reshape(-1, 2)   #reshape is a convenient way to create matrices
print(f"a.shape: {a.shape}, \na= {a}")

#access an element
print("access an element")
print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

#access a row
print("access a row")
print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")

print()

# SLICING
print("SLICING")
#vector 2-D slicing operations
print("vector 2-D slicing operations")
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

#access 5 consecutive elements (start:stop:step)
print("access 5 consecutive elements (start:stop:step)")
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("access 5 consecutive elements (start:stop:step) in two rows")
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("access all elements")
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("access all elements in one row (very common usage)")
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("same as")
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")
