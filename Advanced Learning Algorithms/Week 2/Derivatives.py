from sympy import symbols, diff

J = (3)**2
J_epsilon = (3 + 0.001)**2
k = (J_epsilon - J)/0.001    # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k:0.6f} ")

# We have increased the input value a little bit (0.001), causing the output to change from 9 to 9.006001, an increase of 6 times the input 
# increase. Referencing (1) above, this says that 𝑘=6, so ∂𝐽(𝑤)∂𝑤≈6. If you are familiar with calculus, you know, 
# written symbolically, ∂𝐽(𝑤)∂𝑤=2𝑤. With 𝑤=3 this is 6. Our calculation above is not exactly 6 because to be exactly correct 𝜖 
# would need to be infinitesimally small or really, really small. That is why we use the symbols ≈ or ~= rather than =. 
# Let's see what happens if we make 𝜖 smaller.

J = (3)**2
J_epsilon = (3 + 0.000000001)**2
k = (J_epsilon - J)/0.000000001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")


# J = w^2

J, w = symbols('J, w')

J=w**2
print(J)

dJ_dw = diff(J,w)
print(dJ_dw)

print(dJ_dw.subs([(w,2)]))    # derivative at the point w = 2

print(dJ_dw.subs([(w,3)]))    # derivative at the point w = 3

print(dJ_dw.subs([(w,-3)]))    # derivative at the point w = -3

# J = 2w

w, J = symbols('w, J')

J = 2 * w
print(J)

dJ_dw = diff(J,w)
print(dJ_dw)

print(dJ_dw.subs([(w,-3)]))    # derivative at the point w = -3

J = 2*3
J_epsilon = 2*(3 + 0.001)
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")

# For the function 𝐽=2𝑤, it is easy to see that any change in 𝑤 will result in 2 times that amount of change in the output 𝐽, 
# regardless of the starting value of 𝑤. Our NumPy and arithmetic results confirm this

# J = w^3

J, w = symbols('J, w')

J=w**3
print(J)

dJ_dw = diff(J,w)
print(dJ_dw)

print(dJ_dw.subs([(w,2)]))   # derivative at the point w=2

J = (2)**3
J_epsilon = (2+0.001)**3
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")

# J = 1 / w 

J, w = symbols('J, w')

J= 1/w
print(J)

dJ_dw = diff(J,w)
print(dJ_dw)

print(dJ_dw.subs([(w,2)]))

J = 1/2
J_epsilon = 1/(2+0.001)
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")

# J = 1 / w^2

J, w = symbols('J, w')

######### Fill in yourself:

J = 1/w**2
print(J)

dJ_dw = diff(J,w)
print(dJ_dw)

print(dJ_dw.subs([(w,4)])) # w = 4

J = 1/4**2
J_epsilon = 1/(4+0.001)**2
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
