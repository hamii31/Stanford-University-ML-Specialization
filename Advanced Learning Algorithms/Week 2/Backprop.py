from sympy import *
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button

# Let's calculate the derivative of this slightly complex expression, 𝐽=(2+3𝑤)2. 
# We would like to find the derivative of 𝐽 with respect to 𝑤 or ∂𝐽∂𝑤.

# Forward prop
w = 3
a = 2+3*w
J = a**2
print(f"a = {a}, J = {J}")

# Backward prop
# Find dj/da arithmetically

a_epsilon = a + 0.001       # a epsilon
J_epsilon = a_epsilon**2    # J_epsilon
k = (J_epsilon - J)/0.001   # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_da ~= k = {k} ")

# Find dj/da symbolically
sw,sJ,sa = symbols('w,J,a')
sJ = sa**2
print(sJ)

print(sJ.subs([(sa,a)]))

dJ_da = diff(sJ, sa)
print(dJ_da)


# Find da/dw arithmetically
w_epsilon = w + 0.001       # a  plus a small value, epsilon
a_epsilon = 2 + 3*w_epsilon
k = (a_epsilon - a)/0.001   # difference divided by epsilon
print(f"a = {a}, a_epsilon = {a_epsilon}, da_dw ~= k = {k} ")

# Find da/dw symbolically
sa = 2 + 3*sw
print(sa)

da_dw = diff(sa,sw)
print(da_dw)

# The next step is the interesting part:

#     We know that a small change in 𝑤

        # will cause 𝑎 to change by 3 times that amount.
        # We know that a small change in 𝑎 will cause 𝐽 to change by 2×𝑎 times that amount. (a=11 in this example) 
        # We know that a small change in 𝑤 will cause 𝐽 to change by 3×2×𝑎 times that amount.

# These cascading changes go by the name of the chain rule. It can be written like this:
# ∂𝐽/∂𝑤=∂𝑎/∂𝑤*∂𝐽/∂𝑎

# Symbolical calculation of dj/dw
dJ_dw = da_dw * dJ_da
print(dJ_dw)

# Arithmetical calculation of dj/dw
w_epsilon = w + 0.001
a_epsilon = 2 + 3*w_epsilon
J_epsilon = a_epsilon**2
k = (J_epsilon - J)/0.001   # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
