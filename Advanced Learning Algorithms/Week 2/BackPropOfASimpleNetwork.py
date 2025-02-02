from sympy import *
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button

# Forward Prop
# Inputs and parameters
x = 2
w = -2
b = 8
y = 1
# calculate per step values   
c = w * x
a = c + b
d = a - y
J = d**2/2
print(f"J={J}, d={d}, a={a}, c={c}")

# Find dj/da arithmetically
d_epsilon = d + 0.001
J_epsilon = d_epsilon**2/2
k = (J_epsilon - J)/0.001   # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dd ~= k = {k} ")

# Symbolically
sx,sw,sb,sy,sJ = symbols('x,w,b,y,J')
sa, sc, sd = symbols('a,c,d')
sJ = sd**2/2
print(sJ)
print(sJ.subs([(sd,d)]))
dJ_dd = diff(sJ, sd)
print(dJ_dd)

# Find dd/da arithmetically
a_epsilon = a + 0.001         # a  plus a small value
d_epsilon = a_epsilon - y
k = (d_epsilon - d)/0.001   # difference divided by epsilon
print(f"d = {d}, d_epsilon = {d_epsilon}, dd_da ~= k = {k} ")

# Symbolically
sd = sa - sy
print(sd)

dd_da = diff(sd,sa)
print(dd_da)

# Find dj/da
dJ_da = dd_da * dJ_dd
print(dJ_da)

a_epsilon = a + 0.001
d_epsilon = a_epsilon - y
J_epsilon = d_epsilon**2/2
k = (J_epsilon - J)/0.001   
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_da ~= k = {k} ")

# calculate the local derivatives da_dc, da_db
sa = sc + sb
sa

da_dc = diff(sa,sc)
da_db = diff(sa,sb)
print(da_dc, da_db)

dJ_dc = da_dc * dJ_da
dJ_db = da_db * dJ_da
print(f"dJ_dc = {dJ_dc},  dJ_db = {dJ_db}")

# calculate the local derivative
sc = sw * sx
sc

dc_dw = diff(sc,sw)
dc_dw

dJ_dw = dc_dw * dJ_dc
dJ_dw

print(f"dJ_dw = {dJ_dw.subs([(sd,d),(sx,x)])}")

J_epsilon = ((w+0.001)*x+b - y)**2/2
k = (J_epsilon - J)/0.001  
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
