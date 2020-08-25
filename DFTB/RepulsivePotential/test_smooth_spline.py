from matplotlib.pyplot import *
from numpy import linspace,exp
from numpy.random import randn
from scipy.interpolate import LSQUnivariateSpline
x = linspace(-3,3,100)
y = exp(-x**2) + randn(100)/2.0
t = [-1,0,1]
s = LSQUnivariateSpline(x,y,t)
xs = linspace(-3,3,1000)
ys = s(xs)

plot(x,y)
plot(xs,ys,ls="-.")
show()
