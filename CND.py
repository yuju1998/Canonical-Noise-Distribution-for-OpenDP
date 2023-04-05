import math
import numpy as np
from scipy.stats import binom, norm
#from Tulap import ptulap
import scipy
from statistics import median
import matplotlib.pyplot as plt

### Make tradeoff from GDP
def f(alpha):
    tradeoff = norm.cdf(norm.ppf(1 - alpha) - 1)
    return tradeoff
### test for tradeoff function
vals = list(range(-10, 10+1, 1))
f_vec = [0]*len(vals)
med = math.ceil(median(range(1, len(vals)+1)))-1
half = int((len(vals)-1)/2)
f_vec[med] = 1/2
for i in range(1, half+1):
    f_vec[med+i] = 1-f(f_vec[med+i-1])
    f_vec[med-i] = f(1-f_vec[med-i+1])
# plt.plot(vals, f_vec, marker='o', markersize=2)        
# plot x and y using default line style and color
# plt.show()
# plt.close()

def qCND(u, f, c):         # CND quantile function for f
    if u < c:
        return qCND(1 - f(u), f, c) - 1
    elif c <= u <= 1-c:
        return (u - 1/2)/(1 - 2*c)
    else:
        return qCND(f(1-u),f ,c) + 1    
    
def pCND(x, f, c):          # CND CDF for f
    np.random.seed(100)
    if x < -1/2:
        return f(1 - pCND(x+1, f, c))
    elif -1/2 <= x and x <= 1/2:
        return c * (1/2 - x) + (1 - c) * (x + 1/2)
    else:
        return 1 - f(pCND(x-1, f, c))
    
def rCND(n, f, c):
    unifs = np.random.rand(n)
    samples = np.zeros(n)
    for i in range(len(unifs)):
        samples[i] = qCND(unifs[i], f, c)
    return samples


def dCND(x, f, F, c, f_deriv=None):
    np.random.seed(100)
    
    if f_deriv == None:
        def f_deriv(alpha, f=f, h=1e-6, *args, **kwargs):
            return (f(alpha + h, *args, **kwargs) - f(alpha - h, *args, **kwargs)) / (2 * h)
    
    if x < -1/2:
        return dCND(-x, f, F, c)
    elif -1/2 <= x and x <= 1/2:
        return 1 - 2 * c
    else:
        return -f_deriv(alpha=F(x-1)) * dCND(x-1, f, F, c)

