import random
import math
import numpy as np
from scipy.stats import binom, norm
import scipy
from statistics import median
import matplotlib.pyplot as plt
from CND import qCND, pCND, rCND, dCND

def f(alpha):       # tradeoff function
    tradeoff = norm.cdf(norm.ppf(1 - alpha) - 1)
    return tradeoff

def numerical_deriv(x, f, h=1e-6, *args, **kwargs):
    return (f(x + h, *args, **kwargs) - f(x - h, *args, **kwargs)) / (2 * h)

def f_deriv(alpha):
    f_prime_alpha = -norm.pdf(norm.ppf(1-alpha)-1)/norm.pdf(norm.ppf(1-alpha))
    return f_prime_alpha

    
n = 1000
data = np.linspace(-3, 3, n)
cdf = []
np.random.seed(100)
for xi in data:
    cdf_value = pCND(x=xi, f=f, c=norm.cdf(-1/2))
    # print(cdf_value)
    cdf.append(cdf_value)
#define x and y values to use for CDF
#plot normal CDF
plt.plot(data, cdf, color='red')
plt.title('CND-CDF')
plt.xlabel('x')
plt.ylabel('CDF')
plt.savefig('CND_cdf.png')
plt.close()


pdf = []
np.random.seed(100)
for xi in data:
    pdf_value = dCND(x=xi, f=f, F=norm.cdf, c=norm.cdf(-1/2), f_deriv=f_deriv)
    # print(pdf_value)
    pdf.append(pdf_value)
#define x and y values to use for PDF
#plot normal PDF
plt.plot(data, pdf, color='red')
plt.title('CND-PDF')
plt.xlabel('x')
plt.ylabel('PDF')
plt.savefig('CND_pdf.png')
plt.close()
    

uniform_data = np.linspace(0, 1, n, endpoint=False)[1:]
quantile = []
np.random.seed(100)
for ui in uniform_data:
    # print(ui)
    quantile_value = qCND(u=ui, f=f, c=norm.cdf(-1/2))
    # print("quantile_value:", quantile_value)
    quantile.append(quantile_value)
plt.plot(uniform_data, quantile, color='red')
plt.title('CND-Quantile')
plt.xlabel('quantile')
plt.savefig('CND_quantile.png')
plt.close()


samples = rCND(n=n, f=f, c=norm.cdf(-1/2))
# print(samples)
plt.hist(samples, bins=50)
plt.savefig('CND_randsamples.png')
plt.close()

