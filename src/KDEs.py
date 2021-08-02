import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.stats import norm
import pandas as pd
import numpy as np
import seaborn as sns
# from statsmodels.distributions.empirical_distribution import ECDF
from toolbox import *
import random
np.random.seed(0)


def K_Uniform(u):
    u = np.abs(u)
    return 0.5*(u <= 1)

def K_Triangle(u):
    u = np.abs(u)
    return (1-u)*(u <= 1)

def K_Epanechnikov(u):
    return (3/4)*(1-u**2)*(np.abs(u)<=1)

def K_Gaussian(u):
    return np.exp(-0.5*u**2)/np.sqrt(2*np.pi)

def sample_Uniform(size, data, h):
    datasample = random.choices(data,k=size)
    u1 = stats.uniform().rvs(size)
    kernelsample = u1*h
    return datasample + kernelsample 

def sample_Triangle(size, data, h):
    datasample = random.choices(data,k=size)
    u1 = stats.uniform().rvs(size)
    u2 = stats.uniform().rvs(size)
    kernelsample = (u1+u2-1) *h
    return datasample + kernelsample 

def sample_Epanechnikov(size, data, h):
    datasample = random.choices(data,k=size)
    u1 = stats.uniform().rvs(size)*2-1
    u2 = stats.uniform().rvs(size)*2-1
    u3 = stats.uniform().rvs(size)*2-1
    kernelsample = u3[:]
    i = ( np.abs(u3) >= np.abs(u2) ) & ( np.abs(u3) >= np.abs(u1) )
    kernelsample[i] = u2[i]
    kernelsample = kernelsample*h
    return datasample + kernelsample 

def sample_Gaussian(size, data, h):
    datasample = random.choices(data,k=size)
    kernelsample = stats.norm().rvs(size)*h
    return datasample + kernelsample 

class KDE():
    def __init__(self, data, kernel_name, bw=None):
        uq               = np.quantile(data, .75)
        lq               = np.quantile(data, .25)
        if bw==None:
            self.h_brot      = 1.06*min(np.std(data), (uq-lq)/1.34)*len(data)**(-1/5)
        else:
            self.h_brot      = bw

        self.kernel_name = kernel_name
        self.data        = data
        self.ecdf        = ECDF(data)
        
        if self.kernel_name == "Uniform":
            self.kernel = K_uniform
        
        elif self.kernel_name == "Triangle":
            self.kernel = K_Triangle
        
        elif self.kernel_name == "Epanechnikov":
            self.kernel = K_Epanechnikov
        
        elif self.kernel_name == "Gaussian":
            self.kernel = K_Gaussian
            
        self.samples = self.rvs(len(data)*100)
        
    def cdf(self, x): # KDE's cdf should be a cdf of a mixture...
        return self.ecdf(x)
    
    def pdf(self, x):
        u = lambda x: (x - self.data)/self.h_brot
        s = np.sum(self.kernel(u(x)))
        return s/(len(self.data)*self.h_brot)
#     def pdf(self, x):
#         s = np.sum(self.kernel((x - self.data)/self.h_brot))
#         return s/(len(self.data)*self.h_brot)
    
    def rvs(self, size):
        if self.kernel_name == "Uniform":
            return sample_Uniform(size, self.data,self.h_brot)
        
        elif self.kernel_name == "Triangle":
            return sample_Triangle(size, self.data,self.h_brot)
        
        elif self.kernel_name == "Epanechnikov":
            return sample_Epanechnikov(size, self.data,self.h_brot)
        
        elif self.kernel_name == "Gaussian":
            return sample_Gaussian(size, self.data,self.h_brot)
        
    def ppf(self, q):
        return np.quantile(self.samples, q)
    
    def plot_density(self):
        a = min(self.data)-np.std(self.data)
        b = max(self.data)+np.std(self.data)

        x_arr = np.linspace(a,b,10000)

        result = []

        for x in x_arr:
            result.append(self.pdf(x))
        
        plt.plot(x_arr, result)
        plt.scatter(self.data,np.zeros(len(self.data)), marker='+')
        
        
        
        
        
        
        
        
        
        
        
        

    
    
    
