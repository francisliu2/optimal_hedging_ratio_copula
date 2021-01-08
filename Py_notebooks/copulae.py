import numpy as np
import scipy
from scipy import stats 
from scipy.stats import norm
from scipy import integrate
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
import scipy.linalg as la
from toolbox import *
class Gaussian:
    def __init__(self, rho, Law_RS, Law_RF):
        self.rho = rho         # Dependence Parameter
        self.Law_RS = Law_RS   # Marginal Distribution of Spot
        self.Law_RF = Law_RF   # Marginal Distribution of Future
        self.meta_Gaussian = stats.multivariate_normal([0,0], # Mean
                                                       [[1,rho], # COV
                                                        [rho,1]])

    def H(self, w, h, r_h): # a helper function to compute the input to F_RF
        A = self.Law_RS.ppf(w) - r_h
        if h!=0:
            B = h
        else:
            B = 0.00001
        return A/B
    
    def g(self, w, h, r_h):
        return self.Law_RF.cdf(self.H(w=w, h=h, r_h=r_h))
    
    def D1C(self, w, h, r_h):
        integrand = lambda u: self.meta_Gaussian.pdf([norm.ppf(w), u]) 
        part2 = 1/norm.pdf(norm.ppf(w))
        return integrate.quad(integrand, -np.infty, norm.ppf(self.g(w, h, r_h)))[0] * part2
    
    def F_RH(self, h, r_h):
        func = partial(self.D1C, h=h, r_h=r_h)
        I = integrate.quad(func, 0.0001, 0.999)
        return 1 - I[0]
    
    def c(self, u, v): # copula density
        part1 = self.meta_Gaussian.pdf([norm.ppf(u), norm.ppf(v)])
        part2 = norm.pdf(norm.ppf(u))* norm.pdf(norm.ppf(v))
        return part1/part2
    
    def f_RH(self, h, r_h):
        part1 = lambda u: self.c(u, self.g(w=u, h=h, r_h=r_h))
        part2 = lambda u: self.Law_RF.pdf(self.H(w=u, h=h, r_h=r_h))
        integrand = lambda u: part1(u)*part2(u)
        return integrate.quad(integrand, 0.0001, 0.999)[0]/np.abs(h)
        
    def sample(self, n):
        copula_samples = self.meta_Gaussian.rvs(n)
        samples = np.zeros((n,2))
        samples[:,0]=self.Law_RS.ppf(norm.cdf(copula_samples[:,0]))
        samples[:,1]=self.Law_RF.ppf(norm.cdf(copula_samples[:,1]))
        return samples
    
    # likelihood function
    def l_fn(self, rho, u, v):
        _meta_Gaussian = stats.multivariate_normal([0,0], 
                                                  [[1,rho], 
                                                   [rho,1]])
        
        z1 = norm.ppf(u)
        z2 = norm.ppf(v)
        
        part1 = []
        for i in range(len(z1)):
            part1.append(_meta_Gaussian.pdf([z1[i], z2[i]]))
        
        part1 = np.array(part1)
        part2 = norm.pdf(norm.ppf(u))*norm.pdf(norm.ppf(v))
        return np.nanmean(np.log(part1/part2))

    def canonical_calibrate(self, u, v):
        fn_toopt = lambda rho: -self.l_fn(rho, u, v)
        result = scipy.optimize.minimize(fn_toopt, x0=0, tol=1e-10)
        self.rho = result.x
        return result

class t_Copula:
    def __init__(self, rho, nu, Law_RS, Law_RF):
        self.rho        = rho      # Dependence Parameter
        self.nu         = nu       # Degree of Freedom 
        self.Law_RS     = Law_RS   # Marginal Distribution of Spot
        self.Law_RF     = Law_RF   # Marginal Distribution of Future
        self.meta_t     = multivariate_t(nu =nu,  # DF
                                         Sigma=np.array([[1,rho], # COV
                                            [rho,1]]))
        self.t1 = stats.t(df=nu) # inner
        self.t2 = stats.t(df=nu) 
        
    def H(self, w, h, r_h): # a helper function to compute the input to F_RF
        A = self.Law_RS.ppf(w) - r_h
        if h!=0:
            B = h
        else:
            B = 0.00001
        return A/B
    
    def g(self, w, h, r_h):
        return self.Law_RF.cdf(self.H(w=w, h=h, r_h=r_h))
    
    def D1C(self, w, h, r_h):
        integrand = lambda u: self.meta_t.pdf(self.t1.ppf(w), u) 
        part2 = 1/self.t1.pdf(self.t1.ppf(w))
        return integrate.quad(integrand, -np.infty, self.t2.ppf(self.g(w, h, r_h)))[0] * part2
    
    def F_RH(self, h, r_h):
        func = partial(self.D1C, h=h, r_h=r_h)
        I = integrate.quad(func, 0.0001, 0.999)
        return 1 - I[0]
    
    def c(self, u, v): # copula density
        part1 = self.meta_t.pdf(self.t1.ppf(u), self.t2.ppf(v))
        part2 = self.t1.pdf(self.t1.ppf(u))* self.t2.pdf(self.t2.ppf(v))
        return part1/part2
    
    def f_RH(self, h, r_h):
        part1 = lambda u: self.c(u, self.g(w=u, h=h, r_h=r_h))
        part2 = lambda u: self.Law_RF.pdf(self.H(w=u, h=h, r_h=r_h))
        integrand = lambda u: part1(u)*part2(u)
        return integrate.quad(integrand, 0.0001, 0.999)[0]/np.abs(h)
        
    def sample(self, n):
        copula_samples = self.meta_t.rvs(n)
        samples = np.zeros((n,2))
        samples[:,0]=self.Law_RS.ppf(self.t1.cdf(copula_samples[:,0]))
        samples[:,1]=self.Law_RF.ppf(self.t2.cdf(copula_samples[:,1]))
        return samples
    
        # likelihood function
    def l_fn(self, nu, rho, u, v):
        if (np.abs(rho)>=1) or (nu <=0):
            return -5000
        
        _meta_t = multivariate_t(nu=nu,  # DF
                                     Sigma=np.array([[1,rho], # COV
                                                     [rho,1]]))
        
        _t1 = stats.t(df=nu)
        _t2 = stats.t(df=nu)
        
        z1 = _t1.ppf(u) # inner
        z2 = _t2.ppf(v)
        
        part1 = []
        for i in range(len(z1)):
            part1.append(_meta_t.pdf(z1[i], z2[i]))
        
        part1 = np.array(part1)
        part2 = _t1.pdf(_t1.ppf(u))*_t2.pdf(_t2.ppf(v))
        return np.nanmean(np.log(part1/part2))

    def canonical_calibrate(self, u, v):
        fn_toopt = lambda theta: -self.l_fn(theta[0],theta[1] , u, v)
        result = scipy.optimize.fmin(fn_toopt, x0=(0.01,0.8), 
                             xtol=1e-10, 
                             maxiter=5000,
                             maxfun=400)
        self.nu  = result[0]
        self.rho = result[1]
        return result

# Archimedean
class Clayton:
    def __init__(self, theta, Law_RS, Law_RF):
        self.theta = theta     # Dependence Parameter
        self.Law_RS = Law_RS   # Marginal Distribution of Spot
        self.Law_RF = Law_RF   # Marginal Distribution of Future
        
    def phi(self, t):
        A = 1/self.theta
        B = t**(-self.theta)-1
        return A*B
    
    def phi_inverse(self, t):
        A = (1+self.theta*t)
        B = -1/self.theta
        return A**B
    
    def d_phi(self, t):
        return -t**(-self.theta-1)
    
    def d_phi_inverse(self, t):
        A = 1+ self.theta*t
        B = (-1/self.theta) - 1
        return -1*A**B
    
    def H(self, w, h, r_h): # a helper function to compute the input to F_RF
        A = self.Law_RS.ppf(w) - r_h
        if h!=0:
            B = h
        else:
            B = 0.00001
        return A/B
    
    def g(self, w, h, r_h):
        return self.Law_RF.cdf(self.H(w=w, h=h, r_h=r_h))
    
    def D1C(self, w, h, r_h):
        a = self.phi(w) + self.phi(self.g(w,h,r_h))
        A = self.d_phi_inverse(a)
        B = self.d_phi(w)
        return A*B
        
    def F_RH(self, h, r_h):
        func = partial(self.D1C, h=h, r_h=r_h)
        I = integrate.quad(func, 0.0001, 0.999)
        return 1 - I[0]
    
    def copula_fn(self, u, v):
        return self.phi_inverse(self.phi(u)+self.phi(v))
    
    def sample(self, n):
        u1 = stats.uniform().rvs(n)
        v2 = stats.uniform().rvs(n)
        u2 = (u1**(-self.theta)*(v2**(-self.theta/(1+self.theta))-1)+1)**(-1/self.theta)
        samples = np.zeros((n,2))
        samples[:,0]=self.Law_RS.ppf(u1)
        samples[:,1]=self.Law_RF.ppf(u2)
        return samples
    
    def c(self, u, v): # copula density
        part1 = (1+self.theta) * (u*v)**(-1-self.theta)
        part2 = (-1 + u**(-self.theta) + v**(-self.theta))**(-2-(1/self.theta))
        return part1*part2
    
    def l_fn(self, theta, u, v): # log dependency likelihood 
        part1 = (1+theta) * (u*v)**(-1-theta)
        part2 = (-1 + u**(-theta) + v**(-theta))**(-2-(1/theta))
        return np.mean(np.log(part1*part2))
    
    def canonical_calibrate(self, u, v):
        fn_toopt = lambda theta: -self.l_fn(theta, u, v)
        result = scipy.optimize.fmin(fn_toopt, x0=5, 
                             xtol=1e-10, 
                             maxiter=5000,
                             maxfun=400)
        self.theta = result[0]
        return result
    
class Frank:
    def __init__(self, theta, Law_RS, Law_RF):
        self.theta = theta     # Dependence Parameter
        self.Law_RS = Law_RS   # Marginal Distribution of Spot
        self.Law_RF = Law_RF   # Marginal Distribution of Future
        
    def phi(self, t):
        A = np.exp(-self.theta * t) - 1
        B = np.exp(-self.theta)     - 1
        return -np.log(A/B)
    
    def phi_inverse(self, t):
        A = -1/self.theta
        B = 1 + (np.exp(-t)*(np.exp(-self.theta)-1))
        return A*np.log(B)
    
    def d_phi(self, t):
        A = self.theta * np.exp(-self.theta * t)
        B = np.exp(-self.theta * t) - 1
        return A/B
    
    def d_phi_inverse(self, t):
        A = 1/self.theta
        B = np.exp(-t) * (np.exp(-self.theta)-1)
        C = 1+B
        return A*B/C
    
    def H(self, w, h, r_h): # a helper function to compute the input to F_RF
        A = self.Law_RS.ppf(w) - r_h
        if h!=0:
            B = h
        else:
            B = 0.00001
        return A/B
    
    def g(self, w, h, r_h):
        return self.Law_RF.cdf(self.H(w=w, h=h, r_h=r_h))
    
    def D1C(self, w, h, r_h):
        a = self.phi(w) + self.phi(self.g(w,h,r_h))
        A = self.d_phi_inverse(a)
        B = self.d_phi(w)
        return A*B
        
    def F_RH(self, h, r_h):
        func = partial(self.D1C, h=h, r_h=r_h)
        I = integrate.quad(func, 0.0001, 0.999)
        return 1 - I[0]
    
    def copula_fn(self, u, v):
        return self.phi_inverse(self.phi(u)+self.phi(v))
    
    def c(self, u, v): # copula density (wiki missed the negative sign in front of theta)
        part1 = -self.theta*np.exp(-self.theta*(u+v))*(np.exp(-self.theta)-1)
        part2 = np.exp(-self.theta) - np.exp(-self.theta*u) - np.exp(-self.theta*v) + np.exp(-self.theta*(u+v))
        return part1 /(part2**2)
    
    def l_fn(self, theta, u, v): # log dependency likelihood 
        part1 = -theta*np.exp(-theta*(u+v))*(np.exp(-theta)-1)
        part2 = np.exp(-theta) - np.exp(-theta*u) - np.exp(-theta*v) + np.exp(-theta*(u+v))
        return np.mean(np.log(part1 /(part2**2)))
    
    def canonical_calibrate(self, u, v):
        fn_toopt = lambda theta: -self.l_fn(theta, u, v)
        result = scipy.optimize.fmin(fn_toopt, x0=5, 
                         xtol=1e-10, 
                         maxiter=5000,
                         maxfun=400)
        self.theta = result[0]
        return result
    
    def tau(theta): # Statistical modeling of joint probability distribution using copula: Application to peak and permanent displacement seismic demands
        part1 = 1 - 4/theta 
        part2 = 4/theta**2
        part3_fn = lambda t: t/(np.exp(t)-1)
        part3 = scipy.integrate.quad(part3_fn, 0, theta)[0]
        return part1+part2*part3
    
if __name__ == "__main__":
    Law_RS=stats.norm
    Law_RF=stats.norm
    c = Gaussian(rho = 0.7,
                Law_RS = Law_RS,
                Law_RF = Law_RF)

    samples = c.sample(1000)
    plt.figure(figsize=(10,10))
    plt.scatter(x=samples[:,0], y=samples[:,1])
    plt.xlabel(r"$r^S$")
    plt.ylabel(r"$r^F$")
    plt.savefig("images/gaussiancopula.png", transparent=True)
    print("Please find a plot of Gaussian copula sample in images/gaussiancopula.png")