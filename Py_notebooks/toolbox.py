import numpy as np
from scipy import stats
import scipy.linalg as la
from functools import partial, lru_cache
from scipy import integrate
import scipy
import dill
dill.settings['recurse'] = True
fs = dill.load(open("NIG_k3-k5_fn", 'rb'))

class multivariate_t:
    def __init__(self, nu, Sigma):
        self.nu = nu
        self.Sigma = Sigma
        self.d = self.Sigma.shape[0] 
        if self.d !=  Sigma.shape[1]:
            return print("Sigma must be a square matrix")
        
    def pdf(self, *argv):
        # mean = \vec{0}
        x = np.array([i for i in argv])

        if len(x) != self.d:
            return print("Dimension not correct")

        try:
            s = la.inv(self.Sigma)[0,1]
        except:
            print("Sigma is not invertable")

        part1 = scipy.special.gamma((self.nu+self.d)/2)
        part2 = scipy.special.gamma(self.nu/2)*np.sqrt(((np.pi*self.nu)**self.d)*la.det(self.Sigma))
        part3 = (1 + (x.dot(la.inv(self.Sigma)).dot(x))/self.nu)**(-(self.nu+self.d)/2)
        return (part1/part2)*part3

    def cdf(self, upper):
        uppers = [[-np.infty, upper[i]] for i in range(len(upper))]
        fn = partial(self.pdf)
        return integrate.nquad(fn, uppers)[0]

    def rvs(self, size): # Sample 
        Z_law = stats.multivariate_normal(np.zeros(self.d), # Mean
           np.eye(self.d))

        A = la.cholesky(self.Sigma, lower=True)
        W = self.nu/stats.chi2(self.nu).rvs(size)
        Z = Z_law.rvs(size)
        X = (A.dot(Z.T))
        sqrt_W = np.sqrt(W)
        for i in range(self.d):
            X[i] = X[i]*sqrt_W
        return X.T
    
# correct as per paper Normal Inverse Gaussian Distributions and Stochastic Volatility Modelling
class invgauss:
    def __init__(self, delta, gamma):
        self.mu = delta/gamma
        self._lambda = self.mu**3 * gamma**3 / delta

    def rvs(self, size):
        nu = stats.norm.rvs(size=size)
        y = nu**2

        x = self.mu + self.mu**2*y/(2*self._lambda) - (self.mu/(2*self._lambda))*np.sqrt(4*self.mu*self._lambda*y + self.mu**2*y**2)
        z = stats.uniform(0,1).rvs(size)

        i = z <= self.mu/(self.mu+x)
        IGs = self.mu**2/x
        IGs[i] = x[i]
        return IGs

    def pdf(self, x):
        part1 = np.sqrt(self._lambda/(2*np.pi*x**3))
        part2 = np.exp(-self._lambda*(x-self.mu)**2/(2*(self.mu**2)*x))
        return part1*part2    
    
class norminvgauss:
    def __init__(self, alpha, beta, mu, delta, transformation=None):
        if transformation == None:
            self.alpha = alpha
            self.beta = beta
            self.mu = mu
            self.delta = delta
        else:
            # transofmration: aX + b
            self.a = transformation["a"]
            self.b = transformation["b"]
            self.alpha = alpha/np.abs(self.a)
            self.beta = beta/self.a
            self.mu = self.a*mu + self.b
            self.delta = delta*np.abs(self.a)
            
        self.gamma = np.sqrt(self.alpha**2 - self.beta**2)

    def pdf(self, x):
        part1 = self.alpha*self.delta
        part2 = scipy.special.kv(1.0, self.alpha*np.sqrt(self.delta**2 + (x-self.mu)**2))
        part3 = np.pi * np.sqrt(self.delta**2 + (x-self.mu)**2)
        part4 = np.exp(self.delta*self.gamma + self.beta*(x-self.mu))
        return part1*part2*part4/part3
    
    def cdf(self, y):
        return scipy.integrate.quad(self.pdf, -np.inf, y)[0]
    
    def mean(self):
        # Analytical 
        return self.mu+self.delta*self.beta/self.gamma
    
    def var(self):
        return (self.delta*self.alpha**2/self.gamma**3)
    
    def std(self):
        return np.sqrt(self.var())
    
    def skewness(self):
        return 3*self.beta / (self.alpha*np.sqrt(self.delta*self.gamma))
    
    def kurtosis(self):
        return 3*(1+4*self.beta**2/self.alpha**2)/(self.delta*self.gamma)
    
    def normalise(self):
        # Standardised NIG for CF approximation use
        fs = dill.load(open("NIG_k3-k5_fn", 'rb'))
        k3_fn = fs['k3']
        k4_fn = fs['k4']
        k5_fn = fs['k5']
        
        # normalise
        self.a = self.std()
        self.b = self.mean()
        self.standardisedNIG = norminvgauss(alpha=self.alpha*self.a,
                                          beta=self.beta*self.a,
                                          mu=(self.mu-self.b)/self.a,
                                          delta=self.delta/self.a)
        
        salpha = self.alpha*self.a
        sbeta  = self.beta*self.a
        smu    = (self.mu-self.b)/self.a
        sdelta = self.delta/self.a
        
        # Cumulants of the standardised NIG distribtion for later approximation use
        self._k3 = k3_fn(salpha,sbeta,smu,sdelta)
        self._k4 = k4_fn(salpha,sbeta,smu,sdelta)
        self._k5 = k5_fn(salpha,sbeta,smu,sdelta)


    def ppf_approx(self, Zq):
        self.normalise()
        # level 0
        part1 =  Zq

        # level 1
        part2 =  self._k3*(Zq**2 - 1)/6

        # level 2
        part3 =  self._k4*(Zq**3 - 3*Zq)/24
        part4 = -self._k3**2*(2*Zq**3 - 5*Zq)/36

        # level 3
        part5 =  self._k5*(Zq**4 - 6*Zq**2 + 3)/120
        part6 = -self._k3*self._k4*(Zq**4-5*Zq**2+2)/24
        part7 =  self._k3**3*(12*Zq**4 - 53*Zq**2+17)/324

        Xq = self.a*(part1 + part2 + part3 + part4 + part5 + part6 + part7) + self.b
        return Xq
    
    def ppf_sampling_approx(self, q_arr, size=5000000):
        NIG = self.rvs(size)
        q_sample = np.quantile(NIG, q_arr)
        return q_sample
    
    def rvs(self, size):
        z = invgauss(delta=self.delta, gamma=self.gamma).rvs(size=size)
        x = stats.norm(loc=self.mu + self.beta*z, scale= np.sqrt(z)).rvs(size=size)
        return x
      
    def ppf(self, q):
        fn_toopt = lambda x: (self.cdf(x) - q)**2
        result  = scipy.optimize.minimize(fn_toopt, x0=self.mean(), tol=1e-10)
        return result.x
    
    def MGF(self, z):
        part1 = self.mu*z 
        part2 = self.delta*(self.gamma-np.sqrt(self.alpha**2-(self.beta+z)**2))
        return np.exp(part1+part2)
    
    def CGF(self, z): # Cumulants Generating Function 
        return np.log(MGF(z))

    def CF(self, z): # Characteristic Function
        part1 = 1j*self.mu*z 
        part2 = self.delta*(self.gamma-np.sqrt(self.alpha**2-(self.beta+1j*z)**2))
        return np.exp(part1+part2)
    
def empirical_lambda(u_arr, v_arr, q):
    if q <=0.5:
        return np.mean( ((u_arr <= q) & (v_arr <= q))/q)
    else:
        return np.mean( ((u_arr > q) & (v_arr > q))/(1-q) )
