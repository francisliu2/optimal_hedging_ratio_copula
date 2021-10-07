import numpy as np
from scipy import stats
import scipy.linalg as la
from functools import partial, lru_cache
from scipy import integrate
from scipy.special import gamma
from statsmodels.distributions.empirical_distribution import ECDF as quickECDF
import scipy
import dill
np.random.seed(0)

dill.settings['recurse'] = True

from pathlib import Path

source_path = str(Path(__file__).resolve().parent)
fs = dill.load(open(source_path + "/NIG_k3-k5_fn", 'rb'))


class multivariate_t:
	def __init__(self, nu, Sigma):
		self.nu = nu
		self.Sigma = Sigma
		self.d = self.Sigma.shape[0]
		if self.d != Sigma.shape[1]:
			return print("Sigma must be a square matrix")
		self.MN = stats.multivariate_normal([0, 0],  # for cdf approximation use
                                            Sigma)

	def pdf(self, *argv):
		# mean = \vec{0}
		x = np.array([i for i in argv])

		if len(x) != self.d:
			return print("Dimension not correct")

		try:
			s = la.inv(self.Sigma)[0, 1]
		except:
			print("Sigma is not invertable")

		part1 = scipy.special.gamma((self.nu + self.d) / 2)
		part2 = scipy.special.gamma(self.nu / 2) * np.sqrt(((np.pi * self.nu) ** self.d) * la.det(self.Sigma))
		part3 = (1 + (x.dot(la.inv(self.Sigma)).dot(x)) / self.nu) ** (-(self.nu + self.d) / 2)
		return (part1 / part2) * part3

	def cdf_ref(self, upper):  # reference computation of cdf
		uppers = [[-np.infty, upper[i]] for i in range(len(upper))]
		fn = partial(self.pdf)
		return integrate.nquad(fn, uppers)[0]

	def cdf(self,
			b):  # cdf approximation by http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.554.9917&rep=rep1&type=pdf equation 2; The equation was originally from Tong (1990) eq. 9.4.1

		fn = lambda s: s ** (self.nu - 1) * np.exp(-s ** 2 / 2) * self.MN.cdf(s * b / np.sqrt(self.nu))
		return 2 ** (1 - (self.nu / 2)) / gamma(self.nu / 2) * scipy.integrate.quad(fn, 0, np.inf)[0]

	def rvs(self, size):  # Sample
		Z_law = stats.multivariate_normal(np.zeros(self.d),  # Mean
										  np.eye(self.d))

		A = la.cholesky(self.Sigma, lower=True)
		W = self.nu / stats.chi2(self.nu).rvs(size)
		Z = Z_law.rvs(size)
		X = (A.dot(Z.T))
		sqrt_W = np.sqrt(W)
		for i in range(self.d):
			X[i] = X[i] * sqrt_W
		return X.T


# correct as per paper Normal Inverse Gaussian Distributions and Stochastic Volatility Modelling
class invgauss:
	def __init__(self, delta, gamma):
		self.mu = delta / gamma
		self._lambda = self.mu ** 3 * gamma ** 3 / delta

	def rvs(self, size):
		nu = stats.norm.rvs(size=size)
		y = nu ** 2

		x = self.mu + self.mu ** 2 * y / (2 * self._lambda) - (self.mu / (2 * self._lambda)) * np.sqrt(
			4 * self.mu * self._lambda * y + self.mu ** 2 * y ** 2)
		z = stats.uniform(0, 1).rvs(size)

		i = z <= self.mu / (self.mu + x)
		IGs = self.mu ** 2 / x
		IGs[i] = x[i]
		return IGs

	def pdf(self, x):
		part1 = np.sqrt(self._lambda / (2 * np.pi * x ** 3))
		part2 = np.exp(-self._lambda * (x - self.mu) ** 2 / (2 * (self.mu ** 2) * x))
		return part1 * part2


class norminvgauss:
	def __init__(self, alpha, beta, mu, delta, transformation=None):
		if transformation == None:
			self.alpha = alpha
			self.beta = beta
			self.mu = mu
			self.delta = delta
		else:
			# transoformation: aX + b
			self.a = transformation["a"]
			self.b = transformation["b"]
			self.alpha = alpha / np.abs(self.a)
			self.beta = beta / self.a
			self.mu = self.a * mu + self.b
			self.delta = delta * np.abs(self.a)

		self.gamma = np.sqrt(self.alpha ** 2 - self.beta ** 2)

	def pdf(self, x):
		part1 = self.alpha * self.delta
		part2 = scipy.special.kv(1.0, self.alpha * np.sqrt(self.delta ** 2 + (x - self.mu) ** 2))
		part3 = np.pi * np.sqrt(self.delta ** 2 + (x - self.mu) ** 2)
		part4 = np.exp(self.delta * self.gamma + self.beta * (x - self.mu))
		return part1 * part2 * part4 / part3

	def cdf(self, y):
		return scipy.integrate.quad(self.pdf, -np.inf, y)[0]

	def mean(self):
		# Analytical
		return self.mu + self.delta * self.beta / self.gamma

	def var(self):
		return (self.delta * self.alpha ** 2 / self.gamma ** 3)

	def std(self):
		return np.sqrt(self.var())

	def skewness(self):
		return 3 * self.beta / (self.alpha * np.sqrt(self.delta * self.gamma))

	def kurtosis(self):
		return 3 * (1 + 4 * self.beta ** 2 / self.alpha ** 2) / (self.delta * self.gamma)

	def normalise(self):
		# Standardised NIG for CF approximation use
		fs = dill.load(open("NIG_k3-k5_fn", 'rb'))
		k3_fn = fs['k3']
		k4_fn = fs['k4']
		k5_fn = fs['k5']

		# normalise
		self.a = self.std()
		self.b = self.mean()
		self.standardisedNIG = norminvgauss(alpha=self.alpha * self.a,
											beta=self.beta * self.a,
											mu=(self.mu - self.b) / self.a,
											delta=self.delta / self.a)

		salpha = self.alpha * self.a
		sbeta = self.beta * self.a
		smu = (self.mu - self.b) / self.a
		sdelta = self.delta / self.a

		# Cumulants of the standardised NIG distribtion for later approximation use
		self._k3 = k3_fn(salpha, sbeta, smu, sdelta)
		self._k4 = k4_fn(salpha, sbeta, smu, sdelta)
		self._k5 = k5_fn(salpha, sbeta, smu, sdelta)

	def ppf_approx(self, Zq):
		self.normalise()
		# level 0
		part1 = Zq

		# level 1
		part2 = self._k3 * (Zq ** 2 - 1) / 6

		# level 2
		part3 = self._k4 * (Zq ** 3 - 3 * Zq) / 24
		part4 = -self._k3 ** 2 * (2 * Zq ** 3 - 5 * Zq) / 36

		# level 3
		part5 = self._k5 * (Zq ** 4 - 6 * Zq ** 2 + 3) / 120
		part6 = -self._k3 * self._k4 * (Zq ** 4 - 5 * Zq ** 2 + 2) / 24
		part7 = self._k3 ** 3 * (12 * Zq ** 4 - 53 * Zq ** 2 + 17) / 324

		Xq = self.a * (part1 + part2 + part3 + part4 + part5 + part6 + part7) + self.b
		return Xq

	def ppf_sampling_approx(self, q_arr, size=5000000):
		NIG = self.rvs(size)
		q_sample = np.quantile(NIG, q_arr)
		return q_sample

	def rvs(self, size):
		z = invgauss(delta=self.delta, gamma=self.gamma).rvs(size=size)
		x = stats.norm(loc=self.mu + self.beta * z, scale=np.sqrt(z)).rvs(size=size)
		return x

	def ppf(self, q):
		fn_toopt = lambda x: (self.cdf(x) - q) ** 2
		result = scipy.optimize.fmin(fn_toopt, x0=self.ppf_approx(q))
		return result

	def ppf_approx2(self, q_arr):
		# The main idea of this function is to leverage the numpy ability to to quick array operation.
		# First step: calculate the pdf using an array with equally spaced elements with values ranging from a min to a max.
		# Next step: calculate the pdf array to calculate percentage points using trapezoidal rule and nu.cumsum.
		# Third step: for each q in q_arr, search for the nearest percentage point and return the corresponding element in x_arr
		order = np.argsort(q_arr)
		q_arr = np.sort(q_arr)
		_max = self.ppf(1)
		_min = self.ppf(0)
		upper = np.ceil(_max)
		lower = np.ceil(_min)
		# create an array
		L = upper - lower
		d = 10 ** (-4)
		n = int(L / d)
		x_arr = np.linspace(lower, upper, n)[:, 0]

		integrand = self.pdf(x_arr)
		Probabilities = np.cumsum((integrand[:-1] + integrand[1:]) * d / 2)

		result = []

		# search for the corresponding x_arr as quantile
		m = 0
		for q in q_arr:
			k = np.sum(Probabilities[m:] <= q)
			result.append(x_arr[m + k])
			m += k
		result = np.array(result)
		result[q_arr == 0] = _min
		result[q_arr == 1] = _max
		return result[order]

	def MGF(self, z):
		part1 = self.mu * z
		part2 = self.delta * (self.gamma - np.sqrt(self.alpha ** 2 - (self.beta + z) ** 2))
		return np.exp(part1 + part2)

	def CGF(self, z):  # Cumulants Generating Function
		return np.log(MGF(z))

	def CF(self, z):  # Characteristic Function
		part1 = 1j * self.mu * z
		part2 = self.delta * (self.gamma - np.sqrt(self.alpha ** 2 - (self.beta + 1j * z) ** 2))
		return np.exp(part1 + part2)


def empirical_lambda(u_arr, v_arr, q):
	if q <= 0.5:
		return np.mean(((u_arr <= q) & (v_arr <= q)) / q)
	else:
		return np.mean(((u_arr > q) & (v_arr > q)) / (1 - q))


def ERM_weight(k, s):
	return k * np.exp(-k * s) / (1 - np.exp(-k))


class stable:
	def __init__(self, alpha, beta, mu, sigma):
		self.alpha = alpha
		self.beta = beta
		self.mu = mu
		self.sigma = sigma

	def log_phi(self, t):
		part1 = -1 * np.abs(self.sigma * t) ** self.alpha
		part3 = 1j * self.mu * t

		if self.alpha != 1:
			part2 = 1 - 1j * self.beta * np.sign(t) * np.tan(np.pi * self.alpha / 2)
		else:
			part2 = 1 + 1j * self.beta * np.sign(t) * 2 / np.pi * np.log(np.abs(t))
		return part1 * part2 + part3

	def pdf(self, x):
		fn = lambda t: np.exp(-1j * t * x) * np.exp(self.log_phi(t))
		return scipy.integrate.quad(fn, -np.inf, np.inf)[0] / (2 * np.pi)

	def rvs(self, size):

		U = stats.uniform.rvs(size=size) * 2 * np.pi - np.pi
		W = stats.expon.rvs(size=size)

		zi = -self.beta * np.tan(np.pi * self.alpha / 2)

		if self.alpha != 1:
			xi = 1 / self.alpha * np.arctan(-zi)
		else:
			xi = np.pi / 2

		if self.alpha != 1:
			part1 = (1 + zi ** 2) ** (0.5 / self.alpha)
			part2 = np.sin(self.alpha * (U + xi)) / (np.cos(U) ** (1 / self.alpha))
			part3 = (np.cos(U - self.alpha * (U + xi)) / W) ** ((1 - self.alpha) / self.alpha)
			X = part1 * part2 * part3
		else:
			part1 = ((np.pi / 2) + self.beta * U) * np.tan(U)
			part2a = (np.pi / 2) * W * np.cos(U)
			part2b = np.pi / 2 + self.beta * U
			part2 = self.beta * np.log(part2a / part2b)
			X = 1 / xi * (part1 - part2)

		if self.alpha == 1:
			Y = self.sigma * X + 2 / np.pi * self.beta * self.sigma * np.log(self.sigma) + self.mu
		else:
			Y = self.sigma * X + self.mu
		return Y


#         if removenan:
#             return Y[~np.isnan(Y)][:size/3]
#         else:
#             return Y[:size/3]

def Variance(rh):
	return np.var(rh)


def ERM_estimate_trapezoidal(k, rh):
	rh = np.sort(rh)
	s = quickECDF(rh)(rh)
	d = s[1:] - s[:-1]
	toint = ERM_weight(k, s) * rh
	return -np.sum((toint[:-1] + toint[1:]) * d) / 2


def ES(q, rh):
	b = np.quantile(rh, q, interpolation='nearest')
	return -np.mean(rh[rh <= b])


def VaR(q, rh):
	return -np.quantile(rh, q, interpolation='nearest')


def wrapper(rs, rf, h, risk_measure):
	rh = rs - h * rf
	return risk_measure(rh)


def optimize_h(C, k_arr, q_arr_ES, q_arr_VaR):
	np.random.seed(0)
	sample = C.sample(1000000)
	rs = sample[:, 0]
	rf = sample[:, 1]
	best_h = []

	fn = lambda h: wrapper(rs, rf, h, Variance)
	best_h.append(scipy.optimize.fmin(fn, 1)[0])

	for k in k_arr:
		fn = lambda h: wrapper(rs, rf, h, partial(ERM_estimate_trapezoidal, k))
		best_h.append(scipy.optimize.fmin(fn, 1)[0])

	for q in q_arr_ES:
		fn = lambda h: wrapper(rs, rf, h, partial(ES, q))
		best_h.append(scipy.optimize.fmin(fn, 1)[0])

	for q in q_arr_VaR:
		fn = lambda h: wrapper(rs, rf, h, partial(VaR, q))
		best_h.append(scipy.optimize.fmin(fn, 1)[0])
	return best_h


def risk_measures(k_arr, q_arr_ES, q_arr_VaR, rh):
	results = []
	results.append(Var(rh))

	for k in k_arr:
		results.append(ERM_estimate_trapezoidal(k, rh))

	for q in q_arr_ES:
		results.append(ES(q, rh))

	for q in q_arr_VaR:
		results.append(VaR(q, rh))

	return np.array(results)


def rh_PnL(rh):
	Mean = np.mean(rh)
	Std = np.std(rh)
	Max = np.max(rh)
	UQ = np.quantile(rh, 0.75)
	LQ = np.quantile(rh, 0.25)
	Min = np.min(rh)
	return Mean, Std, Max, UQ, LQ, Min


# def hedging_effectiveness(h_arr, spot, future, k_arr, q_arr):
# 	results = np.ones((len(h_arr), 1 + len(k_arr) + len(q_arr)))
# 	for i, h in enumerate(h_arr):
# 		rh = spot - h * future
# 		results[i, :] = 1 - risk_measures(k_arr, q_arr, rh) / risk_measures(k_arr, q_arr, spot)
# 	return np.array([results[i, i] for i in range(len(h_arr))])


def clip_h(h, _min, _max):
	# Usage columns = best_h_results_pd.columns
	# for c in columns:
	#      best_h_results_pd.loc[:,c] = best_h_results_pd.loc[:,c].apply(cap)
    if h < 0:
        return 0
    elif h >1:
        return 1
    else:
        return h

class ECDF():
	def __init__(self, data):
		self.data = data

	def __call__(self, x):
		return np.array([np.sum(self.data <= x[i]) for i in range(len(x))]) / (len(self.data) + 1)
