import numpy as np
np.random.seed(0)

import pandas as pd

import scipy
from scipy import stats
from scipy.stats import norm
from scipy import integrate
from scipy.special import beta
import scipy.linalg as la
# from statsmodels.distributions.empirical_distribution import ECDF

import matplotlib.pyplot as plt
import seaborn as sns

from functools import partial
from tqdm import tqdm
from toolbox import *


class Copula(object):
	def __init__(self):
		self.samples = []

	def H(self, w, h, r_h):  # a helper function to compute the input to F_RF
		A = self.Law_RS.ppf(w) - r_h
		if h != 0:
			B = h
		else:
			B = 0.00001
		return A / B

	def g(self, w, h, r_h):
		a = self.Law_RF.cdf(self.H(w=w, h=h, r_h=r_h))
		if a >= 1: # to stablize the futher calulation
			return .999
		elif a <= 0:
			return .0001
		else:
			return a
	def F_RH(self, h, r_h):
		func = partial(self.D1C, h=h, r_h=r_h)
		I = integrate.quad(func, 0.0001, 0.999)
		return 1 - I[0]

	def f_RH(self, h, r_h):
		part1 = lambda u: self.c(u, self.g(w=u, h=h, r_h=r_h))
		part2 = lambda u: self.Law_RF.pdf(self.H(w=u, h=h, r_h=r_h))
		integrand = lambda u: part1(u) * part2(u)
		return integrate.quad(integrand, 0.0001, 0.999)[0] / np.abs(h)

	def _lambda(self, q):
		if q <= 0.5:
			return self.C(q, q) / q
		else:
			return (1 - (2 * q) + self.C(q, q)) / (1 - q)

class Gaussian(Copula):
	def __init__(self, paras, Law_RS, Law_RF):
		super().__init__()
		self.paras = paras
		self.rho = paras["rho"]  # Dependence Parameter
		self.Law_RS = Law_RS  # Marginal Distribution of Spot
		self.Law_RF = Law_RF  # Marginal Distribution of Future
		self.meta_Gaussian = stats.multivariate_normal([0, 0],  # Mean
													    [[1, self.rho],  # COV
														 [self.rho, 1]])

	def C(self, u, v):  # Copula Function
		return self.meta_Gaussian.cdf([norm.ppf(u), norm.ppf(v)])

	def c(self, u, v):  # copula density
		X1 = norm.ppf(u)
		X2 = norm.ppf(v)
		part1 = self.meta_Gaussian.pdf([X1, X2])
		part2 = norm.pdf(X1) * norm.pdf(X2)
		result = part1 / part2

		if np.isnan(result):
			return 0
		else:
			return result

	#         n = len(u)
	#         u[u==0] = 0.5/n
	#         v[v==0] = 0.5/n
	#         u[u==1] = 1-0.5/n
	#         v[v==1] = 1-0.5/n
	#         X1 = norm.ppf(u)
	#         X2 = norm.ppf(v)
	#         _c = 1/np.sqrt(1-self.rho**2)*np.exp(-(self.rho**2*(X1**2+X2**2)-2*self.rho*X1*X2)/(2-2*self.rho**2))
	#         return _c

	def D1C(self, w, h, r_h):
		integrand = lambda u: self.meta_Gaussian.pdf([norm.ppf(w), u])
		part2 = 1 / norm.pdf(norm.ppf(w))
		return integrate.quad(integrand, -np.infty, norm.ppf(self.g(w, h, r_h)))[0] * part2

	def l_fn(self, rho, u, v):
		if rho >= 1:
			return -5000
		_meta_Gaussian = stats.multivariate_normal([0, 0],
												   [[1, rho],
													[rho, 1]])

		z1 = norm.ppf(u)
		z2 = norm.ppf(v)

		part1 = []
		for i in range(len(z1)):
			part1.append(_meta_Gaussian.pdf([z1[i], z2[i]]))

		part1 = np.array(part1)
		part2 = norm.pdf(norm.ppf(u)) * norm.pdf(norm.ppf(v))
		return np.sum(np.log(part1 / part2))

	def dependency_likelihood(self, u, v):
		rho = self.rho
		return self.l_fn(rho, u, v)

	def canonical_calibrate(self, u, v):
		fn_toopt = lambda rho: -self.l_fn(rho, u, v)
		result = scipy.optimize.fmin(fn_toopt, x0=self.rho,
									 xtol=1e-10,
									 maxiter=5000,
									 maxfun=400)
		self.paras = {"rho": result[0]}
		self.rho = result[0]
		self.meta_Gaussian = stats.multivariate_normal([0, 0],  # Mean
													   [[1, self.rho],  # COV
														[self.rho, 1]])
		return result

	def sample(self, n):
		copula_samples = self.meta_Gaussian.rvs(n)
		samples = np.zeros((n, 2))
		samples[:, 0] = self.Law_RS.ppf(norm.cdf(copula_samples[:, 0]))
		samples[:, 1] = self.Law_RF.ppf(norm.cdf(copula_samples[:, 1]))
		return samples

	def sample_uv(self, n):  # sample only the copula (Assuming uniform distribution of marginals)
		copula_samples = self.meta_Gaussian.rvs(n)
		samples = np.zeros((n, 2))
		samples[:, 0] = norm.cdf(copula_samples[:, 0])
		samples[:, 1] = norm.cdf(copula_samples[:, 1])
		return samples

	def tau(self):
		return 2 / np.pi * np.arcsin(self.rho)

	def spearman_rho(self):
		return 6 / np.pi * np.arcsin(self.rho / 2)

	def Fisher_information(rho):
		return (1 + rho ** 2) / (1 - rho ** 2) ** 2

	def mm_loss(self, paras, u, v, q_arr):
		rho = paras
		if np.abs(rho) >= 1:
			return 5000
		self.rho = rho
		self.meta_Gaussian = stats.multivariate_normal([0, 0],  # Mean
													   [[1, rho],  # COV
														[rho, 1]])
		m = np.array([stats.kendalltau(u, v)[0]] + [empirical_lambda(u, v, q) for q in q_arr])
		m_hat = np.array([self.tau()] + [self._lambda(q) for q in q_arr])
		g = m - m_hat
		return g.dot(g.T)

	def mm_calibrate(self, u, v, q_arr):
		fn_toopt = lambda rho: self.mm_loss(rho, u, v, q_arr)
		result = scipy.optimize.fmin(fn_toopt, x0=self.rho,
									 xtol=1e-10,
									 maxiter=5000,
									 maxfun=400)
		self.paras = {"rho": result[0]}
		self.rho = result[0]
		self.meta_Gaussian = stats.multivariate_normal([0, 0],  # Mean
													   [[1, self.rho],  # COV
														[self.rho, 1]])
		return result

	def moment_conditions(self, paras, u, v, q_arr):
		old_paras = self.rho
		rho = paras['rho']
		if np.abs(rho) >= 1:
			return 5000
		self.rho = rho
		self.meta_Gaussian = stats.multivariate_normal([0, 0],  # Mean
													   [[1, rho],  # COV
														[rho, 1]])

		m = np.array([stats.kendalltau(u, v)[0]] + [empirical_lambda(u, v, q) for q in q_arr])
		m_hat = np.array([self.tau()] + [self._lambda(q) for q in q_arr])
		g = m - m_hat

		# Restore old paras
		self.rho = old_paras
		self.meta_Gaussian = stats.multivariate_normal([0, 0],  # Mean
													   [[1, self.rho],  # COV
														[self.rho, 1]])
		mc = pd.DataFrame((m, m_hat, g))
		mc.index = ['Empirical', 'Parametric', 'Difference']
		mc.columns = ['Tau'] + ["lambda " + str(q) for q in q_arr]
		return mc


class t_Copula(Copula):
	def __init__(self, paras, Law_RS, Law_RF, nu_lowerbound):
		super().__init__()
		self.paras = paras
		self.rho = paras["rho"]  # Dependence Parameter
		self.nu = paras["nu"]  # Degree of Freedom
		self.Law_RS = Law_RS  # Marginal Distribution of Spot
		self.Law_RF = Law_RF  # Marginal Distribution of Future
		self.meta_t = multivariate_t(nu=self.nu,  # DF
									 Sigma=np.array([[1, self.rho],  # COV
													 [self.rho, 1]]))
		self.t1 = stats.t(df=self.nu)  # inner
		self.t2 = stats.t(df=self.nu)

		if nu_lowerbound == None:  # lowerest value permitted for degree of freedom
			self.nu_lowerbound = 2
		else:
			self.nu_lowerbound = nu_lowerbound

	def C(self, u, v):  # Copula Function
		return self.meta_t.cdf(np.array([self.t1.ppf(u), self.t2.ppf(v)]))

	def c(self, u, v):  # copula density
		part1 = self.meta_t.pdf(self.t1.ppf(u), self.t2.ppf(v))
		part2 = self.t1.pdf(self.t1.ppf(u)) * self.t2.pdf(self.t2.ppf(v))
		return part1 / part2

	def D1C(self, w, h, r_h):  # D1Operator
		integrand = lambda u: self.meta_t.pdf(self.t1.ppf(w), u)
		part2 = 1 / self.t1.pdf(self.t1.ppf(w))
		return integrate.quad(integrand, -np.infty, self.t2.ppf(self.g(w, h, r_h)))[0] * part2

	def l_fn(self, rho, nu, u, v, nu_lowerbound=2):  # Likelihood Function
		if (np.abs(rho) >= 1) or (nu < nu_lowerbound):
			return -5000

		_meta_t = multivariate_t(nu=nu,  # DF
								 Sigma=np.array([[1, rho],  # COV
												 [rho, 1]]))

		_t1 = stats.t(df=nu)
		_t2 = stats.t(df=nu)

		z1 = _t1.ppf(u)  # inner
		z2 = _t2.ppf(v)

		part1 = []
		for i in range(len(z1)):
			part1.append(_meta_t.pdf(z1[i], z2[i]))

		part1 = np.array(part1)
		part2 = _t1.pdf(_t1.ppf(u)) * _t2.pdf(_t2.ppf(v))
		return np.sum(np.log(part1 / part2))

	def dependency_likelihood(self, u, v):
		rho = self.rho
		nu = self.nu
		return self.l_fn(rho, nu, u, v, nu_lowerbound=2)

	def canonical_calibrate(self, u, v):
		fn_toopt = lambda theta: -self.l_fn(theta[0], theta[1], u, v, self.nu_lowerbound)
		result = scipy.optimize.fmin(fn_toopt, x0=(self.rho, self.nu),
									 xtol=1e-10,
									 maxiter=5000,
									 maxfun=400)
		self.rho = result[0]
		self.nu = result[1]
		self.paras = {"rho": result[0], "nu": result[1]}

		self.meta_t = multivariate_t(nu=self.nu,  # DF
									 Sigma=np.array([[1, self.rho],  # COV
													 [self.rho, 1]]))
		return result

	def sample(self, n):
		copula_samples = self.meta_t.rvs(n)
		samples = np.zeros((n, 2))
		samples[:, 0] = self.Law_RS.ppf(self.t1.cdf(copula_samples[:, 0]))
		samples[:, 1] = self.Law_RF.ppf(self.t2.cdf(copula_samples[:, 1]))
		return samples

	def Fisher_information(rho, nu):
		# I_rho
		part1 = (1 + rho ** 2) / (1 - rho ** 2) ** 2
		part2 = (nu ** 2 + 2 * nu) * (rho ** 2) / (4 * (1 - rho ** 2) ** 2) * beta(3, nu / 2)
		part3 = (nu ** 2 + 2 * nu) * (2 - 3 * rho ** 2 + rho ** 6) / (16 * (1 - rho ** 2) ** 4) * beta(3, nu / 2)
		part4 = (nu ** 2 + 2 * nu) * (1 + rho ** 2) / (2 * (1 - rho ** 2) ** 2) * beta(2, nu / 2)
		I_rho = part1 + part2 + part3 + part4

		# I_nu
		I_nu = 1 / nu * beta(2, nu / 2) - (nu + 2) / (4 * nu) * beta(3, nu / 2)

		# I_rhonu
		I_rhonu = -rho / (2 * (1 - rho ** 2)) * (beta(2, nu / 2) - (nu + 2) / 2 * beta(3, nu / 2))
		return I_rho, I_nu, I_rhonu

	def tau(self):
		return 2 / np.pi * np.arcsin(self.rho)

	def mm_loss(self, paras, u, v, q_arr):
		rho, nu = paras
		if np.abs(rho) >= 1:
			return 5000
		if nu < self.nu_lowerbound:
			return 5000

		self.__init__({'rho': rho, 'nu': nu}, Law_RS=self.Law_RS, Law_RF=self.Law_RF, nu_lowerbound=self.nu_lowerbound)

		m = np.array([stats.kendalltau(u, v)[0]] + [empirical_lambda(u, v, q) for q in q_arr])
		m_hat = np.array([self.tau()] + [self._lambda(q) for q in q_arr])
		g = m - m_hat
		return g.dot(g.T)

	def mm_calibrate(self, u, v, q_arr):
		fn_toopt = lambda paras: self.mm_loss(paras, u, v, q_arr)
		result = scipy.optimize.fmin(fn_toopt, x0=(self.rho, self.nu),
									 xtol=1e-10,
									 maxiter=5000,
									 maxfun=400)
		self.__init__({'rho': result[0], 'nu': result[1]}, self.Law_RS, self.Law_RF, self.nu_lowerbound)
		return result

	def moment_conditions(self, paras, u, v, q_arr):
		old_paras = self.paras
		self.__init__(paras, Law_RS=self.Law_RS, Law_RF=self.Law_RF, nu_lowerbound=self.nu_lowerbound)

		m = np.array([stats.kendalltau(u, v)[0]] + [empirical_lambda(u, v, q) for q in q_arr])
		m_hat = np.array([self.tau()] + [self._lambda(q) for q in q_arr])
		g = m - m_hat

		self.__init__(old_paras, Law_RS=self.Law_RS, Law_RF=self.Law_RF, nu_lowerbound=self.nu_lowerbound)
		mc = pd.DataFrame((m, m_hat, g))
		mc.index = ['Empirical', 'Parametric', 'Difference']
		mc.columns = ['Tau'] + ["lambda " + str(q) for q in q_arr]
		return mc


class Clayton(Copula):
	def __init__(self, paras, Law_RS, Law_RF):
		super().__init__()
		self.paras = paras
		self.theta = paras["theta"]  # Dependence Parameter
		self.Law_RS = Law_RS  # Marginal Distribution of Spot
		self.Law_RF = Law_RF  # Marginal Distribution of Future

	def phi(self, t):
		A = 1 / self.theta
		B = t ** (-self.theta) - 1
		return A * B

	def phi_inverse(self, t):
		A = (1 + self.theta * t)
		B = -1 / self.theta
		return A ** B

	def d_phi(self, t):
		return -t ** (-self.theta - 1)

	def d_phi_inverse(self, t):
		A = 1 + self.theta * t
		B = (-1 / self.theta) - 1
		return -1 * A ** B

	def C(self, u, v):
		return self.phi_inverse(self.phi(u) + self.phi(v))

	def c(self, u, v):  # copula density
		part1 = (1 + self.theta) * (u * v) ** (-1 - self.theta)
		part2 = (-1 + u ** (-self.theta) + v ** (-self.theta)) ** (-2 - (1 / self.theta))
		return part1 * part2

	def D1C(self, w, h, r_h):
		a = self.phi(w) + self.phi(self.g(w, h, r_h))
		A = self.d_phi_inverse(a)
		B = self.d_phi(w)
		return A * B

	def canonical_calibrate(self, u, v):
		fn_toopt = lambda theta: -self.l_fn(theta, u, v)
		result = scipy.optimize.fmin(fn_toopt, x0=self.theta,
									 xtol=1e-10,
									 maxiter=5000,
									 maxfun=400)
		self.theta = result[0]
		self.paras = {"theta": result[0]}
		return result

	def l_fn(self, theta, u, v):  # log dependency likelihood
		part1 = (1 + theta) * (u * v) ** (-1 - theta)
		part2 = (-1 + u ** (-theta) + v ** (-theta)) ** (-2 - (1 / theta))
		l = np.log(part1 * part2)
		l[~np.isfinite(l)] = np.min(l[np.isfinite(l)]) * 10
		return np.sum(l)

	def dependency_likelihood(self, u, v):
		theta = self.theta
		return self.l_fn(theta, u, v)

	def sample(self, n):
		u1 = stats.uniform().rvs(n)
		v2 = stats.uniform().rvs(n)
		u2 = (u1 ** (-self.theta) * (v2 ** (-self.theta / (1 + self.theta)) - 1) + 1) ** (-1 / self.theta)
		samples = np.zeros((n, 2))
		samples[:, 0] = self.Law_RS.ppf(u1)
		samples[:, 1] = self.Law_RF.ppf(u2)
		return samples

	def Fisher_information(theta):  # Clayton
		# rho(theta)
		part1 = 1 / ((3 * theta - 2) * (2 * theta - 1))

		part2a = theta / (2 * (3 * theta - 2) * (2 * theta - 1) * (theta - 1))
		part2b = zeta(2, 1 / (2 * (theta - 1))) - zeta(2, theta / (
				2 * (theta - 1)))  # Trigamma is a special case of Huritz zeta function
		part2 = part2a * part2b

		part3a = 1 / (2 * (3 * theta - 2) * (2 * theta - 1) * (theta - 1))
		part3b = zeta(2, theta / (2 * (theta - 1))) - zeta(2, (2 * theta - 1) / (
				2 * (theta - 1)))  # Trigamma is a special case of Huritz zeta function
		part3 = part3a * part3b
		rho = part1 + part2 + part3

		I = 1 / theta ** 2 + 2 / (theta * (theta - 1) * (2 * theta - 1)) + 4 * theta / (3 * theta - 2) - 2 * (
				2 * theta - 1) / (theta - 1) * rho
		return I

	def tau(self): # Example 5.4 in Nelsen's Book
		return (self.theta / (self.theta + 2))

	def mm_loss(self, paras, u, v, q_arr):
		theta = paras
		self.__init__({'theta': theta}, Law_RS=self.Law_RS, Law_RF=self.Law_RF)
		m = np.array([stats.kendalltau(u, v)[0]] + [empirical_lambda(u, v, q) for q in q_arr]).reshape((1 + len(q_arr)))
		m_hat = np.array([self.tau()] + [self._lambda(q) for q in q_arr]).reshape((1 + len(q_arr)))
		g = m - m_hat
		g = g.reshape((-1, 1))
		return g.T.dot(g)[0][0]

	def mm_calibrate(self, u, v, q_arr):
		fn_toopt = lambda paras: self.mm_loss(paras, u, v, q_arr)
		result = scipy.optimize.fmin(fn_toopt, x0=self.theta,
									 xtol=1e-10,
									 maxiter=5000,
									 maxfun=400)
		self.__init__({'theta': result[0]}, self.Law_RS, self.Law_RF)
		return result

	def moment_conditions(self, paras, u, v, q_arr):
		old_paras = self.paras
		self.__init__(paras, Law_RS=self.Law_RS, Law_RF=self.Law_RF)

		m = np.array([stats.kendalltau(u, v)[0]] + [empirical_lambda(u, v, q) for q in q_arr]).reshape((1 + len(q_arr)))
		m_hat = np.array([self.tau()] + [self._lambda(q) for q in q_arr])
		g = m - m_hat

		self.__init__(old_paras, Law_RS=self.Law_RS, Law_RF=self.Law_RF)
		mc = pd.DataFrame((m, m_hat, g))
		mc.index = ['Empirical', 'Parametric', 'Difference']
		mc.columns = ['Tau'] + ["lambda " + str(q) for q in q_arr]
		return mc


class Frank(Copula):
	def __init__(self, paras, Law_RS, Law_RF):
		self.paras = paras
		self.theta = paras["theta"]  # Dependence Parameter
		self.Law_RS = Law_RS  # Marginal Distribution of Spot
		self.Law_RF = Law_RF  # Marginal Distribution of Future

	def phi(self, t):
		A = np.exp(-self.theta * t) - 1
		B = np.exp(-self.theta) - 1
		return -np.log(A / B)

	def phi_inverse(self, t):
		A = -1 / self.theta
		B = 1 + (np.exp(-t) * (np.exp(-self.theta) - 1))
		return A * np.log(B)

	def d_phi(self, t):
		A = self.theta * np.exp(-self.theta * t)
		B = np.exp(-self.theta * t) - 1
		return A / B

	def d_phi_inverse(self, t):
		A = 1 / self.theta
		B = np.exp(-t) * (np.exp(-self.theta) - 1)
		C = 1 + B
		return A * B / C

	def D1C(self, w, h, r_h):
		a = self.phi(w) + self.phi(self.g(w, h, r_h))
		A = self.d_phi_inverse(a)
		B = self.d_phi(w)
		return A * B

	def C(self, u, v):
		return self.phi_inverse(self.phi(u) + self.phi(v))

	def c(self, u, v):  # copula density (wiki missed the negative sign in front of theta)
		part1 = -self.theta * np.exp(-self.theta * (u + v)) * (np.exp(-self.theta) - 1)
		part2 = np.exp(-self.theta) - np.exp(-self.theta * u) - np.exp(-self.theta * v) + np.exp(-self.theta * (u + v))
		return part1 / (part2 ** 2)

	def l_fn(self, theta, u, v):  # log dependency likelihood
		part1 = -theta * np.exp(-theta * (u + v)) * (np.exp(-theta) - 1)
		part2 = np.exp(-theta) - np.exp(-theta * u) - np.exp(-theta * v) + np.exp(-theta * (u + v))
		return np.sum(np.log(part1 / (part2 ** 2)))

	def dependency_likelihood(self, u, v):
		theta = self.theta
		return self.l_fn(theta, u, v)

	def canonical_calibrate(self, u, v):
		fn_toopt = lambda theta: -self.l_fn(theta, u, v)
		result = scipy.optimize.fmin(fn_toopt, x0=self.theta,
									 xtol=1e-10,
									 maxiter=5000,
									 maxfun=400)
		self.theta = result[0]
		self.paras = {"theta": result[0]}
		return result

	def tau(self, theta=None):  # Statistical modeling of joint probability distribution using copula: Application to peak
		# and permanent displacement seismic demands
		# Exercise 5.9 in Nelsen
		if theta == None:
			theta = self.theta
		part1 = 1 - 4 / theta
		part2 = 4 / theta ** 2
		part3_fn = lambda t: t / (np.exp(t) - 1)
		part3 = scipy.integrate.quad(part3_fn, 0, theta)[0]
		return (part1 + part2 * part3)

	def D1C_inv(self, u, v):  # D1C's inverse
		part0 = -1 / self.theta
		part1a = (1 - np.exp(-self.theta))
		part1b = (1 / v - 1) * np.exp(-self.theta * u) + 1
		part1 = np.log(1 - part1a / part1b)
		return part0 * part1

	def sample(self, size):
		samples = np.ones((size, 2))
		u1 = stats.uniform.rvs(size=size)
		v = stats.uniform.rvs(size=size)
		u2 = self.D1C_inv(u1, v)
		u2[u2>1] = 1 # prevent u2 > 1
		samples[:, 0] = self.Law_RS.ppf(u1)
		samples[:, 1] = self.Law_RF.ppf(u2)
		return samples

	def mm_loss(self, paras, u, v, q_arr):
		theta = paras
		self.__init__({'theta': theta}, Law_RS=self.Law_RS, Law_RF=self.Law_RF)
		m = np.array([stats.kendalltau(u, v)[0]] + [empirical_lambda(u, v, q) for q in q_arr]).reshape((1 + len(q_arr)))
		m_hat = np.array([self.tau()] + [self._lambda(q) for q in q_arr]).reshape((1 + len(q_arr)))
		g = m - m_hat
		g = g.reshape((-1, 1))
		return g.T.dot(g)[0][0]

	def mm_calibrate(self, u, v, q_arr):
		fn_toopt = lambda paras: self.mm_loss(paras, u, v, q_arr)
		result = scipy.optimize.fmin(fn_toopt, x0=self.theta,
									 xtol=1e-10,
									 maxiter=5000,
									 maxfun=400)
		self.__init__({'theta': result[0]}, self.Law_RS, self.Law_RF)
		return result

	def moment_conditions(self, paras, u, v, q_arr):
		old_paras = self.paras
		self.__init__(paras, Law_RS=self.Law_RS, Law_RF=self.Law_RF)

		m = np.array([stats.kendalltau(u, v)[0]] + [empirical_lambda(u, v, q) for q in q_arr]).reshape((1 + len(q_arr)))
		m_hat = np.array([self.tau()] + [self._lambda(q) for q in q_arr])
		g = m - m_hat

		self.__init__(old_paras, Law_RS=self.Law_RS, Law_RF=self.Law_RF)
		mc = pd.DataFrame((m, m_hat, g))
		mc.index = ['Empirical', 'Parametric', 'Difference']
		mc.columns = ['Tau'] + ["lambda " + str(q) for q in q_arr]
		return mc


class Gumbel(Copula):
	def __init__(self, paras, Law_RS, Law_RF):
		self.paras = paras
		self.theta = paras["theta"]  # Dependence Parameter
		self.Law_RS = Law_RS  # Marginal Distribution of Spot
		self.Law_RF = Law_RF  # Marginal Distribution of Future

	def phi(self, t):
		return (-np.log(t)) ** self.theta

	def phi_inverse(self, t):
		return np.exp(-(t ** (1 / self.theta)))

	def d_phi(self, t):
		return self.theta * self.phi(t) / (t * np.log(t))

	def d_phi_inverse(self, t):
		A = -1 / self.theta
		B = t ** ((1 / self.theta) - 1)
		C = self.phi_inverse(t)
		return A * B * C

	def C(self, u, v):
		return self.phi_inverse(self.phi(u) + self.phi(v))

	def c(self, u, v):  # p.142 of Joe
		u_t = -np.log(u)
		v_t = -np.log(v)
		part1 = self.C(u, v) / (u * v)
		part2 = (u_t * v_t) ** (self.theta - 1) / (u_t ** self.theta + v_t ** self.theta) ** (2 - 1 / self.theta)
		part3 = (u_t ** self.theta + v_t ** self.theta) ** (1 / self.theta) + self.theta - 1
		return part1 * part2 * part3

	def D1C(self, w, h, r_h):
		a = self.phi(w) + self.phi(self.g(w, h, r_h))
		A = self.d_phi_inverse(a)
		B = self.d_phi(w)
		return A * B

	def Gumbel_copula(self, u, v, theta):  # Copula function for calibration
		t1 = (-np.log(u)) ** theta
		t2 = (-np.log(v)) ** theta
		part1 = t1 + t2
		part2 = part1 ** (1 / theta)
		return np.exp(-part2)

	def l_fn(self, theta, u, v, verbose=False):
		if theta < 1:
			print("theta is smaller then 1; consider changing x0 of fmin by initiating the class with different theta")
			return -5000
		try:  # turn u==1 to a slightly smaller number to avoid inf
			u[u == 1] = max(u[u != 1]) + 0.9 / len(u)
			v[v == 1] = max(v[v != 1]) + 0.9 / len(v)
		except:
			pass
		t1 = -np.log(u)
		t2 = -np.log(v)
		part1 = 1 / (u * v)
		part2 = self.Gumbel_copula(u, v, theta)
		part3 = t1 ** (-1 + theta)
		part4 = t2 ** (-1 + theta)
		part5 = -1 + theta + (t1 ** theta + t2 ** theta) ** (1 / theta)
		part6 = (t1 ** theta + t2 ** theta) ** (-2 + (1 / theta))
		if np.sum(np.isfinite(part6)) > 0:
			part6[~np.isfinite(part6)] = np.max(part6[np.isfinite(part6)]) * 10
		else:
			part6[~np.isfinite(part6)] = 100
		if verbose:
			print(part1, part2, part3, part4, part5, part6)
		return np.sum(np.log(part1 * part2 * part3 * part4 * part5 * part6))

	def dependency_likelihood(self, u, v):
		theta = self.theta
		return self.l_fn(theta, u, v)

	def canonical_calibrate(self, u, v):
		fn_toopt = lambda theta: -self.l_fn(theta, u, v)
		result = scipy.optimize.fmin(fn_toopt, x0=self.theta,
									 xtol=1e-10,
									 maxiter=5000,
									 maxfun=400)
		self.theta = result[0]
		self.paras = {"theta": result[0]}
		return result[0]

	def sample(self, size):
		samples = np.zeros((size, 2))
		gamma = np.cos(np.pi / (2 * self.theta)) ** self.theta
		V = stable(1 / self.theta, 1, 0, gamma).rvs(size)
		G_hat = lambda t: np.exp(-t ** (1 / self.theta))

		X1 = stats.uniform.rvs(size=size)
		X2 = stats.uniform.rvs(size=size)

		U1 = G_hat(-np.log(X1) / V)
		U2 = G_hat(-np.log(X2) / V)

		# samples of stable contain NaNs; Draw more samples and discard NaNs until len(V) = size
		nan_id = np.isnan(U1)
		while np.sum(nan_id) != 0:
			size1 = np.sum(nan_id)
			V = stable(1 / self.theta, 1, 0, gamma).rvs(size1)
			X1 = stats.uniform.rvs(size=size1)
			X2 = stats.uniform.rvs(size=size1)

			U1[nan_id] = G_hat(-np.log(X1) / V)
			U2[nan_id] = G_hat(-np.log(X2) / V)
			nan_id = np.isnan(U1)

		samples[:, 0] = self.Law_RS.ppf(U1)
		samples[:, 1] = self.Law_RF.ppf(U2)
		return samples

	def tau(self):
		return (self.theta - 1)/ self.theta

	def mm_loss(self, paras, u, v, q_arr):
		theta = paras

		self.__init__({'theta': theta}, Law_RS=self.Law_RS, Law_RF=self.Law_RF)

		m = np.array([stats.kendalltau(u, v)[0]] + [empirical_lambda(u, v, q) for q in q_arr]).reshape((1 + len(q_arr)))
		m_hat = np.array([self.tau()] + [self._lambda(q) for q in q_arr]).reshape((1 + len(q_arr)))
		g = m - m_hat
		g = g.reshape((-1, 1))
		return g.T.dot(g)[0][0]

	def mm_calibrate(self, u, v, q_arr):
		fn_toopt = lambda paras: self.mm_loss(paras, u, v, q_arr)
		result = scipy.optimize.fmin(fn_toopt, x0=self.theta,
									 xtol=1e-10,
									 maxiter=5000,
									 maxfun=400)
		self.__init__({'theta': result[0]}, self.Law_RS, self.Law_RF)
		return result

	def moment_conditions(self, paras, u, v, q_arr):
		old_paras = self.paras
		self.__init__(paras, Law_RS=self.Law_RS, Law_RF=self.Law_RF)

		m = np.array([stats.kendalltau(u, v)[0]] + [empirical_lambda(u, v, q) for q in q_arr])
		m_hat = np.array([self.tau()] + [self._lambda(q) for q in q_arr]).reshape((1 + len(q_arr)))
		g = m - m_hat

		self.__init__(old_paras, Law_RS=self.Law_RS, Law_RF=self.Law_RF)
		mc = pd.DataFrame((m, m_hat, g))
		mc.index = ['Empirical', 'Parametric', 'Difference']
		mc.columns = ['Tau'] + ["lambda " + str(q) for q in q_arr]
		return mc


class rot180Gumbel(Gumbel):
	def C(self, u, v):
		u = 1 - u
		v = 1 - v
		return self.phi_inverse(self.phi(u) + self.phi(v))

	def c(self, u, v):
		u = 1 - u
		v = 1 - v

		u_t = -np.log(u)
		v_t = -np.log(v)
		part1 = self.C(1 - u, 1 - v) / (u * v)
		part2 = (u_t * v_t) ** (self.theta - 1) / (u_t ** self.theta + v_t ** self.theta) ** (2 - 1 / self.theta)
		part3 = (u_t ** self.theta + v_t ** self.theta) ** (1 / self.theta) + self.theta - 1
		return part1 * part2 * part3

	def Gumbel_copula(self, u, v, theta):  # Copula function for calibration
		u = 1 - u
		v = 1 - v

		t1 = (-np.log(u)) ** theta
		t2 = (-np.log(v)) ** theta
		part1 = t1 + t2
		part2 = part1 ** (1 / theta)
		return np.exp(-part2)

	def l_fn(self, theta, u, v, verbose=False):
		u = 1 - u  # the rotation is in effect in this function, no rotation is needed
		v = 1 - v

		if theta < 1:
			print("theta is smaller then 1; consider changing x0 of fmin by initiating the class with different theta")
			return -5000
		try:  # turn u==1 to a slightly smaller number to avoid inf
			u[u == 1] = max(u[u != 1]) + 0.9 / len(u)
			v[v == 1] = max(v[v != 1]) + 0.9 / len(v)
		except:
			pass
		t1 = -np.log(u)
		t2 = -np.log(v)
		part1 = 1 / (u * v)
		part2 = self.Gumbel_copula(1 - u, 1 - v, theta)
		part3 = t1 ** (-1 + theta)
		part4 = t2 ** (-1 + theta)
		part5 = -1 + theta + (t1 ** theta + t2 ** theta) ** (1 / theta)
		part6 = (t1 ** theta + t2 ** theta) ** (-2 + (1 / theta))
		if np.sum(np.isfinite(part6)) > 0:
			part6[~np.isfinite(part6)] = np.max(part6[np.isfinite(part6)]) * 10
		else:
			part6[~np.isfinite(part6)] = 100
		if verbose:
			print(part1, part2, part3, part4, part5, part6)
		return np.sum(np.log(part1 * part2 * part3 * part4 * part5 * part6))

	def sample(self, size):
		samples = np.zeros((size, 2))
		gamma = np.cos(np.pi / (2 * self.theta)) ** self.theta
		V = stable(1 / self.theta, 1, 0, gamma).rvs(size)
		G_hat = lambda t: np.exp(-t ** (1 / self.theta))

		X1 = stats.uniform.rvs(size=size)
		X2 = stats.uniform.rvs(size=size)

		U1 = G_hat(-np.log(X1) / V)
		U2 = G_hat(-np.log(X2) / V)

		# samples of stable contain NaNs; Draw more samples and discard NaNs until len(V) = size
		nan_id = np.isnan(U1)
		while np.sum(nan_id) != 0:
			size1 = np.sum(nan_id)
			V = stable(1 / self.theta, 1, 0, gamma).rvs(size1)
			X1 = stats.uniform.rvs(size=size1)
			X2 = stats.uniform.rvs(size=size1)

			U1[nan_id] = G_hat(-np.log(X1) / V)
			U2[nan_id] = G_hat(-np.log(X2) / V)
			nan_id = np.isnan(U1)

		samples[:, 0] = self.Law_RS.ppf(1 - U1)
		samples[:, 1] = self.Law_RF.ppf(1 - U2)
		return samples

	def _lambda(self, q):
		q = 1 - q
		if q <= 0.5:
			return self.C(1 - q, 1 - q) / q
		else:
			return (1 - (2 * q) + self.C(1 - q, 1 - q)) / (1 - q)


class Plackett(Copula):
	def __init__(self, paras, Law_RS, Law_RF):
		super().__init__()
		self.paras = paras
		self.theta = paras["theta"]  # Dependence Parameter
		self.Law_RS = Law_RS  # Marginal Distribution of Spot
		self.Law_RF = Law_RF  # Marginal Distribution of Future

	# from Joe's book, p. 141 (Family B2)
	def C(self, u, v):
		eta = self.theta - 1
		part1 = 1 / (2 * eta)
		part2 = 1 + (eta * (u + v))
		part3 = (part2 ** 2 - 4 * eta * self.theta * u * v) ** 0.5
		return part1 * (part2 - part3)

	def c(self, u, v):
		eta = self.theta - 1
		part1 = ((1 + eta * (u + v)) ** 2 - 4 * self.theta * eta * u * v) ** (-3 / 2)
		part2 = self.theta * (1 + eta * (u + v - 2 * u * v))
		return part1 * part2

	def D1C_original(self, u, v):
		eta = self.theta - 1
		part1 = 0.5 * (eta * u + 1 - (eta + 2) * v)
		part2 = ((1 + eta * (u + v)) ** 2 - 4 * self.theta * eta * u * v) ** 0.5
		return 0.5 - part1 / part2

	def D1C(self, w, h, r_h):
		return self.D1C_original(w, self.g(w, h, r_h))

	def get_theta(self, u, v):
		C = self.C(u, v)
		part1 = C * (1 - u - v + C)
		part2 = (u - C) * (v - C)
		return part1 / part2

	def sample(self, size):
		samples = np.ones((size, 2))
		u = stats.uniform.rvs(size=size)
		t = stats.uniform.rvs(size=size)

		a = t * (1 - t)
		b = self.theta + a * (self.theta - 1) ** 2
		c = 2 * a * (u * self.theta ** 2 + 1 - u) + self.theta * (1 - 2 * a)
		d = np.sqrt(self.theta * (self.theta + 4 * a * u * (1 - u) * (1 - self.theta) ** 2))
		v = (c - (1 - 2 * t) * d) / (2 * b)
		samples[:, 0] = self.Law_RS.ppf(u)
		samples[:, 1] = self.Law_RF.ppf(v)
		return samples

	def dependency_likelihood(self, u, v):
		theta = self.theta
		return self.l_fn(theta, u, v)

	def canonical_calibrate(self, u, v):
		fn_toopt = lambda theta: -self.l_fn(theta, u, v)
		result = scipy.optimize.fmin(fn_toopt, x0=self.theta,
									 xtol=1e-10,
									 maxiter=5000,
									 maxfun=400)
		self.theta = result[0]
		self.paras = {"theta": result[0]}
		return result[0]

	def l_fn(self, theta, u, v):  # log dependency likelihood
		eta = theta - 1
		part1 = ((1 + eta * (u + v)) ** 2 - 4 * theta * eta * u * v) ** (-3 / 2)
		part2 = theta * (1 + eta * (u + v - 2 * u * v))
		return np.sum(np.log(part1 * part2))

	# From Appendix C.7. of "Extreme in Nature"
	def spearman_rho(self):
		part1 = (self.theta + 1) / (self.theta - 1)
		part2 = 2 * self.theta * np.log(self.theta) / (self.theta - 1) ** 2
		return part1 - part2

	def mm_loss(self, paras, u, v, q_arr):
		theta = paras
		self.__init__({'theta': theta}, Law_RS=self.Law_RS, Law_RF=self.Law_RF)
		m = np.array([stats.spearmanr(u, v)[0]] + [empirical_lambda(u, v, q) for q in q_arr]).reshape((1 + len(q_arr)))
		m_hat = np.array([self.spearman_rho()] + [self._lambda(q) for q in q_arr]).reshape((1 + len(q_arr)))
		g = m - m_hat
		g = g.reshape((-1, 1))
		return g.T.dot(g)[0][0]

	def mm_calibrate(self, u, v, q_arr):
		fn_toopt = lambda paras: self.mm_loss(paras, u, v, q_arr)
		result = scipy.optimize.fmin(fn_toopt, x0=self.theta,
									 xtol=1e-10,
									 maxiter=5000,
									 maxfun=400)
		self.__init__({'theta': result[0]}, self.Law_RS, self.Law_RF)
		return result

	def moment_conditions(self, paras, u, v, q_arr):
		old_paras = self.paras
		self.__init__(paras, Law_RS=self.Law_RS, Law_RF=self.Law_RF)
		m = np.array([stats.spearmanr(u, v)[0]] + [empirical_lambda(u, v, q) for q in q_arr]).reshape((1 + len(q_arr)))
		m_hat = np.array([self.spearman_rho()] + [self._lambda(q) for q in q_arr]).reshape((1 + len(q_arr)))
		g = m - m_hat

		self.__init__(old_paras, Law_RS=self.Law_RS, Law_RF=self.Law_RF)
		mc = pd.DataFrame((m, m_hat, g))
		mc.index = ['Empirical', 'Parametric', 'Difference']
		mc.columns = ['Spearman Rho'] + ["lambda " + str(q) for q in q_arr]
		return mc


class Gaussian_Mix_Independent(Copula):
	def __init__(self, paras, Law_RS, Law_RF):
		super().__init__()
		self.paras = paras
		self.rho = paras["rho"]
		self.p = paras["p"]
		self.Gaussian = Gaussian({"rho": self.rho}, Law_RS, Law_RF)
		self.Independent = Gaussian({"rho": 0}, Law_RS, Law_RF)
		self.Law_RS = Law_RS  # Marginal Distribution of Spot
		self.Law_RF = Law_RF  # Marginal Distribution of Future

	def C(self, u, v):
		return self.p * self.Gaussian.C(u, v) + (1 - self.p) * self.Independent.C(u, v)

	def c(self, u, v):
		return self.p * self.Gaussian.c(u, v) + (1 - self.p)

	def D1C(self, w, h, r_h):
		return self.p * self.Gaussian.D1C(w, h, r_h) + (1 - self.p) * self.g(w, h, r_h)

	def l_fn(self, rho, p, u, v):
		if (p < 0) or (p > 1) or (np.abs(rho) > .999):
			return -5000
		_Gaussian = Gaussian({"rho": rho}, stats.norm, stats.norm)
		_Gaussian_c = np.array([_Gaussian.c(u[i], v[i]) for i in range(len(u))])
		return np.sum(np.log(p * _Gaussian_c + (1 - p)))

	def dependency_likelihood(self, u, v):
		rho = self.rho
		p = self.p
		return self.l_fn(rho, p, u, v)

	def canonical_calibrate(self, u, v):
		fn_toopt = lambda para: -self.l_fn(para[0], para[1], u, v)
		result = scipy.optimize.fmin(fn_toopt, x0=(self.rho, self.p),
									 xtol=1e-10,
									 maxiter=5000,
									 maxfun=400)
		self.rho = result[0]
		self.p = result[1]
		self.paras = {"rho": result[0], "p": result[1]}
		self.Gaussian = Gaussian({"rho": self.rho}, self.Law_RS, self.Law_RF)
		return result

	def sample(self, size):
		samples = np.zeros((size, 2))
		n_Gaussian = int(self.p * size)
		n_Independent = size - n_Gaussian
		samples[:n_Gaussian, :] = self.Gaussian.sample(n_Gaussian)

		I1 = self.Law_RS.ppf(stats.uniform.rvs(size=n_Independent))
		I2 = self.Law_RF.ppf(stats.uniform.rvs(size=n_Independent))

		samples[n_Gaussian:, 0] = I1
		samples[n_Gaussian:, 1] = I2

		return samples

	def spearman_rho(self):
		return self.p * self.Gaussian.spearman_rho()

	def mm_loss(self, paras, u, v, q_arr):
		rho, p = paras
		if np.abs(rho) > 0.999:
			return 5000
		if p > 0.999:
			return 5000

		self.__init__({'rho': rho, 'p': p}, Law_RS=self.Law_RS, Law_RF=self.Law_RF)

		m = np.array([stats.spearmanr(u, v)[0]] + [empirical_lambda(u, v, q) for q in q_arr]).reshape((1 + len(q_arr)))
		m_hat = np.array([self.spearman_rho()] + [self._lambda(q) for q in q_arr])
		g = m - m_hat
		return g.dot(g.T)

	def mm_calibrate(self, u, v, q_arr):
		fn_toopt = lambda paras: self.mm_loss(paras, u, v, q_arr)
		result = scipy.optimize.fmin(fn_toopt, x0=(self.rho, self.p),
									 xtol=1e-10,
									 maxiter=5000,
									 maxfun=400)
		self.__init__({'rho': result[0], 'p': result[1]}, self.Law_RS, self.Law_RF)
		return result

	def moment_conditions(self, paras, u, v, q_arr):
		old_paras = self.paras
		self.__init__(paras, Law_RS=self.Law_RS, Law_RF=self.Law_RF)

		m = np.array([stats.spearmanr(u, v)[0]] + [empirical_lambda(u, v, q) for q in q_arr]).reshape((1 + len(q_arr)))
		m_hat = np.array([self.spearman_rho()] + [self._lambda(q) for q in q_arr])
		g = m - m_hat

		self.__init__(old_paras, Law_RS=self.Law_RS, Law_RF=self.Law_RF)
		mc = pd.DataFrame((m, m_hat, g))
		mc.index = ['Empirical', 'Parametric', 'Difference']
		mc.columns = ['Spearman Rho'] + ["lambda " + str(q) for q in q_arr]
		return mc
    
    
    
class Gaussian_Mix_Gaussian(Copula):
	def __init__(self, paras, Law_RS, Law_RF):
		super().__init__()
		self.paras = paras
		self.rho1 = paras["rho1"]
		self.rho2 = paras["rho2"]
		self.p = paras["p"]
		self.Gaussian1 = Gaussian({"rho": self.rho1}, Law_RS, Law_RF)
		self.Gaussian2 = Gaussian({"rho": self.rho2}, Law_RS, Law_RF)
		self.Law_RS = Law_RS  # Marginal Distribution of Spot
		self.Law_RF = Law_RF  # Marginal Distribution of Future

	def C(self, u, v):
		return self.p * self.Gaussian1.C(u, v) + (1 - self.p) * self.Gaussian2.C(u, v)

	def c(self, u, v):
		return self.p * self.Gaussian.c(u, v) + (1 - self.p)* self.Gaussian2.C(u, v)

	def l_fn(self, rho1, rho2, p, u, v):
		if (p < 0) or (p > 1) or (np.abs(rho1) > .999) or (np.abs(rho2) > .999):
			return -5000
		_Gaussian1 = Gaussian({"rho": rho1}, stats.norm, stats.norm)
		_Gaussian_c1 = np.array([_Gaussian1.c(u[i], v[i]) for i in range(len(u))])
		_Gaussian2 = Gaussian({"rho": rho2}, stats.norm, stats.norm)
		_Gaussian_c2 = np.array([_Gaussian2.c(u[i], v[i]) for i in range(len(u))])
		return np.sum(np.log(p * _Gaussian_c1 + (1 - p)* _Gaussian_c2))

	def dependency_likelihood(self, u, v):
		rho1 = self.rho1
		rho2 = self.rho2

		p = self.p
		return self.l_fn(rho1, rho2, p, u, v)

	def canonical_calibrate(self, u, v):
		fn_toopt = lambda para: -self.l_fn(para[0], para[1], para[2], u, v)
		result = scipy.optimize.fmin(fn_toopt, x0=(self.rho1,self.rho2, self.p),
									 xtol=1e-10,
									 maxiter=5000,
									 maxfun=400)
		self.rho1 = result[0]
		self.rho2 = result[1]
		self.p = result[2]
		self.paras = {"rho1": result[0],"rho2": result[1], "p": result[2]}
		self.Gaussian1 = Gaussian({"rho": self.rho1}, self.Law_RS, self.Law_RF)
		self.Gaussian2 = Gaussian({"rho": self.rho2}, self.Law_RS, self.Law_RF)
		return result

	def sample(self, size):
		samples = np.zeros((size, 2))
		n_Gaussian1 = int(self.p * size)
		n_Gaussian2 = size - n_Gaussian1
		samples[:n_Gaussian, :] = self.Gaussian1.sample(n_Gaussian1)
		samples[n_Gaussian:, :] = self.Gaussian2.sample(n_Gaussian2)

		return samples

	def spearman_rho(self):
		return self.p * self.Gaussian1.spearman_rho() + (1-self.p)*self.p * self.Gaussian2.spearman_rho()