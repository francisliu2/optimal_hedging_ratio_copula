from copulae1 import *
from KDEs import *
from toolbox import *
import json
import argparse
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.stats import norm
import pandas as pd
import numpy as np
import seaborn as sns
# from statsmodels.distributions.empirical_distribution import ECDF # don't use that
import os
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="input config json file path")
args = parser.parse_args()
config_path = args.config
with open(config_path) as f:
    config = json.load(f)

# Data source
data_name = config['data_name']
data_path = "../processed_data/" + data_name + "/"
spot_name = config['spot_name']
future_name = config['future_name']

# Calibration Method
calibration_method = config['calibration_method']  # MM or MLE
if calibration_method =='MM':
    q_arr = config['q_arr']  # moment conditions for MM

# Parameters
if calibration_method == "MLE":
    paras = pd.read_json("../results/" + data_name + "/MLE/parameters.json")
elif calibration_method == "MM":
    paras = pd.read_json("../results/" + data_name + "/MM/parameters.json")

# Risk Measure Specification
k_arr = config["k_ERM"] # Absolute risk aversion for exponential risk measure
q_arr_ES = config["q_arr_ES"] # Quantile level for expected shortfall
q_arr_VaR = config["q_arr_VaR"] # Quantile level for Value at Risk

# To clip the h
min_h = config["h_Clip"][0]
max_h = config["h_Clip"][1]

# Gaussian, t_Copula, Clayton, Frank, Gumbel, Plackett, Gaussian mix Indep
C1 = Gaussian(dict(rho=0.9), Law_RS=stats.norm, Law_RF=stats.norm)  # fix the maringals!
C2 = t_Copula(dict(rho=0.1, nu=4), Law_RS=stats.norm, Law_RF=stats.norm, nu_lowerbound=2)
C2c = t_Copula(dict(rho=0.1, nu=4), Law_RS=stats.norm, Law_RF=stats.norm, nu_lowerbound=4)
C3 = Clayton(dict(theta=0.1), Law_RS=stats.norm, Law_RF=stats.norm)
C4 = Frank(dict(theta=0.1), Law_RS=stats.norm, Law_RF=stats.norm)
C5 = Gumbel(dict(theta=3), Law_RS=stats.norm, Law_RF=stats.norm)
C6 = Plackett(dict(theta=10), Law_RS=stats.norm, Law_RF=stats.norm)
C7 = Gaussian_Mix_Independent(dict(rho=.5, p=0.7), Law_RS=stats.norm, Law_RF=stats.norm)
C8 = rot180Gumbel(dict(theta=3), Law_RS=stats.norm, Law_RF=stats.norm)

Copulae_names = ['Gaussian', 't_Copula', 't_Copula_Capped', 'Clayton', 'Frank', 'Gumbel', 'Plackett', 'Gauss Mix Indep', 'rotGumbel']
Copulae_arr = [C1, C2, C2c, C3, C4, C5, C6, C7, C8]
Copulae = dict(zip(Copulae_names, Copulae_arr))

# # Get List of csv files
# ls = os.listdir(data_path + 'train/')
# ls = [l for l in ls if l.endswith('.csv')]

# Placeholders for results
best_h_results = []

for file in list(paras.columns):
    train = pd.read_csv(data_path + 'train/' + file)
    spot = train.loc[:, spot_name]
    future = train.loc[:, future_name]
    u = ECDF(spot)(spot)
    v = ECDF(future)(future)

    kde_brr = KDE(spot, "Gaussian")
    kde_btc = KDE(future, "Gaussian")

    # load paras
    for C_name in Copulae:
        p = paras.loc[C_name, file]
        if (C_name == 't_Copula') or (C_name == 't_Copula_Capped'):
            Copulae[C_name].__init__(p, kde_brr, kde_btc, nu_lowerbound=2)
        else:
            Copulae[C_name].__init__(p, kde_brr, kde_btc)

        print(C_name, Copulae[C_name].paras)

    # Get Best h
    best_h = []
    for C_name in Copulae:
        np.random.seed(0)
        best_h.append(optimize_h(Copulae[C_name], k_arr, q_arr_ES, q_arr_VaR))
    best_h = pd.DataFrame(best_h)
    best_h.columns = ['Variance'] + ['ERM k=%i' % k for k in k_arr] + ['ES q=%.2f' % q for q in q_arr_ES] + [
        'VaR q=%.2f' % q for q in q_arr_VaR]
    best_h.index = Copulae_names
    best_h_results.append(best_h)

best_h_results_pd = pd.concat(dict(zip(list(paras.columns), best_h_results)), axis=1)
best_h_results_pd.apply(lambda x: np.clip(x, min_h, max_h))

l_arr = []
date_range_arr = []
for i, file in enumerate(list(paras.columns)):
    train = pd.read_csv(data_path + 'train/' + file)
    date_range = train.Date.iloc[-1] + ' to ' + train.Date.iloc[0]
    date_range_arr.append(date_range)

best_h_results_pd = pd.concat(dict(zip(list(paras.columns), best_h_results)), axis=1)
display_best_h = best_h_results_pd.copy()
display_best_h.columns.set_levels(date_range_arr, level=0, inplace=True)
display_best_h = display_best_h.reindex(sorted(display_best_h.columns), axis=1)

# create csv for further combine with NIG factor copula
h_arr = []
for i, copula in enumerate(list(best_h_results_pd.index)):
    h = best_h_results_pd.iloc[i:i+1,:].melt()
    h.columns = ['file', 'risk measure', 'OHR']
    h.loc[:, 'copula'] = copula
    h_arr.append(h)
best_h_mathematica = pd.concat(h_arr)


if calibration_method == "MLE":
    path = "../results/" + data_name + "/MLE/"
    if os.path.exists(path)==False:
        os.mkdir(path)
elif calibration_method == "MM":
    path = "../results/" + data_name + "/MM/"
    if os.path.exists(path)==False:
        os.mkdir(path)

if calibration_method == "MLE":
    path = "../results/" + data_name + "/MLE/"
    best_h_results_pd.to_hdf(path + 'best_h.h5', key='df', mode='w')
    display_best_h.to_html(path + "OHR.html")
    best_h_mathematica.to_csv(path + "OHR.csv")

elif calibration_method == "MM":
    path = "../results/" + data_name + "/MM/"
    best_h_results_pd.to_hdf(path + 'best_h.h5', key='df', mode='w')
    display_best_h.to_html(path + "OHR.html")
    best_h_mathematica.to_csv(path + "OHR.csv")

print('Done! Please find the OHR in ' + path)