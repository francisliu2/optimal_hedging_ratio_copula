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
from statsmodels.distributions.empirical_distribution import ECDF
import os

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
q_arr = config['q_arr']  # moment conditions for MM

# Parameters
if calibration_method == "MLE":
	paras = pd.read_json("../results/" + data_name + "/MLE/parameters.json")
elif calibration_method == "MM":
	paras = pd.read_json("../results/" + data_name + "/MM/parameters.json")


if calibration_method == 'MLE':
    result_path = "../results/"+data_name+"/MLE/"
elif calibration_method == 'MM':
    result_path = "../results/"+data_name+"/MM/"

# Get List of csv files
ls = os.listdir(data_path + 'train/')
ls = [l for l in ls if l.endswith('.csv')]


def hedging_effectiveness(rm, rs, rf, h):
	if rm.startswith('Variance'):
		rh = rs - h * rf
		return 1 - Variance(rh) / Variance(rs)

	elif rm.startswith('ERM'):
		k = float(rm[rm.find('=') + 1:])
		rh = rs - h * rf
		return 1 - ERM_estimate_trapezoidal(k, rh) / ERM_estimate_trapezoidal(k, rs)

	elif rm.startswith('ES'):
		q = float(rm[rm.find('=') + 1:])
		rh = rs - h * rf
		return 1 - ES(q, rh) / ES(q, rs)

	elif rm.startswith('VaR'):
		q = float(rm[rm.find('=') + 1:])
		rh = rs - h * rf
		return 1 - VaR(q, rh) / VaR(q, rs)

def wrapper_HE(rm, file, h, insample=True):
	if insample:
		data = pd.read_csv(data_path + 'train/' + file)
	else:
		data = pd.read_csv(data_path + 'test/' + file)
	rs = data.loc[:, spot_name]
	rf = data.loc[:, future_name]
	return hedging_effectiveness(rm, rs, rf, h)

OHR = pd.read_hdf(result_path+"best_h.h5")

OHR1 = OHR.reset_index()
OHR1 = pd.melt(OHR1, id_vars='index')
OHR1.columns = ['copula', 'file', 'risk_measure', 'h']
OHR1.loc[:, 'HE'] = OHR1.apply(lambda x: wrapper_HE(rm=x.risk_measure,
													file=x.file,
													h=x.h,
													insample=True), axis=1)

HEs = pd.pivot_table(OHR1, index='copula', columns=['file', 'risk_measure'], values='HE')
HEs = HEs.reindex(OHR.index)

RMs = np.unique(OHR.droplevel(0,1).columns)

fig, ax = plt.subplots(len(RMs), 1, figsize=(10,5*len(RMs)))
for i, name in enumerate(RMs):
    ax[i].boxplot(HEs.droplevel(0,axis=1).loc[:,name])
    ax[i].set_xticklabels(HEs.index)
    ax[i].set_title("In Sample Hedging Effectiveness of %s"%name)

plt.savefig(result_path+'In Sample Hedging Effectiveness.png', transparent=True)
HEs.to_hdf(result_path+'In Sample Hedging Effectiveness.h5', key='df', mode='w')
HEs.to_html(result_path+'In Sample Hedging Effectiveness.html')

OHR1 = OHR.reset_index()
OHR1 = pd.melt(OHR1, id_vars='index')
OHR1.columns = ['copula', 'file', 'risk_measure', 'h']
OHR1.loc[:, 'HE'] = OHR1.apply(lambda x: wrapper_HE(rm=x.risk_measure,
													file=x.file,
													h=x.h,
													insample=False), axis=1)

HEs = pd.pivot_table(OHR1, index='copula', columns=['file', 'risk_measure'], values='HE')
HEs = HEs.reindex(OHR.index)

fig, ax = plt.subplots(len(RMs), 1, figsize=(10,5*len(RMs)))
for i, name in enumerate(RMs):
    ax[i].boxplot(HEs.droplevel(0,axis=1).loc[:,name])
    ax[i].set_xticklabels(HEs.index)
    ax[i].set_title("Out of Sample Hedging Effectiveness of %s"%name)

plt.savefig(result_path+'Out of Sample Hedging Effectiveness.png', transparent=True)
HEs.to_hdf(result_path+'Out of Sample Hedging Effectiveness.h5', key='df', mode='w')
HEs.to_html(result_path+'Out of Sample Hedging Effectiveness.html')
print('Done! Please find the hedging effectiveness in ' + result_path)