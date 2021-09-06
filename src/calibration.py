from copulae1 import *
from KDEs import *
from toolbox import *
import os
import json
import argparse
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="input config json file path")
args = parser.parse_args()
config_path = args.config
with open(config_path) as f:
	config = json.load(f)

# Data source
data_name = config['data_name']
data_path = "../processed_data/"+data_name+"/"
spot_name = config['spot_name']
future_name = config['future_name']

# Calibration Method
calibration_method = config['calibration_method'] # MM or MLE
if calibration_method =='MM':
	q_arr = config['q_arr'] # moment conditions for MM

# Gaussian, t_Copula, Clayton, Frank, Gumbel, Plackett, Gaussian mix Indep
C1  = Gaussian(dict(rho=0.9),       Law_RS=stats.norm, Law_RF=stats.norm)
C2  = t_Copula(dict(rho=0.1, nu=4), Law_RS=stats.norm, Law_RF=stats.norm, nu_lowerbound=2)
C2c = t_Copula(dict(rho=0.1, nu=4), Law_RS=stats.norm, Law_RF=stats.norm, nu_lowerbound=4)
C3  = Clayton(dict(theta=0.1),      Law_RS=stats.norm, Law_RF=stats.norm)
C4  = Frank(dict(theta=0.1),        Law_RS=stats.norm, Law_RF=stats.norm)
C5  = Gumbel(dict(theta=3),         Law_RS=stats.norm, Law_RF=stats.norm)
C6  = Plackett(dict(theta=10),      Law_RS=stats.norm, Law_RF=stats.norm)
C7  = Gaussian_Mix_Independent(dict(rho=.5,p=0.7),Law_RS=stats.norm, Law_RF=stats.norm)
C8  = rot180Gumbel(dict(theta=3), Law_RS=stats.norm, Law_RF=stats.norm)

Copulae_names = ['Gaussian', 't_Copula', 't_Copula_Capped', 'Clayton', 'Frank', 'Gumbel', 'Plackett', 'Gauss Mix Indep', 'rotGumbel']
Copulae_arr = [C1, C2, C2c, C3, C4, C5, C6, C7, C8]
Copulae = dict(zip(Copulae_names, Copulae_arr))

# Get List of csv files
ls = os.listdir(data_path + 'train/')
ls = [l for l in ls if l.endswith('.csv')]

# Placeholders for results
paras_results = []
likelihood_results = []

for file in ls:
	# Calibration
	train = pd.read_csv(data_path + 'train/' + file)
	spot = train.loc[:, spot_name]
	future = train.loc[:, future_name]
	u = ECDF(spot)(spot)
	v = ECDF(future)(future)

	kde_brr = KDE(spot, "Gaussian")
	kde_btc = KDE(future, "Gaussian")

	for C_name in Copulae:
		Copulae[C_name].Law_RS = kde_brr
		Copulae[C_name].Law_RF = kde_btc

	paras = []
	likelihood = []
	best_h = []
	for C_name in Copulae:
		if calibration_method == "MLE":
			Copulae[C_name].canonical_calibrate(u, v)

		elif calibration_method == "MM":
			Copulae[C_name].mm_calibrate(u, v, q_arr)

		print(C_name, 'is done.\n')

	for C_name in Copulae:
		paras.append((C_name, Copulae[C_name].paras))

	for C_name in Copulae:
		ln = Copulae[C_name].dependency_likelihood(u, v)
		likelihood.append((C_name, ln))

	paras_results.append(paras)
	likelihood_results.append(likelihood)

# Prettify the result
c_arr = []
date_range_arr = []
for i, file in enumerate(ls):
	train = pd.read_csv(data_path + 'train/' + file)
	date_range = train.Date.iloc[-1] + ' to ' + train.Date.iloc[0]
	date_range_arr.append(date_range)

	c = pd.DataFrame(paras_results[i])
	c.index = c.iloc[:, 0]
	c = pd.DataFrame(c.iloc[:, 1])
	c_arr.append(c)

paras_results_pd = pd.concat(dict(zip(ls, c_arr)), axis=1)
paras_results_pd.columns = paras_results_pd.columns.droplevel(1)
paras_results_pd.index.name = None

l_arr = []
date_range_arr = []
for i, file in enumerate(list(paras_results_pd.columns)):
	train = pd.read_csv(data_path + 'train/' + file)
	date_range = train.Date.iloc[-1] + ' to ' + train.Date.iloc[0]
	date_range_arr.append(date_range)

	c = pd.DataFrame(likelihood_results[i])
	c.index = c.iloc[:, 0]
	c = pd.DataFrame(c.iloc[:, 1])
	l_arr.append(c)

likelihood_results_pd = pd.concat(dict(zip(ls, l_arr)), axis=1)
likelihood_results_pd.columns = likelihood_results_pd.columns.droplevel(1)
likelihood_results_pd.index.name = None

# Save results as html for display
display_paras = paras_results_pd.copy()
display_paras.columns = date_range_arr
display_paras = display_paras.reindex(sorted(display_paras.columns), axis=1)

display_likelihood = likelihood_results_pd.copy()
display_likelihood.columns = date_range_arr
display_likelihood = display_likelihood.reindex(sorted(display_likelihood.columns), axis=1)

# Save files to designated directories
if not os.path.exists("../results/" + data_name):
	print("Create new folder for results")
	os.mkdir("../results/" + data_name)
	os.mkdir("../results/" + data_name + "/MLE")
	os.mkdir("../results/" + data_name + "/MM")

if calibration_method == "MLE":
	path = "../results/" + data_name + "/MLE/"
	paras_results_pd.to_json(path + "parameters.json")
	likelihood_results_pd.to_json(path + "likelihood.json")

	display_paras.to_html(path + "paras.html")
	display_likelihood.to_html(path + "likelihood.html")

elif calibration_method == "MM":
	path = "../results/" + data_name + "/MM/"
	paras_results_pd.to_json(path + "parameters.json")
	likelihood_results_pd.to_json(path + "likelihood.json")

	display_paras.to_html(path + "paras.html")
	display_likelihood.to_html(path + "likelihood.html")

print('Done')