# visualization

import math
import json
import numpy as np
from matplotlib import pyplot as plt

# get the average magnetism for a bunch of data
def mag_temp(x):
	m = list(map(lambda y: abs(np.mean(y)), x))
	m = np.sum(m)/len(m)
	return m

with open('data/mc_data.json', 'r') as mc_file:
	[mc_data, temperature_list, _] = json.load(mc_file)

with open('data/rbm_data.json', 'r') as rbm_file:
	[rbm_data, _] = json.load(rbm_file)

mc_plot_data = np.array_split( mc_data, len(temperature_list) )
mc_plot_data = list(map(mag_temp, mc_plot_data))
rbm_plot_data = 2. * np.array(rbm_data) - 1.
rbm_plot_data = list(map(mag_temp, rbm_plot_data))

plt.plot(temperature_list, mc_plot_data, c='b', marker="+", label="Monte Carlo", linewidth=1.5, markersize=7)
plt.plot(temperature_list, rbm_plot_data, c='r', marker="x", label="RBM", linewidth=1.5, markersize=7)

plt.title("Magnetism From Monte Carlo and RBM")
plt.xlabel("Temperature")
plt.ylabel("Magnetism")
plt.xlim((1., 3.54))
plt.legend(loc='best')

plt.show()
