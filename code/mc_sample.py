import math
import json
import numpy as np

from _module import mc



# linear size of the system, e.g. N = L**2 is the site number
L = 8

# set the energy unit, e.g. J = 1. for ferromagenetism and J = -1. for anti-ferromagenetism
J = 1.

# one MC time: ensure that each spin will be chosen once on average within this 'time'
mct = L**2

# how many MC times to be taken before sampling
mc_step = 10000

# number of data we pick and uncorrelated time
num_each_temp = 200000
uncorrelate_step = 1  # it should be larger with larger lattice

# lists used to store monte carlo data points
configuration_data = []

# temperature list and number
temperature_list = np.linspace(1., 3.54, 5)
temperature_num = len(temperature_list)


# monte carlo for different temperatures
for temperature in temperature_list:

	# recover the parity symmetry
	for parity in ['up', 'down']:

		ising_lattice = mc.Lattice(L, 'pbc', parity, temperature, J)

		# find the detailed balanced position
		print("I'm finding the detailed balance position, for temperature: " + str(temperature) + ', PARITY: ' + parity + '\n')
		for i in range(mc_step):
			if i % 1000 == 0:
				print("\tThe " + str(i) + " Monte Carlo Time For temperature " + str(temperature) + ", PARITY: " + parity)
			for j in range(mct):
				ising_lattice.flip_spin()

		# sample after equilibrium
		print("\n\tStart to sample at temperature " + str(temperature) + ", PARITY: " + parity + ' -_-\n')
		for i in range(int(num_each_temp/2)):
			for j in range(uncorrelate_step):  # sample uncorrelated points
				for k in range(mct):
					ising_lattice.flip_spin()
			configuration_data.append(list(np.reshape(ising_lattice.configuration[1:-1, 1:-1], L**2)))
		print("\tEnd sampling for temperature " + str(temperature) + ", PARITY: " + parity + ' ^_^\n')


print("Get Total " + str(len(configuration_data)) + " configurations! Data is saving, please wait! @_@")

# save monte carlo sampled data
save_mc_data = [configuration_data, temperature_list.tolist(), num_each_temp]
with open('data/mc_data.json', 'w') as mc_file:
	json.dump(save_mc_data, mc_file)

print("Done!")