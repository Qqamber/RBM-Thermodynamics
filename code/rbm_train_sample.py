import math
import json
import numpy as np

from _module import rbm



# load the data sampled from monte carlo
print("I'm loading data!!!\t -_-")
with open('data/mc_data.json', 'r') as mc_file:
	[mc_data, temperature_list, _] = json.load(mc_file)

# linear size of the system, e.g. N = L**2 is the site number
L = 8

# number of units in visible and hidden layer
v_num = L**2
h_num = 32

# batch size, epoch, learning rate and cd order
batch_size = 200
epoch = 100
lr = 0.0075
cd_order = 5

# # how many gibbs steps to be taken before sampling
eq_step = 10000

# number of data we pick for each temperature
num_each_temp = 200000

# prepare data
train_data = np.array_split( mc_data, len(temperature_list) )
train_data = 0.5 * ( np.array(train_data) + 1. )

rbm_data = []
temperature_iter = iter(temperature_list)

# RBM training and sampling
for data in train_data:

	print("\nRBM Start For Temperature: " + str(next(temperature_iter)) + '\n')
	
	model = rbm.Rbm(v_num, h_num)
	model.train(data, batch_size, epoch, lr, cd_order)
	rbm_sample_data = model.sample(np.random.choice([0., 1.], v_num), eq_step, num_each_temp, 1)
	rbm_data.append(rbm_sample_data.tolist())

print("\nSaving RBM's Sampling data! >_<\n")
save_rbm_data = [rbm_data, temperature_list]
with open('data/rbm_data.json', 'w') as rbm_file:
	json.dump(save_rbm_data, rbm_file)

print("Done! @_@")