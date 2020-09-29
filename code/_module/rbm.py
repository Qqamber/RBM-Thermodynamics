# Restricted Boltzmann Machine(RBM) used to learn the underlying distribution of a data set



import math
import json
import numpy as np



class Rbm():
	'''
	restricted boltzmann machine class for training and sampling
	'''
	def __init__(self, visible_num, hidden_num):
		'''initialize the parameters'''
		# number of visible and hidden units
		self.v_num, self.h_num = visible_num, hidden_num

		# parameters to be learned
		self.weight_w = (1./math.sqrt(visible_num+hidden_num)) * ( np.random.random((hidden_num, visible_num)) - 0.5 )
		self.bias_v = np.zeros([visible_num, 1])
		self.bias_h = np.zeros([hidden_num, 1])


	def load(self, weight_w, bias_v, bias_h):
		'''load a trained model's parameters'''
		self.weight_w = np.array(weight_w)
		self.bias_v = np.array(bias_v).reshape((visible_num, 1))
		self.bias_h = np.array(bias_h).reshape((hidden_num, 1))

		assert np.shape(self.weight_w) == (len(self.bias_h), len(self.bias_v))


	def save(self, file_name):
		'''
		save the trained model in file_name
		file_name: string type, !!!end with .json
		'''
		# !!!convert arrays to lists
		para = {'w': self.weight_w.tolist(), 'v': self.bias_v.tolist(), 'h': self.bias_h.tolist()}
		with open(file_name, 'w') as save_file:
			json.dump(para, save_file)


	def train(self, data_set, batch_size, epoch, learning_rate, cd_step):
		'''
		train the restricted boltzmann machine, the learned parameters will be stored in attributes

		Args:
		data_set: D by self.v_num array(or list), D is the number of data
		batch_size: the size of each minibatch
		epoch: go through the whole data set 'epoch' times
		learning_rate: should be small, like 0.1
		cd_step: the step number for contrastive divergence
		'''
		print('-----------Training Engine-----------')

		for i in range(epoch):
			if i%10 == 0:
				print('\tTraining Process: The ' + str(i+1) + ' Epoch')

			data = data_set.copy()  # np.random.shuffle() will modify the original list in-place
			np.random.shuffle(data)
			batches = np.array_split(data, int(len(data_set)/batch_size))

			for batch in batches:
				dw, dv, dh = self._con_div(batch.T, cd_step)  # note: batch.T, see self.__con_div()
				self.weight_w += learning_rate * dw
				self.bias_v += learning_rate * dv
				self.bias_h += learning_rate * dh


	def sample(self, start_v, eq_step, num, uncor_step, file_name=''):
		'''
		sample from restricted boltzmann machine

		Args:
		start_v: self.v_num by 1 array(or list), [0, 1] valued
		eq_step: begin to sample after eq_step times Gibbs sampling
		num: number of samples we want
		uncor_step: in order to get uncorrelated datas, actually 1 is enough
		file_name: store the data in file_name(optional), end with .json

		Return:
		sample_set: a set of sampling datas, [0, 1] valued, array type
		'''
		print('-----------Sampling Engine-----------')

		# prepare the start point
		v = np.array(start_v).reshape(self.bias_v.shape)
		sample_set = []

		# dragged to detailed balance
		print("\tI'm looking for the detailed balanced position! Please wait!")
		for i in range(eq_step):
			h = self._sample_h(v)
			v = self._sample_v(h)
		print("\tDone!")

		# sampling data
		print("\tI'm collecting samples!")
		for i in range(num):
			for j in range(uncor_step):
				h = self._sample_h(v)
				v = self._sample_v(h)
			sample_set.append(v.reshape(self.v_num).tolist())
			if i%10000 == 0:
				print("\t" + str(i) + "/" + str(num))

		# store the data sampled (optional)
		if file_name:
			with open(file_name, 'w') as save_file:
				json.dump(sample_set, save_file)

		return np.array(sample_set)


	def _con_div(self, v_batch, step):
		'''
		k-step contrastive divergence(parallel style for a whole batch)

		Args:
		v_batch: self.v_num by batch_size array, contains a whole batch data
		step: Gibbs step, which is an integer

		Return:
		dw: self.h_num by self.v_num array, the increment of self.weight_w
		dv: self.v_num by 1 array, the increment of self.bias_v
		dh: self.h_num by 1 array, the increment of self.bias_h
		'''
		vs = v_batch.copy()  # use 'vs = v_batch' will affect v_batch itself!

		for i in range(step):
			hs = self._sample_h(vs)
			vs = self._sample_v(hs)

		h0_pro = self._sigmoid( self.bias_h + np.dot(self.weight_w, v_batch) )
		hk_pro = self._sigmoid( self.bias_h + np.dot(self.weight_w, vs) )

		dw = ( np.dot(h0_pro, v_batch.T) - np.dot(hk_pro, vs.T) ) / float(v_batch.shape[1])
		dv = np.mean(v_batch-vs, axis=1).reshape(self.bias_v.shape)
		dh = np.mean(h0_pro-hk_pro, axis=1).reshape(self.bias_h.shape)

		return dw, dv, dh


	def _sample_h(self, vs):
		'''
		sample hidden layers given visible layers(parallel style for a whole batch)

		assume that you have d values of visible layer, then you should arrange them as
		a numpy array with the shape: (self.v_num, d), i.e. each column is a visible layer value

		Args:
		vs: self.v_num by d array

		Return:
		hs: self.h_num by d array
		'''
		hs_arg = self.bias_h + np.dot(self.weight_w, vs)
		hs_pro = self._sigmoid(hs_arg)
		return 1. * ( np.random.random(hs_pro.shape) < hs_pro ).astype(int)


	def _sample_v(self, hs):
		'''
		sample visible layers given hidden layers(parallel style for a whole batch)

		assume that you have d values of hidden layer, then you should arrange them as
		a numpy array with the shape: (self.h_num, d), i.e. each column is a hidden layer value

		Args:
		hs: self.h_num by d array

		Return:
		vs: self.v_num by d array
		'''
		vs_arg = self.bias_v + np.dot(self.weight_w.T, hs)
		vs_pro = self._sigmoid(vs_arg)
		return 1. * ( np.random.random(vs_pro.shape) < vs_pro ).astype(int)


	def _sigmoid(self, x):
		'''sigmoid function'''
		# return 1. / (1. + np.exp(-x))  # np.exp(-x) may cause OverflowError
		return 0.5 * (1. + np.tanh(0.5 * x))  # a safer version
	