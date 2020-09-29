# Monte Carlo Simulation for the 2d Ising square lattice
# Model Hamiltonian: H = -J \sum_{<i,j>} Si * Sj with periodic boundary condition
# Boltzmann constant is set to 1



import math
import numpy as np



class IsingLattice():
	'''
	A class for 2d square lattice and do monte carlo simulation, 
	periodic boundary condition is assumed
	
	Args:
	length: linear size of lattice
	parity: 'up', 'down' or 'random'
	temperature: temperature of the system
	strength: coupling strength J
	'''
	def __init__(self, length, parity, temperature, strength):
		self.length = length
		self.temperature = temperature
		self.strength = strength
		self.configuration = self._generate_configuration(length, parity) # store spin configuration
		self.energy, self.magnetism = self._get_energy_magnetism() # store the current energy and magnetism


	def flip_spin(self):
		'''do a single spin flip'''
		po1 = np.random.randint(self.length) + 1
		po2 = np.random.randint(self.length) + 1
		energy_change = -2. * self._site_energy(po1, po2, self.strength)  # change of energy when flip a spin

		if np.random.random() < math.exp(-energy_change/self.temperature):  # base on Metropolis Hastings Algorithm
			self.energy += energy_change
			self.magnetism += -2. * self.configuration[po1, po2]
			self.configuration[po1, po2] = -self.configuration[po1, po2]
			if po1 == 1 or po1 == self.length or po2 == 1 or po2 == self.length:
				self._impose_pbc(self.configuration)
		else:
			pass


	def _impose_pbc(self, lattice):
		'''
		impose the periodic boundary condition for a given lattice configuration

		Args:
		lattice: a given lattice configuration with (L+2)*(L+2) size
		'''
		lattice[0] = lattice[-2]
		lattice[-1] = lattice[1]
		lattice[:, 0] = lattice[:, -2]
		lattice[:, -1] = lattice[:, 1]


	def _generate_configuration(self, length, parity='random'):
		'''
		generate a spin configuration imposed by periodic boundary condition
	
		Args:
		length: the linear size of the lattice
		parity: spin configuration of up/down/random type
		'''
		template = []
		if parity == 'random':
			template = [1., -1.]
		elif parity == 'up':
			template = [1.]
		elif parity == 'down':
			template = [-1.]
		else:
			print("Please check your parity input!")

		lattice_size = (length+2, length+2)
		raw_config = np.random.choice(template, size=lattice_size)
		self._impose_pbc(raw_config)
		return raw_config


	def _site_energy(self, row, column, strength):
		'''
		calculate the local energy for a single site contributed from the interaction with neighbors
	
		Args:
		row, column: position of the spin that you consider
		strength: value of the interaction strength J
		'''
		s = - strength * self.configuration[row, column] * (self.configuration[row-1, column] +
			self.configuration[row+1, column] + self.configuration[row, column-1] + self.configuration[row, column+1])
		return s


	def _get_energy_magnetism(self):
		'''calculate the total energy and magnetism of the current spin configuration'''
		e, m = 0., 0.
		for i in range(1, self.length+1):
			for j in range(1, self.length+1):
				e += self._site_energy(i, j, self.strength)
				m += self.configuration[i, j]
		return 0.5*e, m  # note the 1/2 factor for total energy