# Monte Carlo Simulation for the 2d Ising square lattice
# Model Hamiltonian: H = -J Si * Sj with periodic boundary condition
# Boltzmann constant is set to 1



import math
import numpy as np



class Lattice():
	'''
	A class for 2d square lattice and do monte carlo simulation
	
	Args:
	length: linear size of lattice
	boundary: 'pbc' or 'obc'
	direction: 'up' or 'down' or 'random'
	temperature: temperature of the system
	strength: coupling strength J
	'''
	def __init__(self, length, boundary, direction, temperature, strength):
		self.length = length
		self.boundary = boundary
		self.temperature = temperature
		self.strength = strength
		self.configuration = self.__generate_configuration(length, boundary, direction)
		self.energy, self.magnetism = self.__energy_magnetism()


	def flip_spin(self):
		'''do a single monte carlo step'''
		po1 = np.random.randint(self.length) + 1
		po2 = np.random.randint(self.length) + 1
		energy_change = -2. * self.__site_energy(po1, po2, self.strength)  # change of energy when flip a spin

		if np.random.random() < math.exp(-energy_change/self.temperature):  # base on Metropolis Hastings Algorithm
			self.energy += energy_change
			self.magnetism += -2. * self.configuration[po1, po2]
			self.configuration[po1, po2] = -self.configuration[po1, po2]
			if po1 == 1 or po1 == self.length or po2 == 1 or po2 == self.length:
				self.__impose_pbc(self.configuration)
		else:
			pass


	def __impose_pbc(self, lattice):
		'''
		impose the periodic boundary condition for a given lattice configuration

		Args:
		lattice: a given lattice configuration
		'''
		lattice[0] = lattice[-2]
		lattice[-1] = lattice[1]
		lattice[:, 0] = lattice[:, -2]
		lattice[:, -1] = lattice[:, 1]


	def __generate_configuration(self, length, condition='pbc', direction='random'):
		'''
		generate a spin configuration imposed by obc or pbc
	
		Args:
		length: the linear size of the lattice
		condition: periodic or open boundary condition
		direction: spin configuration of up/down/random type
		'''
		template = []
		if direction == 'random':
			template = [1., -1.]
		elif direction == 'up':
			template = [1.]
		elif direction == 'down':
			template = [-1.]
		else:
			print("Please check your direction type!")

		if condition == 'obc':
			lattice_size = (length, length)
			raw_config = np.random.choice(template, size=lattice_size)
			return raw_config
		elif condition == 'pbc':
			lattice_size = (length+2, length+2)
			raw_config = np.random.choice(template, size=lattice_size)
			self.__impose_pbc(raw_config)
			return raw_config
		else:
			print("Please check your boundary condition!")


	def __site_energy(self, row, column, strength):
		'''
		calculate the local energy for a single site contributed by the interaction with neighbors
	
		Args:
		row, column: position of the spin that you consider
		configuration: the lattice configuration
		strength: value of the interaction strength J
		'''
		s = - strength * self.configuration[row, column] * (self.configuration[row-1, column] +
			self.configuration[row+1, column] + self.configuration[row, column-1] + self.configuration[row, column+1])
		return s


	def __energy_magnetism(self):
		'''calculate the total energy and magnetism of the self.configuration'''
		s, m = 0., 0.
		for i in range(1, self.length+1):
			for j in range(1, self.length+1):
				s += self.__site_energy(i, j, self.strength)
				m += self.configuration[i, j]
		return 0.5*s, m  # note the 1/2 factor for total energy