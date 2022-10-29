from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import Reporter
import numpy as np
import random

@dataclass
class Parameters:
	lambdaa	: int					# Population size
	k		: int					# Tournament selection
	its		: int					# Number of iterations
	mu		: Optional[int] = None	# Offspring size

	# Checks after creation if mu is specified, otherwise default mu of double the population size
	def __post_init__(self):
		if self.mu is None:
			self.mu = self.lambdaa * 2

class TSP_problem:

	def __init__(self, d_matrix: np.ndarray):
		self.d_matrix = d_matrix
		self.N = np.size(d_matrix, 0)

	def fitness(self, ind: Individual) -> float:
		distance = 0

		for i in range(0, self.N-1):
			city_current = ind.order[i]
			city_next = ind.order[i+1]
			distance += self.d_matrix[city_current][city_next]

			if distance == np.inf:
				return distance

		distance += self.d_matrix[ind.order[-1]] [ind.order[0]] # Add last route from last to first element
		return distance

	# TODO: simplify code + np.empty arrays (Bixente)
	def recombination(self, ind1: Individual, ind2: Individual) -> Individual:
		subset_indices = np.random.randint(low=0, high=len(ind1.order), size=2)
		low_index = np.min(subset_indices)
		high_index = np.max(subset_indices)
		subset_first_parent = list(ind1.order[low_index:high_index+1])

		offspring = []

		# rotate ind2's order since in the textbook the second list starts from the
		# second crossover point
		rotated_ind2_order = np.concatenate((ind2.order[1:], (ind2.order[:1])))

		# j is the index of ind2 that marks the element being placed in the offspring
		j = 0
		# i is the index of the offspring where an element of ind2 is being placed
		for i in range(0, low_index):
			ind2_value = rotated_ind2_order[j]

			# if the element in ind2 is already in the random subset of the first parent,
			# then skip that element and iterate to the next one
			while ind2_value in subset_first_parent and j < len(ind1.order)-1:
				j += 1
				ind2_value = rotated_ind2_order[j]

			if j >= len(ind1.order):
				break

			offspring.append(ind2_value)
			j += 1

		# now add the subset from the first parent
		offspring += subset_first_parent

		# now do the same loop on ind2 for the remainder of the offspring indices
		for i in range(high_index+1, len(ind1.order)):
			ind2_value = rotated_ind2_order[j]

			# if the element in ind2 is already in the random subset of the first parent,
			# then skip that element and iterate to the next one
			while ind2_value in subset_first_parent and j < len(ind1.order)-1:
				j += 1
				ind2_value = rotated_ind2_order[j]

			if j >= len(ind1.order):
				break

			offspring.append(ind2_value)
			j += 1

		return Individual(order=np.array(offspring), alpha=.05)


	def initialize(self, lambdaa: int) -> np.ndarray:
		return np.array(list(map(lambda x: Individual.random(self), np.empty(lambdaa))))


	# Selection by k-tournament
	def selection(self, population: np.ndarray, k: int) -> Individual:
		selected = np.random.choice(population, k)
		ind_i = np.argmin(np.array( list(map( self.fitness , selected))))

		return selected[ind_i]


	def mutation(self, ind: Individual):
		# swaps two positions if rand [0,1) < 0.05
		if np.random.rand() < ind.alpha:
			i1 = random.randint(0, len(ind.order)-1)
			i2 = random.randint(0, len(ind.order)-1)
			ind.order[i1], ind.order[i2] = ind.order[i2], ind.order[i1]
		return


class Individual:
	def __init__(self, order: np.ndarray, alpha: float):
		self.order = order
		self.alpha = alpha

	@classmethod
	def random(cls, TSP: TSP_problem):
		order = np.arange(TSP.N)
		np.random.shuffle(order)
		return cls( order, 0.05)

'''
class Individual:
	def __init__(self, order=None, alpha=None):
		if order == None:
			self.alpha = 0.05
			self.order = np.arange( len(TSP.values))
			np.random.shuffle(self.order)
		else:
			self.alpha = alpha
			self.order = order
'''


# Modify the class name to match your student number.
class r0123456:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

<<<<<<< HEAD
		# Lambda | k-tournament | Iterations | mu (default = 2 * lambda)
		p = Parameters(100, 5, 300)
		TSP = TSP_problem(distanceMatrix)
		population = TSP.initialize(p.lambdaa)
		"""selection = TSP.selection(population, 3)


		ind = Individual( [0, 1, 2, 3, 4, 5, 6, 7], 0.5)
		ind2 = Individual([7, 6, 1, 3, 4, 0, 5, 2], 0.5)
		TSP.recombination(ind, ind2)"""

		# Your code here.
		fitnesses = np.array(list(map(TSP.fitness, population)))
		#print(0, ": Mean fitness = ", np.mean(fitnesses), "\t Best fitness = ", np.min(fitnesses))

		# Short format
		print("{: >3} {: >15} {: >15}".format(*("i", "Mean Fitness", "Best Fitness")))
		print("{: >3} {: >15.3f} {: >15.3f}".format(*(0, np.mean(fitnesses), np.min(fitnesses))))
=======
		# Potential Parameter values
		p = Parameters(100, 5, 100)
		TSP = TSP_problem(distanceMatrix)
		population = TSP.initialize(p.lambdaa)

		# print initial fitness
		fitnesses = list(map(TSP.fitness, population))
		print(0, ": Mean fitness = ", np.mean(fitnesses), "\t Best fitness = ", np.min(fitnesses))
>>>>>>> a96bf0c10d860a8539780e2cd29d873fb971cc9e

		# TODO: Determine best convergence test
		#yourConvergenceTestsHere = False
		#while( yourConvergenceTestsHere ):
		for i in range(0, p.its):
<<<<<<< HEAD
=======
			offspring = []
			# TODO: how many offspring should we create in each iteration?
			for jj in range(0, p.its):
				ind1 = TSP.selection(population, p.k)
				ind2 = TSP.selection(population, p.k)
				offspring.append(TSP.recombination(ind1, ind2))
				offspring[jj] = TSP.mutation(offspring[jj])
>>>>>>> a96bf0c10d860a8539780e2cd29d873fb971cc9e

			# Create the offspring
			offspring = np.empty(p.mu, dtype=Individual)
			for jj in range(0, p.mu):
				parent1 = TSP.selection(population, p.k)
				parent2 = TSP.selection(population, p.k)
				offspring[jj] = TSP.recombination(parent1, parent2)
				TSP.mutation(offspring[jj])

			# Joint the offspring with the original population
			joinedPopulation = np.concatenate((offspring, population))

			# Elimination uses same method as selection (k-tournament)
			# TODO: possible to change this to map
			population = np.empty(p.lambdaa, dtype=Individual)
			for jj in range(0, p.lambdaa):
				population[jj] = TSP.selection(joinedPopulation, p.k)

			fitnesses = np.array(list(map(TSP.fitness, population)))
			meanObjective = np.mean(fitnesses)
			bestIndex = np.argmin(fitnesses)
			bestObjective = fitnesses[bestIndex]

			bestSolution = population[bestIndex].order

			# Long format
			#print(i, ": Mean fitness = ", meanObjective, "\t Best fitness = ", bestObjective, ": Best path = ", bestSolution)
			
			# Short format
			print("{: >3} {: >15.3f} {: >15.3f}".format(*(i+1,meanObjective,bestObjective)))

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				print("Time expired")
				break

		# Your code here.
		return 0

a = r0123456()
a.optimize('./tour50.csv')


# ind = Individual([0, 1, 2, 3],0.5)
# a.fitness(ind)
