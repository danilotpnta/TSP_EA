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

	# simplified code and using np.empty
	def recombination(self, ind1: Individual, ind2: Individual) -> Individual:
		subset_indices = np.random.randint(low=0, high=self.N, size=2)
		low_index = np.min(subset_indices)
		high_index = np.max(subset_indices)
		subset_first_parent = np.array(ind1.order[low_index:high_index+1])

		offspring = np.empty(shape=self.N, dtype=int)
		rotated_ind2_order = np.concatenate((ind2.order[1:], (ind2.order[:1])))
		remaining_ind2 = np.setdiff1d(rotated_ind2_order, subset_first_parent, assume_unique=True)
		offspring[low_index:high_index+1] = subset_first_parent
		offspring[0:low_index] = remaining_ind2[0:low_index]
		offspring[high_index+1:self.N] = remaining_ind2[low_index:]

		return Individual(order=offspring, alpha=.05)


	def initialize(self, lambdaa: int) -> np.ndarray:
		return np.array(list(map(lambda x: Individual.notinf(self), np.empty(lambdaa))))


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

	@classmethod
	def notinf(cls, TSP: TSP_problem):
		order = np.empty(TSP.N, dtype=int)
		order[0] = 0
		citiesLeft = set(range(1,TSP.N))
		citiesThisRun = set(range(1,TSP.N))
		currCity = 0
		index = 1
		while index != TSP.N:
			nextCity = np.random.choice(tuple(citiesThisRun))
			citiesThisRun.remove(nextCity)

			# Search for a feasable next city
			if TSP.d_matrix[currCity,nextCity] != np.inf:
				order[index] = nextCity
				citiesLeft.remove(nextCity)
				currCity = nextCity
				citiesThisRun = set(citiesLeft) # Constructor needed, otherwise pointer to same memory space
				index += 1

			# No feasable next city
			if len(citiesThisRun) == 0 and index != TSP.N:

				if len(order) == TSP.N-1: 
					order[index] = list(citiesLeft)[0]
					index += 1
				else:
					randCity = np.random.choice(tuple(citiesLeft))
					order[index] = randCity
					citiesLeft.remove(randCity)
					currCity = nextCity
					citiesThisRun = set(citiesLeft) # Constructor needed, otherwise copy to memory
					index += 1

		return cls(np.array(order),0.05)


# Modify the class name to match your student number.
class r0123456:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		print("--- TSP Genetic Algorithm ---")

		print("- Reading distance matrix...")
		# Read distance matrix from file.
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		# Lambda | k-tournament | Iterations | mu (default = 2 * lambda)
		p = Parameters(100, 5, 300)
		TSP = TSP_problem(distanceMatrix)
		print("- Initializing population...")
		population = TSP.initialize(p.lambdaa)

		print("- Running optimization...")

		# Print initial fitness
		fitnesses = np.array(list(map(TSP.fitness, population)))
		
		# Long format
		#print(0, ": Mean fitness = ", np.mean(fitnesses), "\t Best fitness = ", np.min(fitnesses))

		# Short format
		print("{: >3} {: >15} {: >15}".format(*("i", "Mean Fitness", "Best Fitness")))
		print("{: >3} {: >15.3f} {: >15.3f}".format(*(0, np.mean(fitnesses), np.min(fitnesses))))

		# TODO: Determine best convergence test
		nbSameFit = 0
		prevBestFit = 0
		it = 0

		while( nbSameFit < 30 and it < p.its):

			# Create the offspring
			offspring = np.empty(p.mu, dtype=Individual)
			for jj in range(0, p.mu):
				parent1 = TSP.selection(population, p.k)
				parent2 = TSP.selection(population, p.k)
				offspring[jj] = TSP.recombination(parent1, parent2)
				TSP.mutation(offspring[jj])

			# Join the offspring with the original population
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
			print("{: >3} {: >15.3f} {: >15.3f}".format(*(it+1,meanObjective,bestObjective)))

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				print("- Time expired!")
				break

			it += 1

			# ConvergenceTest
			if (np.isclose(prevBestFit, bestObjective, rtol=1e-05)):
				nbSameFit += 1
			else:
				nbSameFit = 0
			prevBestFit = bestObjective

		return 0

a = r0123456()
a.optimize('./tour50.csv')

