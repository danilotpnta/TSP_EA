from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import Reporter
import numpy as np
import random
import csv

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

class TSP_problem:

	''' TSP Problem definition '''
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

	''' TSP Evolutionary Algorithm '''
	# Initialization by Population: Lamda individuals
	def initialize(self, lambdaa: int) -> np.ndarray:
		return np.array(list(map(lambda x: Individual.notinf(self), np.empty(lambdaa))))

	# Selection by k-tournament
	def selection(self, population: np.ndarray, k: int) -> Individual:
		# select k random ind from population
		selected = np.random.choice(population, k)
		ind_i = np.argmin(np.array( list(map( self.fitness , selected))))

		return selected[ind_i]

	# Recombination by Order Crossover (OX)
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

	# Mutation by Swap Operator
	def mutation(self, ind: Individual):
		# swaps two positions if rand [0,1) < 0.05
		if np.random.rand() < ind.alpha:
			i1 = random.randint(0, len(ind.order)-1)
			i2 = random.randint(0, len(ind.order)-1)
			ind.order[i1], ind.order[i2] = ind.order[i2], ind.order[i1]
		return

	''' Helper Functions '''
	# Calculates results for reporter
	def getResults(self, population):
		fitnesses = np.array(list( map(self.fitness, population) ))
		meanObjective = np.mean(fitnesses)
		bestIndex = np.argmin(fitnesses)
		bestObjective = fitnesses[bestIndex]
		bestSolution = population[bestIndex].order
		return meanObjective, bestObjective, bestSolution

	# Make bestSolution's city numbering start from 0
	def getCity0Start(self, order: np.ndarray) -> np.ndarray:
		city0Index = np.where(order == 0)[0][0]
		return np.concatenate((order[city0Index:],order[:city0Index]))

# Modify the class name to match your student number.
class r0916799:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	''' Helper functions '''
	# Read distance matrix from file.
	def getMatrix(self, filename):
		print("--- TSP Genetic Algorithm:", filename, "---")
		print("- Reading distance matrix...")
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()
		return distanceMatrix

	# Prints fitness values of populaiton before & after optimize
	def printResults(self, section, TSP, numSameFit, itr, meanObj, bestObj, bestSol):

		# limits how much of ind's order one can see in terminal
		maxIcanSee = 175
		space, Ndigits, sqrbrackets = 1, TSP.N, 2
		lengthStringOrder = space * Ndigits + len(str(Ndigits)) * Ndigits + sqrbrackets
		orderStr = np.array2string(TSP.getCity0Start(bestSol), max_line_width=lengthStringOrder)
		if lengthStringOrder > maxIcanSee:
			orderStr = orderStr[:maxIcanSee]
		# Print header & fitness values of initial population
		if section == 'init' :
			header = ["Cnvrg","Itr", "Mean Fitness", "Best Fitness", "Order"]
			data = [0, 0, meanObj, bestObj,orderStr]
			print("{: >6} {: >3} {: >15} {: >15} {: >}".format(*header))
			print("{: >6} {: >3} {: >15.3f} {: >15.3f} {: >}".format(*data))

		# Print fitness values from current iteration
		if section == 'inLoop' :
			data = [numSameFit, itr + 1, meanObj, bestObj, orderStr]
			print("{: >6} {: >3} {: >15.3f} {: >15.3f} {: >}".format(*data))

	# Open file to write results
	def saveData(self, section, itr, meanObj,bestObj):

		if section == 'init' :
			header = ['Iteration', 'MeanFit', 'BestFit']
			with open('result.csv', 'w', newline='') as file:
				writer = csv.writer(file)
				file.truncate()
				writer.writerow(header)
				writer.writerow([itr,meanObj, bestObj])
				file.close()

		if section == 'inLoop' :
			with open('result.csv', 'a', newline='') as file:
				writer = csv.writer(file)
				writer.writerow([itr, meanObj, bestObj])
				file.close()

	''' The evolutionary algorithm's main loop '''
	def optimize(self, filename):
		distanceMatrix = self.getMatrix(filename)
		p   = Parameters(lambdaa = 100, k = 6, its = 300)
		TSP = TSP_problem(distanceMatrix)

		print("- Initializing population...")
		population = TSP.initialize(p.lambdaa)

		print("- Running optimization...")
		meanObj, bestObj, bestSol = TSP.getResults(population)
		self.printResults('init', TSP, 0, 0, meanObj, bestObj, bestSol)
		self.saveData('init', 0, meanObj, bestObj)

		# variables for convergence test
		numSameFit, prevBestFit, itr = 0, 0, 0

		while(numSameFit < 50 and itr < p.its):
			# Create the offspring
			offspring = np.empty(p.mu, dtype = Individual)
			for jj in range(0, p.mu):
				parent1 	  = TSP.selection(population, p.k)
				parent2 	  = TSP.selection(population, p.k)
				offspring[jj] = TSP.recombination(parent1, parent2)
				TSP.mutation(offspring[jj])

			# Join offspring with original population
			joinedPopulation = np.concatenate((offspring, population))

			# k-tournament Elimination
			population = np.empty(p.lambdaa, dtype = Individual)
			for jj in range(0, p.lambdaa):
				population[jj] = TSP.selection(joinedPopulation, p.k)

			# Calculate fitness values from current iteration
			meanObj, bestObj, bestSol = TSP.getResults(population)
			self.printResults('inLoop', TSP, numSameFit, itr, meanObj, bestObj, bestSol)
			itr += 1

			# Call the reporter
			timeLeft = self.reporter.report(meanObj, bestObj, TSP.getCity0Start(bestSol))
			if timeLeft < 0:
				print("- Time expired!")
				break

			# ConvergenceTest
			if np.isclose(prevBestFit, bestObj, rtol = 1e-05):
				numSameFit += 1
			else:
				numSameFit = 0
			prevBestFit = bestObj

			# Open file for writing
			self.saveData('inLoop', itr, meanObj, bestObj)

		return 0

if __name__ == '__main__':
	a = r0916799()
	a.optimize('./tour50.csv')
