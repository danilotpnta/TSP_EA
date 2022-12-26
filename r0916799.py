from __future__ import annotations
from typing import Optional
from numba import jit

import time
import Reporter
import numpy as np
import random
import csv

class Parameters:
	def __init__(self, lambdaa, k, its, mu = None,
				 alpha = None, lso_ON = None, lso_percen = None,
				 greedy_percen = None):

		self.lambdaa = lambdaa						# Population size
		self.mu 	 = self.lambdaa * 2				# Offspring size
		self.k	     = k							# Tournament selection
		self.its	 = its							# Number of iterations
		self.alpha   = 0.05							# Probability of mutation
		self.lso_ON  = True							# Apply lso if True
		self.lso_percen = 0.15						# Apply lso w/ default perce
		self.greedy_percen = 0

		# PARAMETERS set by USER
		if mu is not None: self.mu = mu 			# Offspring size specified
		if alpha is not None: self.alpha = alpha 	# Prob. mutation specified
		if lso_ON is not None: self.lso_ON = lso_ON # Activate lso as specified
		if lso_percen is not None: self.lso_percen = lso_percen # Apply lso w/ percen
		if greedy_percen is not None: self.greedy_percen = greedy_percen

class Individual:
	def __init__(self, lenOrder, order = None, alpha = None):
		self.order = np.arange(lenOrder)
		np.random.shuffle(self.order)
		self.alpha =  0.05
		self.edges = getEdges_jit(self.order)

		if order is not None:
			self.order = order
		if alpha is not None:
			self.alpha = alpha

''' Faster functions for TSP_problem'''

@jit(nopython=True)
def cost(d_matrix, city_current: int, city_next: int) -> float:
	distance = d_matrix[city_current][city_next]
	return distance

@jit(nopython=True)
def fitness_jit(d_matrix, order:np.ndarray) -> float:
	distance = 0
	for i in range(0, len(order)-1):
		city_current = order[i]
		city_next = order[i+1]
		distance += d_matrix[city_current][city_next]

		if distance == np.inf:
			return distance
	# Add last route from last to first element
	distance += d_matrix[order[-1]] [order[0]]
	return distance

@jit(nopython=True)
def lso_lecture(d_matrix, order: np.ndarray) -> np.ndarray:
	bestFitness = fitness_jit(d_matrix, order)
	copyInd = np.copy(order)
	bestIndex = (0,0)
	lenInd = len(order)

	# Get index's if new route has better fitValue
	for i in range(1, lenInd - 1):
		for j in range( i + 1, lenInd):
			copyInd[i] = copyInd[j]
			copyInd[j] = copyInd[i]

			tempFitness = fitness_jit(d_matrix, copyInd)
			if tempFitness < bestFitness:
				bestIndex = (i, j)
				bestFitness = tempFitness

	i , j  = bestIndex
	if i == 0 and j == 0: return order
	order[i] = order[j]
	order[j] = order[i]
	return order

@jit(nopython=True)
def lso_expensive(d_matrix, order: np.ndarray) -> np.ndarray:
	bestFitness = fitness_jit(d_matrix, order)
	bestIndex = (0,0)
	lenInd = len(order)

	a = cost(d_matrix, order[-1], order[0]) # costFromLastCitytoStart
	if a == np.inf: return order
	costFromLastCitytoStart = a

	# Get index's if new route has better fitValue
	tempSumFirst = 0
	for i in range(1,lenInd - 2):
		tempSumFirst += cost(d_matrix, order[i-1], order[i])

		tempSumMiddle = 0
		for j in range(i + 2, lenInd):
			tempSumMiddle += cost(d_matrix, order[j-1], order[j-2])
			if tempSumMiddle == np.inf:
				break

			''' Get costs of new links in tempOrder i.e
			originalOrder: [1, 2, 3, 4, 5, 6, 7, 8, 9]
			tempOrder    : [1, 2, 5, 4, 3, 6, 7, 8, 9]
			newCostLink:         ^1       ^2
			'''
			cost1Link = cost(d_matrix, order[i-1], order[j-1])
			cost2Link = cost(d_matrix, order[i], order[j])

			tempSumLast = fitness_jit(d_matrix, copyInd[j:])
			tempFitness = tempSumFirst + cost1Link + tempSumMiddle \
						+ cost2Link + tempSumLast + costFromLastCitytoStart

			if tempFitness < bestFitness:
				bestIndex = (i, j)
				bestFitness = tempFitness

	i , j  = bestIndex
	if i == 0 and j == 0: return order
	orderImproved = np.copy(order)
	orderImproved[i:j] = np.flip(orderImproved[i:j])
	return orderImproved

@jit(nopython=True)
def lso_2_opt(d_matrix, order: np.ndarray) -> np.ndarray:
	bestFitness = fitness_jit(d_matrix, order)
	bestIndex = (0,0)
	lenInd = len(order)

	# Stores summed costs front-end: sumFirst & end-front: sumFinal
	sumsFirst, sumsFinal = np.zeros(lenInd), np.zeros(lenInd)
	a = cost(d_matrix, order[-1], order[0]) # costFromLastCitytoStart
	if a == np.inf: return order
	sumsFinal[-1] = a

	# Keep adding costs a: costFrontCitytoEnd, b: costEndCitytoFront
	for i  in range(1, lenInd - 1 ):
		a = cost(d_matrix, order[i-1], order[i])
		b = cost(d_matrix, order[lenInd - 1 - i], order[lenInd - i])
		if a == np.inf or b == np.inf: return order
		sumsFirst[i] = sumsFirst[i - 1] + a
		sumsFinal[lenInd - 1 - i] = sumsFinal[lenInd - i] + b

	# Get index's if new route has better fitValue
	for i in range(1,lenInd - 2):
		tempSumFirst = sumsFirst[i - 1]
		if tempSumFirst > bestFitness:
			break

		tempSumMiddle = 0
		for j in range(i + 2, lenInd):
			tempSumMiddle += cost(d_matrix, order[j-1], order[j-2])
			if tempSumMiddle == np.inf:
				break

			# early stop
			earlyFitness = tempSumFirst + tempSumMiddle
			if earlyFitness > bestFitness:
				continue

			''' Get costs of new links in tempOrder i.e
			originalOrder: [1, 2, 3, 4, 5, 6, 7, 8, 9]
			tempOrder    : [1, 2, 5, 4, 3, 6, 7, 8, 9]
			newCostLink:         ^1       ^2
			'''
			cost1Link = cost(d_matrix, order[i-1], order[j-1])
			cost2Link = cost(d_matrix, order[i], order[j])

			tempFitness = tempSumFirst + cost1Link + \
						  tempSumMiddle + cost2Link + sumsFinal[j]

			if tempFitness < bestFitness:
				bestIndex = (i, j)
				bestFitness = tempFitness

	i , j  = bestIndex
	if i == 0 and j == 0: return order
	orderImproved = np.copy(order)
	orderImproved[i:j] = np.flip(orderImproved[i:j])
	return orderImproved

# @jit(nopython=True)
def getEdges_jit(order:np.ndarray) -> np.ndarray:
	edges = []
	currentCity = order[0]

	for nextCity in order[1:]:
		edges.append((currentCity, nextCity))
		currentCity = nextCity
	lastToStart = (order[-1], order[0])
	edges.append(lastToStart)

	return edges

class TSP_problem:

	def __init__(self, d_matrix: np.ndarray, p):
		self.d_matrix = d_matrix
		self.N = np.size(d_matrix, 0)
		self.p = p

		# -- INITIALIZATION
		# self.initialize = self.initialize_Randmly
		# self.initialize = self.initialize_validPath
		# self.initialize = self.initialize_NN
		# self.initialize = self.initialize_NN_advanced
		self.initialize = self.initialize_joined

		# -- SELECTION
		self.selection = self.k_tourament

		# -- RECOMBINATION
		self.recombination = self.OrderCrossover

		# -- MUTATION
		# self.mutation = self.mutationSwap
		self.mutation = self.mutationShuf

		# -- ELIMINATION
		# self.elimination = self.k_tourament
		self.elimination = self.sharedElimination


	def fitness(self, ind: Individual) -> float:
		distance = 0
		for i in range(0, self.N-1):
			city_current = ind.order[i]
			city_next = ind.order[i+1]
			distance += self.d_matrix[city_current][city_next]

			if distance == np.inf:
				return distance

		distance += self.d_matrix[ind.order[-1]] [ind.order[0]]
		return distance

# ------------------------------------------------------------------------------
# INITIALIZATION
	def initialize_joined(self, lambdaa: int) -> List[Individual]:
		numGreedyInd = round(lambdaa * self.p.greedy_percen) # __% is NN
		randInd = lambdaa - numGreedyInd
		print(f'number randInd: {randInd}  &  number greedyInd: {numGreedyInd}')
		print(f'computing inds...')
		population = []
		for _ in range(randInd):
			ind = Individual(self.N, order = self.validPath(lambdaa), alpha = max(0.04, 0.70+0.05*random.random()))
			population.append(ind)

		for _ in range(numGreedyInd):
			ind = Individual(self.N, order = self.NN(lambdaa), alpha = max(0.04, 0.10+0.05*random.random()))
			population.append(ind)

		print(f'Population Size: {len(population)}')
		return np.array(population)

	def initialize_Randmly(self, lambdaa: int):
		population = list()
		for _ in range(lambdaa):
			ind = Individual(self.N, alpha = max(0.04, 0.70+0.05*random.random()))
			population.append(ind)
		return np.array(population)

	def initialize_validPath(self, lambdaa: int) -> np.ndarray:
		population = list()
		for _ in range(lambdaa):
			ind = Individual(self.N, order = self.validPath(lambdaa), alpha = max(0.04, 0.70+0.05*random.random()))
			population.append(ind)
		return np.array(population)

	def initialize_NN(self, lambdaa: int) -> np.ndarray:
		population = list()
		for _ in range(lambdaa):
			ind = Individual(self.N, order = self.NN(lambdaa), alpha = max(0.04, 0.10+0.05*random.random()))
			population.append(ind)
		return np.array(population)

	def initialize_NN_advanced(self, lambdaa: int) -> np.ndarray:
		population = list()
		for i in range(lambdaa):
			nd = Individual(self.N, order = self.NN_advanced(lambdaa), alpha = max(0.04, 0.10+0.05*random.random() ))
			if ind == False:
				i -= 1
				continue
			population.append(ind)
		return np.array(population)

	# Creates a valid random Ind
	def validPath(self, lambdaa):

		repeatAgain = True
		while repeatAgain:
			startTime = time.time()
			order = np.negative(np.ones((self.N), dtype=int))
			start = np.random.choice(range(0,self.N))
			order[0] = start
			citiesLeft = set(range(0,self.N)) - set([start])
			citiesThisRun = set(range(0,self.N)) - set([start])
			currCity = start
			index = 1

			while index != self.N:
				if time.time() - startTime > 2.0:
					break

				nextCity = np.random.choice(tuple(citiesThisRun))
				citiesThisRun.remove(nextCity)

				if self.d_matrix[currCity ,nextCity] != np.inf:
					order[index] = nextCity
					citiesLeft.remove(nextCity)
					currCity = nextCity
					citiesThisRun = set(citiesLeft)
					index += 1

				# No feasable next city
				if len(citiesThisRun) == 0 and index != self.N:
					if len(order) == self.N-1:
						order[index] = list(citiesLeft)[0]
						index += 1
					else:
						randCity = np.random.choice(tuple(citiesLeft))
						order[index] = randCity
						citiesLeft.remove(randCity)
						currCity = nextCity
						citiesThisRun = set(citiesLeft)
						index += 1

			a = fitness_jit(self.d_matrix, order) == np.Inf
			if (self.d_matrix[order[-1] ,order[0]] == np.inf) | (a):
				# print('found inf in ValidPath: ')
				repeatAgain = True
			else:
				repeatAgain = False

		return np.array(order)

	# Use nearestNeighbours to create Ind
	def NN(self, lambdaa):
		repeatAgain = True

		while repeatAgain:
			startTime = time.time()

			order = np.negative(np.ones((self.N), dtype=int))
			start = np.random.choice(range(0,self.N))
			order[0] = start
			citiesLeft = set(range(0,self.N)) - set([start])
			index = 1
			numBadGuys = 0

			while index != self.N:
				if time.time() - startTime > 2.0:
					break

				currCity = order[index-1]
				nearestNeighbour = None
				bestCost = np.inf

				for nextCity in citiesLeft:
					tempCost = cost(self.d_matrix, currCity, nextCity)
					if tempCost == np.inf:
						numBadGuys += 1

						if numBadGuys == len(citiesLeft):
							citiesLeft_shuf = random.sample(list(citiesLeft), k=numBadGuys)
							order[-numBadGuys:] = citiesLeft_shuf
							return order

						continue
					if tempCost < bestCost:
						bestCost = tempCost
						nearestNeighbour = nextCity
						numBadGuys = 0

				if nearestNeighbour == None:
					continue

				order[index] = nearestNeighbour
				citiesLeft = citiesLeft - set([nearestNeighbour])
				index += 1

				if len(citiesLeft) == 1:
					order[-1] = list(citiesLeft)[0]
					index +=1

			a = fitness_jit(self.d_matrix, order) == np.Inf
			if (self.d_matrix[order[-1] ,order[0]] == np.inf) | (a):
				# print('found inf in NN')
				repeatAgain = True
			else:
				repeatAgain = False

		return order

	# Use nearestNeighbours Advance to create Ind
	def NN_advanced(self, lambdaa):
		startTime = time.time()
		order = np.negative(np.ones((self.N), dtype=np.int))
		start = np.random.choice(range(0,self.N))
		order[0] = start
		citiesLeft_init = set(range(0,self.N)) - set([start])
		citiesLeft = citiesLeft_init.copy()
		index = 1
		numBadGuys = 0
		laverageAmountBack = 10
		skipOnce = False
		skipGuy = -12345

		while index != self.N:
			currCity = order[index-1]
			nearestNeighbour = None
			bestCost = np.inf

			# for nextCity in citiesLeft.copy():
			for nextCity in citiesLeft_init:
				if (nextCity == skipGuy) & (skipOnce == True):
					skipGuy = -12345
					skipOnce = False
					continue

				tempCost = cost(self.d_matrix, currCity, nextCity)
				if tempCost == np.inf:
					numBadGuys += 1
					if numBadGuys == len(citiesLeft):
						stillToFill = numBadGuys
						movesBack = stillToFill + laverageAmountBack
						skipGuy = order[-(movesBack)]
						citiesLeft.update(set(order[-movesBack:-numBadGuys]))
						order[-movesBack:] = np.negative(np.ones(movesBack, dtype=int))
						index -= laverageAmountBack

						numBadGuys = 0
						skipOnce = True
					continue

				if tempCost < bestCost:
					bestCost = tempCost
					nearestNeighbour = nextCity
					badGuys = set()
					numBadGuys = 0

			if nearestNeighbour == None:
				continue

			order[index] = nearestNeighbour
			citiesLeft = citiesLeft - set([nearestNeighbour])
			index += 1

			if len(citiesLeft) == 1:
				order[-1] = list(citiesLeft)[0]
				index +=1
		return order

# ------------------------------------------------------------------------------
# SELECTION
	def k_tourament(self, population: np.ndarray, k: int) -> Individual:
		fitnesses = []
		bestFitness = np.inf
		kFound = 0
		bestInd = random.choice(population)
		while kFound != k:
			ind = random.choice(population)
			IndFitness = fitness_jit(self.d_matrix, ind.order)
			if IndFitness == np.inf:
				continue
			kFound +=1
			fitnesses.append(IndFitness)
			if IndFitness < bestFitness:
				bestFitness = IndFitness
				bestInd = ind
		return bestInd

# ------------------------------------------------------------------------------
# RECOMBINATION
	def OrderCrossover(self, ind1: Individual, ind2: Individual) -> Individual:
		subset_indices = np.random.randint(low = 0, high = self.N, size = 2)
		low_index = np.min(subset_indices)
		high_index = np.max(subset_indices)
		subset_first_parent = np.array(ind1.order[low_index:high_index+1])
		offspring = np.empty(shape = self.N, dtype = int)
		rotated_ind2_order = np.concatenate((ind2.order[1:], (ind2.order[:1])))
		remaining_ind2 = np.setdiff1d(rotated_ind2_order, subset_first_parent, assume_unique=True)
		offspring[low_index:high_index+1] = subset_first_parent
		offspring[0:low_index] = remaining_ind2[0:low_index]
		offspring[high_index+1:self.N] = remaining_ind2[low_index:]

		beta = 2 * random.random() - 0.5
		alpha = ind1.alpha + beta * (ind2.alpha - ind1.alpha)
		ind = Individual(self.N, order = offspring, alpha = max(0.05, alpha) )
		return ind

# ------------------------------------------------------------------------------
# MUTATION
	def mutationSwap(self, ind: Individual):
		if random.random() < ind.alpha:
			i1 = random.randint(0, len(ind.order)-1)
			i2 = random.randint(0, len(ind.order)-1)
			ind.order[i1], ind.order[i2] = ind.order[i2], ind.order[i1]
		return ind

	def mutationShuf(self, ind: Individual) -> Individual :

		if random.random() < ind.alpha:
			index1  = random.randint(0, len(ind.order)-1)
			index2 = index1
			while index1 == index2:
				index2 = random.randint(0, len(ind.order)-1)
			frm = min(index1, index2)
			to  = max(index1, index2)

			newOrder = ind.order
			shuffledSection = newOrder[frm:to]
			random.shuffle(shuffledSection)
			newOrder[frm:to] = shuffledSection
			ind.order = newOrder

		return ind

	def adaptMutation(self, population: List[Individual]):
		lenPopulation = len(population)
		populationFitness = []
		for ind in population:
			fitnessInd = fitness_jit(self.d_matrix, ind.order)
			populationFitness.append(fitnessInd)
		meanFitnessPopulation = np.mean(populationFitness)

		for i in range(lenPopulation):
			indFitness = populationFitness[i]
			if indFitness > meanFitnessPopulation:
				population[i].alpha += 0.01
			else:
				population[i].alpha -= 0.01

# ------------------------------------------------------------------------------
# ELIMINATION
	def sharedElimination(self, lambdaa, population):
		survivors = []
		splitNum = round(len(population) * 0.70)
		pop_Ktour = population[:splitNum]
		pop_sharedElim = population[splitNum:]

		# Apply k-torunament elimination
		len_pop_Ktour = int(lambdaa * 0.70)
		for i in range(0, len_pop_Ktour):
			survivors.append(self.k_tourament(pop_Ktour, self.p.k))

		# Apply shared fitness elimination
		survivors_frmShared = []
		for i in range(0, lambdaa - len_pop_Ktour):
			fvals = self.sharedFitnessWrapper(fitness_jit, pop_sharedElim, survivors_frmShared[0:i - 1], 1)
			idx = np.argmin(fvals)
			survivors_frmShared.append(pop_sharedElim[idx])

		survivors += survivors_frmShared
		return survivors

	def sharedFitnessWrapper(self, function, X, population = None, betaInit = 0):
		if population is None:
			fitnessValues = [function(self.d_matrix, x.order) for x in X]
			return fitnessValues

		alpha = 0.4
		sigma = self.N * 0.3

		modObjValues = np.zeros(len(X))
		for i, x in enumerate(X):
			ds = np.array([ self.distance(x, y) for y in population])
			onePlusBeta = betaInit
			for distance in ds:
				if distance <= sigma:
					onePlusBeta += 1 - (distance/sigma) ** alpha

			fitnessValue = function(self.d_matrix, x.order)
			modObjValues[i] = fitnessValue * onePlusBeta ** np.sign(fitnessValue)

		return modObjValues

# ------------------------------------------------------------------------------
# HELPER METHODS

	# Calculates results for reporter
	def getResults(self, population):
		fitnesses = np.array([fitness_jit(self.d_matrix, ind.order) for ind in population ])
		meanObjective = np.mean(fitnesses)
		bestIndex = np.argmin(fitnesses)
		bestObjective = fitnesses[bestIndex]
		bestSolution = population[bestIndex].order
		return meanObjective, bestObjective, bestSolution

	# Calculates distace for sharedElimination
	def distance(self, ind1: Individual, ind2: Individual):
		hamming = np.bitwise_xor(ind1.order, ind2.order)
		numDiffCities = hamming.nonzero()[0]
		distance = len(numDiffCities)
		return distance

	# Calculates distace with ind's edges for sharedElimination
	def distance_withEdges(self, ind1: Individual, ind2: Individual):
		ind1Edges = set(ind1.edges)
		ind2Edges = set(ind2.edges)
		distance_edges = len(ind1.order) - len(ind1Edges.intersection(ind2Edges))
		return distance_edges

	# Make bestSolution's city numbering start from 0
	def getCity0Start(self, order: np.ndarray) -> np.ndarray:
		city0Index = np.where(order == 0)[0][0]
		return np.concatenate((order[city0Index:],order[:city0Index]))

# Modify the class name to match your student number.
class r0916799:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	# Read distance matrix from file.
	def getMatrix(self, filename):
		print("--- TSP Genetic Algorithm:", filename, "---")
		print("- Reading distance matrix...")
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()
		return distanceMatrix

	# Prints fitness values of population before & after optimize
	def printResults(self, section, TSP, numSameFit, itr, meanObj, bestObj, bestSol):

		# Limits how much of ind's order one can see in terminal
		maxIcanSee = 75
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
	def saveData(self, section, p, itr, meanObj,bestObj):

		if section == 'init' :
			header_r = ['Iteration', 'Mean Fitness', 'Best Fitness']
			header_p = ['populationSize',
			 			'offspringSize',
						'k-tournament',
						'alpha',
						'its',
						'lso_ON',
						'lso_percen',
						'greedy_percen'
						]
			if p.lso_ON == True: lso_ON_int = 1
			else               : lso_ON_int = -1

			values_p = [p.lambdaa,
						p.mu,
			       		p.k,
						p.alpha,
						p.its,
						lso_ON_int,
						p.lso_percen,
						p.greedy_percen]

			with open('result.csv', 'w', newline='') as file:
				writer = csv.writer(file)
				file.truncate()
				writer.writerow(header_p)
				writer.writerow(values_p)
				writer.writerow(header_r)
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
		p   = Parameters(lambdaa = 100,
						 k = 4,
						 its = 50000,
						 lso_ON = True,
						 lso_percen = 0.15,
						 greedy_percen = 0.15,)

		TSP = TSP_problem(distanceMatrix, p)
		print("- Initializing population...")
		start_time = time.time()
		population = TSP.initialize(p.lambdaa)
		print(f'Init: {time.time() - start_time:.2f} sec')
		print("- Running optimization...")
		meanObj, bestObj, bestSol = TSP.getResults(population)
		self.printResults('init', TSP, 0, 0, meanObj, bestObj, bestSol)
		self.saveData('init', p, 0, meanObj, bestObj)

		# Variables for ConvergenceTest
		numSameFit, prevBestFit, itr = 0, 0, 0
		totalTime = 0.0

		while(numSameFit < 10000 and itr < p.its):
			itrStartTime = time.time()

			# Create the offspring
			offspring = np.empty(p.mu, dtype = Individual)
			for jj in range(0, p.mu):
				parent1 	  = TSP.selection(population, p.k)
				parent2 	  = TSP.selection(population, p.k)
				offspring[jj] = TSP.recombination(parent1, parent2)
				offspring[jj] = TSP.mutation(offspring[jj])

			# Adjust ind's mutation rate based on meanFit Population
			TSP.adaptMutation(population)

			# Apply LSO to population
			numberImproved_byLSO = round(p.lambdaa * p.lso_percen)
			for i in range(len(population)):
				population[i] = TSP.mutation(population[i])
				if p.lso_ON & (numberImproved_byLSO !=0) :
					improvedOrder = lso_2_opt(distanceMatrix, population[i].order)
					population[i] = Individual(TSP.N, improvedOrder, population[i].alpha)
					population[i] = Individual(TSP.N, improvedOrder, population[i].alpha)
					numberImproved_byLSO -= 1

			# Join offspring with original population
			joinedPopulation = np.concatenate((offspring, population))

			# Elimination by k-tournament & sharedElimination
			population = TSP.elimination(p.lambdaa, joinedPopulation)

			# Calculate fitness values from current iteration
			meanObj, bestObj, bestSol = TSP.getResults(population)
			self.printResults('inLoop', TSP, numSameFit, itr, meanObj, bestObj, bestSol)
			itr += 1

			# Call the reporter aft 1 itr for redeable plot
			if itr > 1 :
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
			self.saveData('inLoop', p, itr, meanObj, bestObj)
			itrDuration = time.time() - itrStartTime
			totalTime += itrDuration
			if (itr % 50) == 0:
				print(f"--- Itr totalTime : {totalTime:.2f} sec ---")
				print(f"--- Rep timeLeft  : {timeLeft:.2f} sec ---" )

		return

if __name__ == '__main__':

	a = r0916799()
	# a.optimize('./tour50.csv')

	# COMMENT AFTERWARDS
	N = 750
	# N ='Inf'
	filename = f'./tour{N}.csv'
	a.optimize(filename)
	from plotResult import doPlots
	doPlots(N)
