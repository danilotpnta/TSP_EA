import Reporter
import numpy as np
import random
import statistics

class TSP_problem:
	def __init__(self, d_matrix):
		self.d_matrix = d_matrix
		self.N = np.size(d_matrix,0)

	def fitness(self, ind) :
		'''
		ind = [1,2,3,4,5]
		ind = [2,4,1,3,5]
		'''
		distance = 0

		for i in range(0, self.N - 1):
			city_current = ind.order[i]
			city_next = ind.order[i+1]

			distance += self.d_matrix[city_current][city_next]
			#print(self.d_matrix[city_current][city_next])

			if distance == np.inf:
				return distance
		distance += self.d_matrix[ind.order[-1]] [ind.order[0]]
		#print(self.d_matrix[ind.order[-1]] [ind.order[0]])
		#print(distance)
		return distance

	def recombination(self, ind1, ind2):
		subset_indices = np.random.randint(low=0, high=len(ind1.order), size=2)
		print("indices:", subset_indices)
		low_index = np.min(subset_indices)
		high_index = np.max(subset_indices)
		subset_first_parent = ind1.order[low_index:high_index+1]
		print("first parent: ", ind1.order)
		print("second parent: ", ind2.order)
		print("subset of first parent: ", subset_first_parent)
		offspring = []
		# j is the index of ind2 that marks the element being placed in the offspring
		j = 0
		# i is the index of the offspring where an element of ind2 is being placed
		for i in range(0, low_index):
			print(j)
			ind2_value = ind2.order[j]

			# if the element in ind2 is already in the random subset of the first parent,
			# then skip that element and iterate to the next one
			while ind2_value in subset_first_parent and j < len(ind1.order)-1:
				j += 1
				ind2_value = ind2.order[j]

			if j >= len(ind1.order):
				break

			offspring.append(ind2_value)
			j += 1

		# now add the subset from the first parent
		offspring += subset_first_parent
		print(j)

		# now do the same loop on ind2 for the remainder of the offspring indices
		for i in range(high_index+1, len(ind1.order)):
			print(j)
			ind2_value = ind2.order[j]

			# if the element in ind2 is already in the random subset of the first parent,
			# then skip that element and iterate to the next one
			while ind2_value in subset_first_parent and j < len(ind1.order)-1:
				j += 1
				ind2_value = ind2.order[j]

			if j >= len(ind1.order):
				break

			offspring.append(ind2_value)
			j += 1

		print("offspring: ", offspring)
		return offspring


class Individual:
	def __init__(self, order=None, alpha=None):
		if order == None:
			self.alpha = 0.05
			self.order = np.arange( len(TSP.values))
			np.random.shuffle(self.order)
		else:
			self.alpha = alpha
			self.order = order


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

		# print(type(distanceMatrix[5,7]))

		TSP = TSP_problem(distanceMatrix)
		ind = Individual( [0, 1, 2, 3, 4, 5, 6, 7],0.5)
		ind2 = Individual([7, 6, 1, 3, 4, 0, 5, 2], 0.5)
		TSP.recombination(ind, ind2)

		# Your code here.
		yourConvergenceTestsHere = False
		while( yourConvergenceTestsHere ):
			meanObjective = 0.0
			bestObjective = 0.0
			bestSolution = np.array([1,2,3,4,5])


			# Your code here.

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			if timeLeft < 0:
				break

		# Your code here.
		return 0

a = r0123456()
a.optimize('./tour50.csv')


# ind = Individual([0, 1, 2, 3],0.5)
# a.fitness(ind)
