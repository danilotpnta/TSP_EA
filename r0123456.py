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
		ind = Individual([0, 1, 2, 3],0.5)
		TSP_fitness= TSP.fitness(ind)

		# Your code here.
		yourConvergenceTestsHere = True
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
