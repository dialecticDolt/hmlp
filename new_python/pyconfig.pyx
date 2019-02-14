from config cimport *

cdef class PyConfig:
	cdef Configuration[float] c_config
	
	def __cinit__(self, DistanceMetric metric_type, int problem_size, int leaf_node_size, int neighbor_size, int maximum_rank, float tolerance, float budget, bool secure_accuracy):
		self.c_config = Configuration[float](metric_type, problem_size, leaf_node_size, neighbor_size, maximum_rank, tolerance, budget, secure_accuracy)
	
	def getMetricType(self):
		return self.c_config.MetricType()
	
	def getProblemSize(self):
		return self.c_config.ProblemSize()
	
	def getMaximumDepth(self):
		return self.c_config.getMaximumDepth()

	def getLeafNodeSize(self):
		return self.c_config.getLeafNodeSize()

	def isSymmetric(self):
		return self.c_config.IsSymmetric()

	def isAdaptive(self):
		return self.c_config.UseAdaptiveRanks()

	def isSecure(self):
		return self.c_config.SecureAccuracy()

	def setLeafNodeSize(self, int leaf_node_size):
		self.c_config.setLeafNodeSize(leaf_node_size)
	


	
