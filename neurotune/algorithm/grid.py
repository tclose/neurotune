from __future__ import absolute_import
from .__init__ import Algorithm


class GridAlgorithm(Algorithm):
    
    def optimize(self, population_size, evaluator, max_generations=100, seeds=None, 
                 random_seed=None, **kwargs):
        