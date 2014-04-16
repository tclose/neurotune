"""
Algorithm objects are thin wrappers around built-in optimisation library algorithms.
At this stage they come only from the inspyred library but other optimisation libraries could
be added as well.
"""
from __future__ import absolute_import
from abc import ABCMeta # Metaclass for abstract base classes


class Algorithm(object):
    """
    Base optimization algorithm class
    """
    __metaclass__ = ABCMeta # Declare this class abstract to avoid accidental construction

    def __init__(self, evaluator, mutation_rate, maximize, seeds, population_size):
        self.evaluator = evaluator
        self.population_size = population_size
        self.maximize = maximize
        self.mutation_rate = mutation_rate
        self.seeds = seeds
        
    def optimize(self, evaluator):
        raise NotImplementedError("'optimize' is not implemented by derived class of 'Algorithm'"
                                  ",'{}'".format(self.__class__.__name__))
        
    def _set_tuneable_parameters(self, tuneable_parameters):
        self.genome_size = len(tuneable_parameters)
        self.constraints = [(p.lbound, p.ubound) for p in tuneable_parameters]
        
    def uniform_random_chromosome(self, random, _):
        return [random.uniform(lo, hi) for lo, hi in self.constraints]

