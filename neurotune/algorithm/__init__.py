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
        _, _, lbounds, ubounds = zip(*tuneable_parameters)
        self.constraints = zip(lbounds, ubounds)
        
    def uniform_random_chromosome(self, random, _):
        chromosome = []
        for lo, hi in zip(self.constraints):
            chromosome.append(random.uniform(lo, hi))
        return chromosome

    def generate_description(self, random, args=None): #@UnusedVariable
        ret = [random.uniform(0.0, 1.) for i in xrange(self.genome_size)] #@UnusedVariable
        return ret

