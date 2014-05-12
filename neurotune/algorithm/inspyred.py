"""
This module represents a thin wrapper around the evolutionary algorithms provided by the inspyred
library (https://pypi.python.org/pypi/inspyred)
"""
from __future__ import absolute_import
import os.path
from abc import ABCMeta  # Metaclass for abstract base classes
from time import time
from random import Random
from inspyred import ec
from . import Algorithm


class InspyredAlgorithm(Algorithm):
    """
    Base class for inspyred evolutionary algorithms based algorithm objects
    """

    __metaclass__ = ABCMeta  # Declare this class abstract to avoid accidental construction

    _ea_attribute_names = ['selector', 'variator', 'replacer', 'migrator', 'archiver', 'observer',
                           'terminator', 'logger'] 

# terminator=terminators.generation_termination,
# variator=[variators.blend_crossover, variators.gaussian_mutation],
# observer=observers.file_observer, 

    def __init__(self, population_size=100, max_generations=100, seeds=None, random_seed=None, 
                 output_dir=os.getcwd(), **kwargs):
        """
        `max_generations` -- the maximum number of iterations to perform
        `seeds`           -- initial starting states of the algorithm
        `random_seed`     -- the seed to initialise the candidates with
        `stats_filename`  -- the name of the file to save the generation-based statistics in
        `indiv_filename`  -- the name of the file to save the candidate parameters in
        `kwargs`          -- optional arguments to be passed to the optimisation algorithm
        """
        self.ea_attributes = {}
        for key in self._ea_attribute_names:
            if kwargs.has_key(key):
                self.ea_attributes = kwargs.pop(key)
        self.population_size = population_size
        self.evolve_args = kwargs
        self.evolve_args['max_generations'] = max_generations
        self._rng = Random()
        self.output_dir=output_dir
        self.set_random_seed(random_seed)
        self.set_seeds(seeds)

    def optimize(self, evaluator, **kwargs):
        ea = self._InspyredClass(self._rng)
        for key, val in self.ea_attributes.iteritems():
            setattr(ea, key, val)
        evolve_kwargs = self.evolve_args
        evolve_kwargs.update(kwargs)
        with open(os.path.join(self.output_dir, 'statistics.txt'), 'w') as stats_file, \
             open(os.path.join(self.output_dir, 'individuals.txt'), 'w') as indiv_file:
            pop = ea.evolve(generator=self.uniform_random_chromosome,
                            evaluator=evaluator,
                            pop_size=self.population_size,
                            bounder=ec.Bounder(*zip(*self.constraints)),
                            maximize=False,
                            seeds=self.seeds,
                            statistics_file=stats_file,
                            individual_file=indiv_file,
                            **evolve_kwargs)
        return pop, ea

    def set_random_seed(self, seed=None):
        if seed is None:
            seed = (long(time() * 256))
        self._rng.seed(seed)

    def set_seeds(self, seeds):
        self.seeds = seeds


class MultiObjectiveInspyredAlgorithm(InspyredAlgorithm):

    __metaclass__ = ABCMeta  # Declare this class abstract to avoid accidental construction

    def optimize(self, evaluator, **kwargs):
        # Wrap the list returned from the multi-objective objective in the required class
        # after it is evaluated (this saves having to import the inspyred module into 
        # objective.combined, an allows it to be more general)
        super(MultiObjectiveInspyredAlgorithm, self).optimize(lambda c: ec.emo.Pareto(evaluator(c)),
                                                              **kwargs)
            
        
class GAAlgorithm(InspyredAlgorithm):
    """
    Evolutionary computation representing a canonical genetic algorithm.

    This class represents a genetic algorithm which uses, by default, rank selection, n-point
    crossover, bit-flip mutation, and generational replacement. In the case of bit-flip mutation, it
    is expected that each candidate solution is a Sequence of binary values.
    
    Optional keyword arguments in evolve args parameter:
    
    num_selected – the number of individuals to be selected (default len(population))
    crossover_rate – the rate at which crossover is performed (default 1.0)
    num_crossover_points – the n crossover points used (default 1)
    mutation_rate – the rate at which mutation is performed (default 0.1)
    num_elites – number of elites to consider (default 0)
    """
    _InspyredClass = ec.GA
    

class EDAAlgorithm(InspyredAlgorithm):
    """
    Evolutionary computation representing a canonical estimation of distribution algorithm.
    This class represents an estimation of distribution algorithm which uses, by default, 
    truncation selection, an internal estimation of distribution variation, and generational 
    replacement. It is expected that each candidate solution is a Sequence of real values.
     
    The variation used here creates a statistical model based on the set of candidates. The 
    offspring are then generated from this model. This function also makes use of the bounder 
    function as specified in the EC’s evolve method.
     
    Optional keyword arguments in evolve args parameter:
     
    num_selected – the number of individuals to be selected (default len(population)/2)
    num_offspring – the number of offspring to create (default len(population))
    num_elites – number of elites to consider (default 0)
    """
    _InspyredClass = ec.EDA
    

class ESAlgorithm(InspyredAlgorithm):
    """
    tau – a proportionality constant (default None)
    tau_prime – a proportionality constant (default None)
    epsilon – the minimum allowed strategy parameter (default 0.00001)
    If tau is None, it will be set to 1 / sqrt(2 * sqrt(n)), where n is the length of a candidate. 
    If tau_prime is None, it will be set to 1 / sqrt(2 * n). The strategy parameters are updated as 
    follows:

    """
    _InspyredClass = ec.ES
    
    
class DEAAlgorithm(InspyredAlgorithm):
    """
    Evolutionary computation representing a differential evolutionary algorithm.
     
    This class represents a differential evolutionary algorithm which uses, by default, tournament
    selection, heuristic crossover, Gaussian mutation, and steady-state replacement. It is expected
    that each candidate solution is a Sequence of real values.
     
    Optional keyword arguments in evolve args parameter:
     
    num_selected – the number of individuals to be selected (default 2)
    tournament_size – the tournament size (default 2)
    crossover_rate – the rate at which crossover is performed (default 1.0)
    mutation_rate – the rate at which mutation is performed (default 0.1)
    gaussian_mean – the mean used in the Gaussian function (default 0)
    gaussian_stdev – the standard deviation used in the Gaussian function (default 1)
    """
    _InspyredClass = ec.DEA
    
    
class SAAlgorithm(InspyredAlgorithm):
    """
    Evolutionary computation representing simulated annealing.
     
    This class represents a simulated annealing algorithm. It accomplishes this by using default
    selection (i.e., all individuals are parents), Gaussian mutation, and simulated annealing
    replacement. It is expected that each candidate solution is a Sequence of real values. Consult
    the documentation for the simulated_annealing_replacement for more details on the keyword
    arguments listed below.
     
    Note The pop_size parameter to evolve will always be set to 1, even if a different value is
    passed. Optional keyword arguments in evolve args parameter:
     
    temperature – the initial temperature
    cooling_rate – a real-valued coefficient in the range (0, 1) by which the temperature should be reduced
    mutation_rate – the rate at which mutation is performed (default 0.1)
    gaussian_mean – the mean used in the Gaussian function (default 0)
    gaussian_stdev – the standard deviation used in the Gaussian function (default 1)        
    """
    _InspyredClass = ec.SA
    

class NSGA2Algorithm(MultiObjectiveInspyredAlgorithm):
    """
    Evolutionary computation representing the nondominated sorting genetic algorithm.

    This class represents the nondominated sorting genetic algorithm (NSGA-II) of Kalyanmoy Deb et
    al. It uses nondominated sorting with crowding for replacement, binary tournament selection to
    produce population size children, and a Pareto archival strategy. The remaining operators take
    on the typical default values but they may be specified by the designer.
    """
    _InspyredClass = ec.emo.NSGA2


class PAESAlgorithm(MultiObjectiveInspyredAlgorithm):
    """
    Evolutionary computation representing the Pareto Archived Evolution Strategy.

    This class represents the Pareto Archived Evolution Strategy of Joshua Knowles and David Corne.
    It is essentially a (1+1)-ES with an adaptive grid archive that is used as a part of the
    replacement process.
    """
    _InspyredClass = ec.emo.PAES
    