from __future__ import absolute_import
from abc import ABCMeta  # Metaclass for abstract base classes
from random import Random
from time import time
from inspyred import ec
from inspyred.ec import observers
from inspyred.ec import terminators
from inspyred.ec import replacers
from inspyred.ec import variators
from .__init__ import Algorithm


class InspyredAlgorithm(Algorithm):
    """
    Base class for inspyred (https://pypi.python.org/pypi/inspyred) evolutionary algorithms based 
    algorithm objects
    """

    __metaclass__ = ABCMeta  # Declare this class abstract to avoid accidental construction

    def __init__(self, population_size=100, max_generations=100, mutation_rate=None,
                 num_elites=None, stdev=None, terminator=terminators.generation_termination,
                 variator=[variators.blend_crossover, variators.gaussian_mutation],
                 observer=observers.file_observer, seeds=None,
                 random_seed=None, replacer=replacers.random_replacement, allow_identical=True, 
                 stats_filename=None, indiv_filename=None,
                 **kwargs):
        """
        `max_iterations`  -- the maximum number of iterations to perform
        `random_seed`     -- the seed to initialise the candidates with
        `stats_filename`  -- the name of the file to save the generation-based statistics in
        `indiv_filename`  -- the name of the file to save the candidate parameters in
        `kwargs`          -- optional arguments to be passed to the optimisation algorithm
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.num_elites = num_elites
        self.stdev = stdev
        self.observer = observer
        self.variator = variator
        self.terminator = terminator
        self.replacer = replacer
        self.allow_indentical = allow_identical
        self.set_seeds(seeds)
        self._rng = Random()
        self.set_random_seed(random_seed)
        self.evolve_kwargs = kwargs
        self.stats_filename=stats_filename
        self.indiv_filename=indiv_filename

    def optimize(self, evaluator, **kwargs):
        ea = self._InspyredClass(self._rng)
        ea.observer, ea.variator, ea.terminator = self.observer, self.variator, self.terminator
        self._open_readout_files(self.stats_filename, self.indiv_filename, kwargs)
        pop = ea.evolve(generator=self.uniform_random_chromosome,
                        evaluator=evaluator,
                        pop_size=self.population_size,
                        bounder=ec.Bounder(*zip(*self.constraints)),
                        maximize=False,
                        seeds=self.seeds,
                        max_generations=self.max_generations,
                        mutation_rate=self.mutation_rate,
                        num_elites=self.num_elites,
                        replacer=self.replacer,
                        stdev=self.stdev,
                        allow_indetical=self.allow_indentical,
                        **self.evolve_kwargs)
        self._close_readout_files()
        return pop, ea

    def set_random_seed(self, seed=None):
        if seed is None:
            seed = (long(time() * 256))
        self._rng.seed(seed)

    def set_seeds(self, seeds):
        self.seeds = seeds
        
    def print_report(self, final_pop, do_plot, stat_file_name):
        print(max(final_pop))
        # Sort and print the fitest individual, which will be at index 0.
        final_pop.sort(reverse=True)
        print '\nfitest individual:'
        print(final_pop[0])

        if do_plot:
            from inspyred.ec import analysis
            analysis.generation_plot(stat_file_name, errorbars=False)    

    def _open_readout_files(self, stats_filename, indiv_filename, kwargs):
        if stats_filename:
            self.stats_file = kwargs['statistics_file'] = open(stats_filename, 'w')
        else:
            self.stats_file = None
        if indiv_filename:
            self.indiv_file = kwargs['individual_file'] = open(indiv_filename, 'w')
        else:
            self.indiv_file = None
            
    def _close_readout_files(self):
        if self.stats_file:
            self.stats_file.close()
        if self.indiv_file:
            self.indiv_file.close()


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
     
    σ′i=σi+eτ⋅N(0,1)+τ′⋅N(0,1)
    σ′i=max(σ′i,ϵ)
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
    