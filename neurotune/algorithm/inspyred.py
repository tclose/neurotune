# -*- coding: utf-8 -*-
"""
This module represents a thin wrapper around the evolutionary algorithms
provided by the inspyred library (https://pypi.python.org/pypi/inspyred)
"""
from __future__ import absolute_import, print_function
import os.path
from copy import copy
from abc import ABCMeta  # Metaclass for abstract base classes
from time import time
from random import Random
from inspyred import ec
from . import Algorithm


class InspyredAlgorithm(Algorithm):
    """
    Base class for inspyred evolutionary algorithms based algorithm objects
    """

    # Declare this class abstract to avoid accidental construction
    __metaclass__ = ABCMeta

    _ea_attribute_names = ['selector', 'variator', 'replacer', 'migrator',
                           'archiver', 'observer', 'terminator', 'logger']

    _ea_defaults = {'terminator': ec.terminators.generation_termination,
                    'observer': [ec.observers.best_observer]}

    def __init__(self, pop_size, output_dir=os.getcwd(),
                 max_generations=100, seeds=None, random_seed=None, **kwargs):
        """
        `pop_size`        -- the size of the population in each generation
        `max_generations` -- places a limit on the maximum number of
                             generations
        `seeds`           -- initial starting states of the algorithm
        `random_seed`     -- the seed to initialise the candidates with
        `output_dir`      -- the path of the directory to save the optimisation
                             statistics in
        `kwargs`          -- optional arguments to be passed to the
                             optimisation algorithm
        """
        self.ea_attributes = copy(self._ea_defaults)
        for key in self._ea_attribute_names:
            if key in kwargs:
                self.ea_attributes[key] = kwargs.pop(key)
        # Ensure file observer is part of observers
        observers = self.ea_attributes['observer']
        try:
            if ec.observers.file_observer not in observers:
                observers.append(ec.observers.file_observer)
        except AttributeError:
            if observers != ec.observers.file_observer:
                self.ea_attributes['observer'] = [observers,
                                                  ec.observers.file_observer]
        self.pop_size = pop_size
        self.evolve_args = kwargs
        self.evolve_args['max_generations'] = max_generations
        self._rng = Random()
        self.output_dir = output_dir
        self.set_random_seed(random_seed)
        self.set_seeds(seeds)

    def optimize(self, evaluator, **kwargs):
        if not self.tuner:
            raise Exception("optimize method of algorithm must be called from "
                            "within tuner")
        ea = self._InspyredClass(self._rng)
        for key, val in self.ea_attributes.iteritems():
            setattr(ea, key, val)
        # Combine the keyword arguments from the __init__ method and the
        # optimise method
        output_dir = kwargs.pop('output_dir', self.output_dir)
        evolve_kwargs = self.evolve_args
        evolve_kwargs.update(kwargs)
        # Get file paths for population and individual statistics
        stats_path = os.path.join(output_dir, 'statistics.csv')
        indiv_path = os.path.join(output_dir, 'individuals.csv')
        print("Population statistics will be saved to '{}'"
              .format(stats_path))
        print("Population individuals will be saved to '{}'"
              .format(indiv_path))
        # Ensure the files don't exist, deleting them if they do
        if os.path.exists(stats_path):
            if __debug__:
                os.remove(stats_path)
            else:
                raise Exception("Statistics file '{}' already exists"
                                .format(stats_path))
        if os.path.exists(indiv_path):
            if __debug__:
                os.remove(indiv_path)
            else:
                raise Exception("Individuals file '{}' already exists"
                                .format(stats_path))
        with open(stats_path, 'w') as stats_f, open(indiv_path, 'w') as ind_f:
            pop = ea.evolve(generator=self.uniform_random_chromosome,
                            evaluator=evaluator,
                            pop_size=self.pop_size,
                            bounder=ec.Bounder(*zip(*self.constraints)),
                            maximize=False,
                            seeds=self.seeds,
                            statistics_file=stats_f,
                            individuals_file=ind_f,
                            **evolve_kwargs)
        fittest = min(pop, key=lambda c: c.fitness)
        return fittest.candidate, fittest.fitness, (pop, ea)

    def set_random_seed(self, seed=None):
        if seed is None:
            seed = (long(time() * 256))
        self._rng.seed(seed)

    def set_seeds(self, seeds):
        self.seeds = seeds


class MultiObjectiveInspyredAlgorithm(InspyredAlgorithm):

    # Declare this class abstract to avoid accidental construction
    __metaclass__ = ABCMeta

    def optimize(self, evaluator, **kwargs):
        # Wrap the list returned from the multi-objective objective in the
        # required class after it is evaluated (this saves having to import the
        # inspyred module into objective.combined, an allows it to be more
        # general)
        def pareto_evaluator(candidates, args):  # @UnusedVariable
            return ec.emo.Pareto(evaluator(candidates))
        return super(MultiObjectiveInspyredAlgorithm, self).\
                                           optimize(pareto_evaluator, **kwargs)


class GAAlgorithm(InspyredAlgorithm):
    """
    Evolutionary computation representing a canonical genetic algorithm.

    This class represents a genetic algorithm which uses, by default, rank
    selection, n-point crossover, bit-flip mutation, and generational
    replacement. In the case of bit-flip mutation, it is expected that each
    candidate solution is a Sequence of binary values.

    Optional keyword arguments in evolve args parameter:

    num_selected – the number of individuals to be selected
                   (default len(population))
    crossover_rate - the rate at which crossover is performed (default 1.0)
    num_crossover_points – the n crossover points used (default 1)
    mutation_rate – the rate at which mutation is performed (default 0.1)
    num_elites – number of elites to consider (default 0)
    """
    _InspyredClass = ec.GA


class EDAAlgorithm(InspyredAlgorithm):
    """
    Evolutionary computation representing a canonical estimation of
    distribution algorithm. This class represents an estimation of distribution
    algorithm which uses, by default, truncation selection, an internal
    estimation of distribution variation, and generational replacement. It is
    expected that each candidate solution is a Sequence of real values.

    The variation used here creates a statistical model based on the set of
    candidates. The offspring are then generated from this model. This
    function also makes use of the bounder function as specified in the EC’s
    evolve method.

    Optional keyword arguments in evolve args parameter:

    num_selected – the number of individuals to be selected
                   (default len(population)/2)
    num_offspring – the number of offspring to create (default len(population))
    num_elites – number of elites to consider (default 0)
    """
    _InspyredClass = ec.EDA

    def __init__(self, pop_size,
                 terminator=[ec.terminators.diversity_termination,
                             ec.terminators.generation_termination],
                 replacer=ec.replacers.crowding_replacement,
                 min_diversity=0.01, max_generations=100, **kwargs):
        super(EDAAlgorithm, self).__init__(pop_size, terminator=terminator,
                                           replacer=replacer,
                                           min_diversity=min_diversity,
                                           max_generations=max_generations,
                                           **kwargs)


class ESAlgorithm(InspyredAlgorithm):
    """
    Evolutionary computation representing a canonical evolution strategy.

    This class represents an evolution strategy which uses, by default, the
    default selection (i.e., all individuals are selected), an internal
    adaptive mutation using strategy parameters, and ‘plus’ replacement. It is
    expected that each candidate solution is a Sequence of real values.

    The candidate solutions to an ES are augmented by strategy parameters of
    the same length (using ec.generators.strategize). These strategy
    parameters are evolved along with the candidates and are used as the
    mutation rates for each element of the candidates. The evaluator is
    modified internally to use only the actual candidate elements (rather than
    also the strategy parameters), so normal evaluator functions may be used
    seamlessly.

    tau – a proportionality constant (default None)
    tau_prime – a proportionality constant (default None)
    epsilon – the minimum allowed strategy parameter (default 0.00001)

    If tau is None, it will be set to 1 / sqrt(2 * sqrt(n)), where n is the
    length of a candidate. If tau_prime is None, it will be set to 1 / sqrt(2 *
    n). The strategy parameters are updated as follows:

    """
    _InspyredClass = ec.ES


class DEAAlgorithm(InspyredAlgorithm):
    """
    Evolutionary computation representing a differential evolutionary
    algorithm.

    This class represents a differential evolutionary algorithm which uses, by
    default, tournament selection, heuristic crossover, Gaussian mutation, and
    steady-state replacement. It is expected that each candidate solution is a
    Sequence of real values.

    Optional keyword arguments in evolve args parameter:

    num_selected – the number of individuals to be selected (default 2)
    tournament_size – the tournament size (default 2)
    crossover_rate – the rate at which crossover is performed (default 1.0)
    mutation_rate – the rate at which mutation is performed (default 0.1)
    gaussian_mean – the mean used in the Gaussian function (default 0)
    gaussian_stdev – the standard deviation used in the Gaussian function
                     (default 1)
    """
    _InspyredClass = ec.DEA


class SAAlgorithm(InspyredAlgorithm):
    """
    Evolutionary computation representing simulated annealing.

    This class represents a simulated annealing algorithm. It accomplishes this
    by using default selection (i.e., all individuals are parents), Gaussian
    mutation, and simulated annealing replacement. It is expected that each
    candidate solution is a Sequence of real values. Consult the documentation
    for the simulated_annealing_replacement for more details on the keyword
    arguments listed below.

    Note The pop_size parameter to evolve will always be set to 1, even if a
    different value is passed. Optional keyword arguments in evolve args
    parameter:

    temperature – the initial temperature
    cooling_rate – a real-valued coefficient in the range (0, 1) by which the
                   temperature should be reduced
    mutation_rate – the rate at which mutation is performed (default 0.1)
    gaussian_mean – the mean used in the Gaussian function (default 0)
    gaussian_stdev – the standard deviation used in the Gaussian function
                     (default 1)
    """
    _InspyredClass = ec.SA


class NSGA2Algorithm(MultiObjectiveInspyredAlgorithm):
    """
    Evolutionary computation representing the nondominated sorting genetic
    algorithm.

    This class represents the nondominated sorting genetic algorithm (NSGA-II)
    of Kalyanmoy Deb et al. It uses nondominated sorting with crowding for
    replacement, binary tournament selection to produce population size
    children, and a Pareto archival strategy. The remaining operators take on
    the typical default values but they may be specified by the designer.
    """
    _InspyredClass = ec.emo.NSGA2

    def __init__(self, pop_size,
                 variators=[ec.variators.blend_crossover,
                            ec.variators.gaussian_mutation], **kwargs):
        super(NSGA2Algorithm, self).__init__(pop_size, variators=variators,
                                             **kwargs)


class PAESAlgorithm(MultiObjectiveInspyredAlgorithm):
    """
    Evolutionary computation representing the Pareto Archived Evolution
    Strategy.

    This class represents the Pareto Archived Evolution Strategy of Joshua
    Knowles and David Corne. It is essentially a (1+1)-ES with an adaptive grid
    archive that is used as a part of the replacement process.
    """
    _InspyredClass = ec.emo.PAES


algorithm_types = {'genetic': GAAlgorithm,
                   'eda': EDAAlgorithm,
                   'es': ESAlgorithm,
                   'diff': DEAAlgorithm,
                   'annealing': SAAlgorithm,
                   'nsga2': NSGA2Algorithm,
                   'pareto_archived': PAESAlgorithm}

replacer_types = {'truncation': ec.replacers.truncation_replacement,
                  'steady_state': ec.replacers.steady_state_replacement,
                  'generational': ec.replacers.generational_replacement,
                  'random': ec.replacers.random_replacement,
                  'plus': ec.replacers.plus_replacement,
                  'comma': ec.replacers.comma_replacement,
                  'crowding': ec.replacers.crowding_replacement,
                  'simulated_annealing': ec.replacers.\
                                               simulated_annealing_replacement}
