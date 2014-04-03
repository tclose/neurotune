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
        pop = ea.evolve(generator=self.generate_description,
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



class EDAAlgorithm(InspyredAlgorithm):

    _InspyredClass = ec.EDA


class NSGA2Algorithm(InspyredAlgorithm):

    _InspyredClass = ec.emo.NSGA2

