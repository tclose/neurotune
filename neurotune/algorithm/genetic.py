from __future__ import absolute_import
from abc import ABCMeta # Metaclass for abstract base classes
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

    __metaclass__ = ABCMeta # Declare this class abstract to avoid accidental construction

    def __init__(self, mutation_rate=None, num_elites=None, stdev=None, 
                 terminator=terminators.generation_termination,
                 variator=[variators.blend_crossover, variators.gaussian_mutation],
                 observer=observers.file_observer):
        self.mutation_rate = mutation_rate
        self.num_elites=num_elites
        self.stdev=stdev
        self.observer = observer
        self.variator = variator
        self.terminator = terminator
        
    def _set_tuneable_parameters(self, tuneable_parameters):
        super(InspyredAlgorithm, self)._set_tuneable_parameters(tuneable_parameters)
        self.bounder = ec.Bounder(*zip(*self.constraints))

    def print_report(self, final_pop, do_plot, stat_file_name):
        print(max(final_pop))
        # Sort and print the fitest individual, which will be at index 0.
        final_pop.sort(reverse=True)
        print '\nfitest individual:'
        print(final_pop[0])

        if do_plot:
            from inspyred.ec import analysis
            analysis.generation_plot(stat_file_name, errorbars=False)


class EDAAlgorithm(InspyredAlgorithm):

    def optimize(self, population_size, evaluator, max_generations=100, seeds=None, 
                 random_seed=None, **kwargs):
        if random_seed is None:
            random_seed = (long(time() * 256))
        rng = Random()
        rng.seed(random_seed)
        ea = ec.EDA(rng)
        ea.observer = self.observer
        ea.variator = self.variator
        ea.terminator = self.terminator
        for key in ('mutation_rate', 'num_elites', 'stdev'):
            kwargs[key] = getattr(self, key)
        pop = ea.evolve(generator=self.generate_description,
                        evaluator=evaluator,
                        pop_size=population_size,
                        bounder=self.bounder,
                        maximize=False,
                        seeds=seeds,
                        max_generations=max_generations,
                        **kwargs)
        return pop, ea


class NSGA2Algorithm(InspyredAlgorithm):

    def __init__(self, mutation_rate, num_elites=None, stdev=None, 
                 allow_indentical=True, terminator=terminators.generation_termination,
                 variator=[variators.blend_crossover, variators.gaussian_mutation],
                 replacer= replacers.random_replacement,
                 observer=observers.file_observer):
        super(NSGA2Algorithm, self).__init__(mutation_rate=mutation_rate, 
                                             num_elites=num_elites, stdev=stdev, 
                                             terminator=terminator, variator=variator, 
                                             observer=observer)
        self.allow_identical=allow_indentical,
        self.replacer = replacer
        

    def optimize(self, population_size, evaluator, max_generations=100, seeds=None, 
                 random_seed=None, **kwargs):
        if random_seed is None:
            random_seed = long(time.time() * 256)
            print "Using random seed: {}".format(random_seed)
        rng = Random()
        rng.seed(random_seed)
        ea = ec.emo.NSGA2(rng)
        ea.observer = self.observer
        ea.variator = self.variator
        ea.terminator = self.terminator
        for key in ('mutation_rate', 'num_elites' 'stdev'):
            kwargs[key] = self.__getattr__(key)
        pop = ea.evolve(generator=self.generate_description,
                        evaluator=evaluator,
                        pop_size=population_size,
                        bounder=self.bounder,
                        maximize=False,
                        seeds=seeds,
                        max_generations=max_generations,
                        **kwargs)
        return pop, ea
