"""
Algorithm objects are thin wrappers around built-in optimisation library algorithms.
At this stage they come only from the inspyred library but other optimisation libraries could
be added as well.
"""
from abc import ABCMeta # Metaclass for abstract base classes
from inspyred import ec
from inspyred.ec import observers
from inspyred.ec import terminators
from inspyred.ec import replacers
from inspyred.ec import variators
from random import Random
from time import time


class _Algorithm(object):
    """
    Base optimization algorithm class
    """
    __metaclass__ = ABCMeta # Declare this class abstract to avoid accidental construction

    def __init__(self, max_constraints, min_constraints, evaluator,
                mutation_rate, maximize, seeds, population_size):

        self.max_constraints = max_constraints
        self.min_constraints = min_constraints
        self.evaluator = evaluator
        self.population_size = population_size
        self.maximize = maximize
        self.mutation_rate = mutation_rate
        self.seeds = seeds

    def uniform_random_chromosome(self, random, args):
        chromosome = []
        for lo, hi in zip(self.max_constraints, self.min_constraints):
            chromosome.append(random.uniform(lo, hi))
        return chromosome

    def print_report(self, final_pop, do_plot, stat_file_name):
        print(max(final_pop))
        # Sort and print the fitest individual, which will be at index 0.
        final_pop.sort(reverse=True)
        print '\nfitest individual:'
        print(final_pop[0])

        if do_plot:
            from inspyred.ec import analysis
            analysis.generation_plot(stat_file_name, errorbars=False)

    def generate_description(self, random):
        ret = [random.uniform(0.0, 1.) for i in xrange(self.genome_size)] #@UnusedVariable
        return ret


class _InspyredAlgorithm(_Algorithm):

    __metaclass__ = ABCMeta # Declare this class abstract to avoid accidental construction

    def __init__(self, constraints, mutation_rate=None, num_elites=None, stdev=None, 
                 terminator=terminators.generation_termination,
                 variator=[variators.blend_crossover, variators.gaussian_mutation],
                 observer=observers.file_observer):
        self.genome_size = len(constraints)
        self.mutation_rate = mutation_rate
        self.num_elites=num_elites
        self.stdev=stdev
        self.observer = observer
        self.variator = variator
        self.terminator = terminator
        self.bounder = ec.Bounder(*zip(constraints))  
        

class EDAAlgorithm(_InspyredAlgorithm):

    def optimize(self, population_size, evaluator, random_seed=None, **kwargs):
        if random_seed is None:
            random_seed = (long(time.time() * 256))
        rng = Random()
        rng.seed(random_seed)
        ea = ec.EDA(rng)
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
                        **kwargs)
        return pop, ea


class NSGA2Algorithm(_InspyredAlgorithm):

    def __init__(self, constraints, mutation_rate, num_elites=None, stdev=None, 
                 allow_indentical=True, terminator=terminators.generation_termination,
                 variator=[variators.blend_crossover, variators.gaussian_mutation],
                 replacer= replacers.random_replacement,
                 observer=observers.file_observer):
        super(NSGA2Algorithm, self).__init__(constraints=constraints, mutation_rate=mutation_rate, 
                                             num_elites=num_elites, stdev=stdev, 
                                             terminator=terminator, variator=variator, 
                                             observer=observer)
        self.allow_identical=allow_indentical,
        self.replacer = replacer
        

    def optimize(self, population_size, run_and_evaluate, max_generations=100, seeds=None,  
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
                        evaluator=run_and_evaluate,
                        pop_size=population_size,
                        bounder=self.bounder,
                        maximize=False,
                        seeds=seeds,
                        max_generations=max_generations,
                        **kwargs)
        return pop, ea
