"""
Algorithm objects are thin wrappers around built-in optimisation library
algorithms. At this stage they come only from the inspyred library but other
optimisation libraries could be added as well.
"""
from __future__ import absolute_import
from abc import ABCMeta  # Metaclass for abstract base classes
import pkgutil


class Algorithm(object):
    """
    Base optimization algorithm class
    """
    # Declare this class abstract to avoid accidental construction
    __metaclass__ = ABCMeta

    # A value assigned to simulations that fail due to numerical instability
    # caused by unrealistic parameters. Can be overridden by algorithms that
    # can handle such values more gracefully (such as GridAlgorithm).
    BAD_FITNESS_VALUE = 1e20

    def optimize(self, evaluator):
        """
        The optimisation algorithm, which takes the tries to minimise the
        supplied evaluator function

        `evaluator` -- a function that takes a candidate (iterable of floats)
                       and evaluates its fitness. This function is supplied by
                       the relevant Tuner class and is transparent to the
                       algorithm.
        """
        raise NotImplementedError("'optimize' is not implemented by derived "
                                  "class of 'Algorithm' ,'{}'"
                                  .format(self.__class__.__name__))

    def set_tune_parameters(self, tune_parameters):
        self.genome_size = len(tune_parameters)
        self.constraints = [(p.lbound, p.ubound) for p in tune_parameters]

    def uniform_random_chromosome(self, random, args):  # @UnusedVariable
        return [random.uniform(lo, hi) for lo, hi in self.constraints]


def algorithm_factory(args):
    try:
        return available_algorithms[args.algorithm](args)
    except KeyError:
        raise Exception("Unrecognised algorithm type '{}'. Valid options are "
                        "'{}'. Make sure the module of the algorithm you wish "
                        "to use can be imported successfully")

# TODO: These variables should be enclosed in a singleton 'Factory' class
available_algorithms = {}
script_options = []


def add_factory_to_register(key, loader):
    if key in available_algorithms:
        raise Exception("Attempted to add onflicting algorithm keys '{}' to"
                        "loader")
    available_algorithms[key] = loader


def add_option_adder_to_register(option_adder):
    script_options.append(option_adder)


# Import sub-modules who should then register a loader in the
# algorithm register
for _, modname, _ in pkgutil.iter_modules(__path__, prefix=__name__ + '.'):
    try:
        __import__(modname)
    except ImportError:
        print ("Algorithm module '{}' is not installed or did not load "
               "correctly".format(modname))
