# -*- coding: utf-8 -*-
"""
This module represents a thin wrapper around the IDEA algorithms of
Peter A.N. Bosman's AMaLGaM Estimation of Distribution Optimisation
Algorithm (EDA) source code, which can be downloaded here
http://homepages.cwi.nl/~bosman/source_code.php
"""
from __future__ import absolute_import, print_function
from abc import ABCMeta  # Metaclass for abstract base classes
from . import Algorithm, add_factory_to_register
from amalgam import amalgam_full


class AmalgamAlgorithm(Algorithm):
    """
    Base class for AMaLGaM IDEA (multiple estimation of distribution) algorithm
    objects
    """

    # Declare this class abstract to avoid accidental construction
    __metaclass__ = ABCMeta

    def optimize(self, evaluator, **kwargs):
        """
        `evaluator`              -- a function that takes a list of parameters
                                    and returns a fitness value
        `number_of_parameters`   -- The number of parameters to tune
        `lower_bounds`           -- The lower bounds of the parameters. Either
                                    a list of values or function that takes the
                                    dimension as an argument
        `upper_bounds`           -- The upper bounds of the parameters. Either
                                    a list of values or function that takes the
                                    dimension as an argument
        `rotation_angle`         -- The angle of rotation to be applied to the
                                    problem.
        `num_populations`        -- The number of parallel populations that
                                    initially partition the search space.
        `max_evaluations`        -- The maximum number of evaluations.
        `value_to_reach`         -- The value-to-reach (function value of best
                                    solution that is feasible).
        `fitness_variance_tol`   -- The minimum fitness variance level that is
                                    allowed.
        `tau`                    -- The selection truncation percentile (in
                                    [1/population_size,1]).
        `population_size`        -- The size of each population.
        `distr_mult_decrease`    -- The multiplicative distribution
                                    multiplier decrease.
        `st_dev_ratio_threshold` -- The maximum ratio of the distance of the
                                    average improvement to the mean compared to
                                    the distance of one standard deviation
                                    before triggering AVS (SDR mechanism).
        `max_no_improvement`     -- The maximum number of subsequent
                                    generations without an improvement while
                                    the distribution multiplier is <= 1.0.
        """
        if not self.tuner:
            raise Exception("optimize method of algorithm must be called from "
                            "within tuner")
        self.kwargs.update(kwargs)
        lbounds, ubounds = zip(*self.constraints)
        fitness, candidate = amalgam_full(evaluator, self.tuner.num_parameters,
                                          lbounds, ubounds, **self.kwargs)
        # The third value is provided to match other libraries, which might
        # return other information about the tuning process
        return candidate, fitness, None


class FullAmalgamAlgorithm(AmalgamAlgorithm):
    """
    A python wrapper around the the AMaLGaM-Full estimation of distributions
    algorithm (IDEA) developed by Peter A.N. Bosmanm, Dirk Thierens and Jorn
    Grahl
    """

    _algorithm = amalgam_full

    def __init__(self, rotation_angle=0.0, num_populations=5,
                 max_evaluations=1000, fitness_variance_tol=1e-4,
                 value_to_reach=0.0, tau=0.35, population_size=None,
                 distr_mult_decrease=0.9, st_dev_ratio_threshold=1.0,
                 max_no_improvement=None, output_dir=None):
        """
        `rotation_angle`         -- The angle of rotation to be applied to the
                                    problem.
        `num_populations`        -- The number of parallel populations that
                                    initially partition the search space.
        `max_evaluations`        -- The maximum number of evaluations.
        `fitness_variance_tol`   -- The minimum fitness variance level that is
                                    allowed.
        `value_to_reach`         -- The value-to-reach (function value of best
                                    solution that is feasible).
        `tau`                    -- The selection truncation percentile (in
                                    [1/population_size,1]).
        `population_size`        -- The size of each population.
        `distr_mult_decrease`    -- The multiplicative distribution
                                    multiplier decrease.
        `st_dev_ratio_threshold` -- The maximum ratio of the distance of the
                                    average improvement to the mean compared to
                                    the distance of one standard deviation
                                    before triggering AVS (SDR mechanism).
        `max_no_improvement`     -- The maximum number of subsequent
                                    generations without an improvement while
                                    the distribution multiplier is <= 1.0.
        """
        self.kwargs = {'rotation_angle': rotation_angle,
                       'number_of_populations': num_populations,
                       'maximum_number_of_evaluations': max_evaluations,
                       'vtr': value_to_reach,
                       'fitness_variance_tolerance': fitness_variance_tol,
                       'tau': tau, 'population_size': population_size,
                       'distribution_multiplier_decrease': distr_mult_decrease,
                       'st_dev_ratio_threshold': st_dev_ratio_threshold,
                       'maximum_no_improvement_stretch': max_no_improvement,
                       'output_dir': output_dir}

algorithm_types = {'amalgam-full': FullAmalgamAlgorithm}


def algorithm_factory(args):
    try:
        Algorithm = algorithm_types[args.algorithm]
    except KeyError:
        raise Exception("Unrecognised algorithm '{}'".format(args.algorithm))
    kwargs = dict(args.optimize_argument)
    return Algorithm(**kwargs)


# Register algorithm loader with the short names for the algorithms
for key in algorithm_types.iterkeys():
    add_factory_to_register(key, algorithm_factory)
