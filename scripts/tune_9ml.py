#!/usr/bin/env python
"""
Tunes a model against a reference trace or 9ml simulation
"""
from __future__ import absolute_import
import argparse
import os.path
import shutil
import math
from copy import deepcopy
import neo
from nineline.cells.neuron import NineCellMetaClass, simulation_controller
from nineline.cells.build import BUILD_MODE_OPTIONS
from nineline.arguments import outputpath
from neurotune import Parameter
from neurotune.tuner import EvaluationException
from neurotune.objective.phase_plane import (PhasePlaneHistObjective,
                                             PhasePlanePointwiseObjective)
from neurotune.objective.multi import MultiObjective, WeightedSumObjective
from neurotune.objective.spike import (SpikeFrequencyObjective,
                                       SpikeTimesObjective,
                                       MinCurrentToSpikeObjective)
from neurotune.algorithm import algorithm_factory, available_algorithms
from neurotune.simulation.nineline import NineLineSimulation
try:
    from neurotune.tuner.mpi import MPITuner as Tuner
except ImportError:
    from neurotune.tuner import Tuner
import cPickle as pkl

true_parameters = []

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('model', type=str,
                    help="The path of the 9ml cell to tune")
parser.add_argument('reference', type=str,
                    help="Either a path to a analog signal trace in "
                         "Neo format or a path to a 9ml cell model "
                         "which will be simulated and the resulting "
                         "trace will be used as a reference")
parser.add_argument('--build', type=str, default='lazy',
                    help="Option to build the NMODL files before "
                         "running (can be one of {})"
                         .format(BUILD_MODE_OPTIONS))
parser.add_argument('--time', type=float, default=2000.0,
                   help="Recording time")
parser.add_argument('--output', type=outputpath,
                    default=os.path.join(os.environ['HOME'], 'grid.pkl'),
                    help="The path to the output file where the grid will"
                         "be written")
parser.add_argument('--timestep', type=float, default=0.025,
                    help="The timestep used for the simulation "
                         "(default: %(default)s)")


def add_tune_arguments(parser):
    """
    Adds tuner specific arguments, which can be reused over multiple scripts
    """
    parser.add_argument('-o', '--objective', type=str, nargs='+',
                        default=[], action='append',
                        metavar=('OBJECTIVE_NAME', 'WEIGHTING'),
                        help="Selects which objective function to use "
                             "out of 'histogram', 'pointwise', 'frequency', "
                             "'spike_times' or a combination (potentially "
                             "weighted) of them (default: 'pointwise')")
    parser.add_argument('-p', '--parameter', nargs=4, default=[],
                        action='append',
                        metavar=('NAME', 'LBOUND', 'UBOUND', 'LOG_SCALE'),
                        help="Sets a parameter to tune and its lower and upper"
                             " bounds")
    parser.add_argument('--parameter_set', type=str, default=[], nargs='+',
                        metavar=('SET_NAME', 'SET_ARGS'),
                        help="Select which parameter set to tune from a few "
                             "descriptions")
    parser.add_argument('--num_generations', type=int, default=100,
                        help="The number of generations (iterations) to run "
                             "the algorithm for")
    parser.add_argument('--population_size', type=int, default=100,
                        help="The number of genomes in a generation")
    parser.add_argument('--algorithm', type=str, default='eda',
                        help="The type of algorithm used for the tuning. Can "
                             " be one of '{}' (default: %(default)s)"
                             .format("', '".join(available_algorithms.keys())))
    parser.add_argument('-a', '--optimize_argument', nargs=2, action='append',
                        default=[], metavar=('KEY', 'ARG'),
                        help="Extra arguments to be passed to the algorithm")
    parser.add_argument('-b', '--objective_argument', nargs=3, action='append',
                        metavar=('KEY', 'ARG', 'OBJECTIVE_INDEX'), default=[],
                        help="Extra keyword arguments to pass to the objective"
                             "function (can specify which objective in a "
                             "objective method by the third \"index\" "
                             "argument")
    parser.add_argument('--verbose', action='store_true', default=False,
                        help="Whether to print out which candidates are being "
                        "evaluated on which nodes")

# Add tuner specific arguments
add_tune_arguments(parser)


obj_dict = {'histogram': PhasePlaneHistObjective,
            'pointwise': PhasePlanePointwiseObjective,
            'frequency': SpikeFrequencyObjective,
            'spike_times': SpikeTimesObjective,
            'min_curr2spike': MinCurrentToSpikeObjective}

multi_objective_algorithms = ('nsga2', 'pareto_archived', 'multi-grid')


def load_reference(args):
    if args.reference.endswith('.9ml'):
        # Generate the reference trace from the original class
        cell = NineCellMetaClass(args.reference, build_mode=args.build)()
        cell.record('v')
        simulation_controller.run(simulation_time=args.time,
                                  timestep=args.timestep)
        reference = cell.get_recording('v')
    else:
        if args.reference.endswith('.neo.pkl'):
            block = neo.PickleIO(args.reference).read()
        elif args.reference.endswith('.neo.h5'):
            block = neo.NeoHdf5IO(args.reference).read()
        else:
            raise Exception("Unrecognised extension of reference file '{}'"
                            .format(args.reference))
        reference = block.segments[0].analogsignals[0]
    return reference


def get_objective(args, reference=None):
    if reference is None:
        reference = load_reference(args)
    # Distribute the objective arguments between the (possibly) multiple
    # objectives
    objective_args = [{}] * (len(args.objective) if args.objective else 2)
    for oa in args.objective_argument:
        try:
            index = int(oa[2])
        except IndexError:
            index = 0
        objective_args[index][oa[0]] = oa[1]
    try:
        # Use a mult-objective, or weighted sum objective depending on the
        # algorithm selected
        if len(args.objective) > 1:
            objs = [obj_dict[o[0]](reference, **kwargs)
                    for (o, kwargs) in zip(args.objective, objective_args)]
            if args.algorithm in multi_objective_algorithms:
                objective = MultiObjective(*objs)
            else:
                weights = [float(o[1]) for o in args.objective]
                objective = WeightedSumObjective(*zip(weights, objs))
        # Use a single objective function
        elif args.objective:
            objective = obj_dict[args.objective[0][0]](reference,
                                                       **objective_args[0])
        # Use the default objective
        else:
            if args.algorithm in multi_objective_algorithms:
                objective = MultiObjective(
                                       PhasePlanePointwiseObjective(reference),
                                       SpikeFrequencyObjective(reference))
            else:
                objective = WeightedSumObjective(
                                (1.0, PhasePlanePointwiseObjective(reference)),
                                (75.0, SpikeFrequencyObjective(reference)))
    except KeyError as e:
        raise Exception("Unrecognised objective '{}' passed to '--objective' "
                        "option".format(e))
    return objective


def get_parameters(args):
    if args.parameter and args.parameter_set:
        raise Exception("Cannot use --parameter and --parameter_set options "
                        "simulataneously")
    if args.parameter:
        parameters = [Parameter(p[0], 'S/cm^2', p[1], p[2], p[3])
                      for p in args.parameter]
    # The parameters to be tuned by the tuner
    elif args.parameter_set:
        if args.parameter_set[0] == 'all-gmaxes':
            parameter_string = ''
            if len(args.parameter_set) != 2:
                raise Exception("Range of parameters from initial values needs"
                                " to be provided for 'all-gmaxes' parameter "
                                "set")
            bound_range = float(args.parameter_set[1])
            from nineml.extensions.biophysics import parse
            bio_model = next(parse(args.model).itervalues())
            parameters = []
            for comp in bio_model.components.itervalues():
                if comp.type == 'ionic-current':
                    gbar = float(comp.parameters['g'].value)
                    gbar_log = math.log(gbar)
                    lbound = gbar_log - bound_range
                    ubound = gbar_log + bound_range
                    parameter_string += ("-p soma.{}.gbar {} {} 1 "
                                         .format(comp.name, lbound, ubound))
                    parameters.append(Parameter('soma.{}.gbar'
                                                .format(comp.name),
                                                'S/cm^2', lbound, ubound,
                                                log_scale=True,
                                                initial_value=gbar_log))
                    true_parameters.append(gbar)
        elif args.parameter_set:
            raise Exception("Unrecognised name '{}' passed to "
                            "'--parameter_set' option. Can be one of "
                            "('original', 'all-gmaxes')."
                            .format(args.parameter_set))
    else:
        raise Exception("No --parameter or --parameter set arguments passed "
                        "to tuning script")
    return parameters


def get_simulation(args):
    if args.model.endswith('.9ml'):
        simulation = NineLineSimulation(args.model, build_mode=args.build)
    else:
        raise Exception("Unrecognised model format '{}'".format(args.model))
    return simulation


def run(args, parameters=None, algorithm=None, objective=None,
        simulation=None, save_output=True):
    # Instantiate the tuner
    if not parameters:
        parameters = get_parameters(args)
    if not algorithm:
        algorithm = algorithm_factory(args)
    if not objective:
        objective = get_objective(args)
    if not simulation:
        simulation = get_simulation(args)
    if args.model == args.reference:  # For testing purposes
        print ("Target parameters:\n{}"
               .format(', '.join([str(p.initial_value) for p in parameters])))
    tuner = Tuner(parameters,
                  objective,
                  algorithm,
                  simulation,
                  verbose=args.verbose)
    tuner.true_candidate = true_parameters
    # Run the tuner
    try:
        candidate, fitness, algorithm_state = tuner.tune()
    except EvaluationException as e:
        e.save(os.path.join(os.path.dirname(args.output),
                            'evaluation_exception.pkl'))
        raise
    # Save the file if the tuner is the master
    if tuner.is_master() and save_output:
        print ("Fittest candidate (fitness {}): {}"
              .format(fitness, candidate))
        # Save the grid to file
        with open(args.output, 'w') as f:
            pkl.dump((candidate, fitness), f)
    return candidate, fitness, algorithm_state


def prepare_work_dir(submitter, args):
    os.mkdir(os.path.join(submitter.work_dir, '9ml'))
    copied_reference = os.path.join(submitter.work_dir, '9ml',
                                    os.path.basename(args.reference))
    shutil.copy(args.reference, copied_reference)
    copied_to_tune = os.path.join(submitter.work_dir, '9ml',
                                  os.path.basename(args.model))
    shutil.copy(args.model, copied_to_tune)
    NineCellMetaClass(copied_reference, build_mode='build_only')
    NineCellMetaClass(copied_to_tune, build_mode='build_only')
    args.reference = copied_reference
    args.model = copied_to_tune


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
