#!/usr/bin/env python
"""
Tunes a 9ml file against a 9ml reference simulation
"""
import argparse
import os.path
import shutil
import math
from nineline.cells.neuron import NineCellMetaClass, simulation_controller
from nineline.cells.build import BUILD_MODE_OPTIONS
from nineline.arguments import outputpath
from neurotune import Parameter
from neurotune.tuner import EvaluationException
from neurotune.objective.phase_plane import (PhasePlaneHistObjective,
                                             PhasePlanePointwiseObjective)
from neurotune.objective.multi import MultiObjective, WeightedSumObjective
from neurotune.objective.spike import (SpikeFrequencyObjective,
                                       SpikeTimesObjective)
from neurotune.algorithm.inspyred import ec, algorithm_types, replacer_types
from neurotune.simulation.nineline import NineLineSimulation
try:
    from neurotune.tuner.mpi import MPITuner as Tuner
except ImportError:
    from neurotune.tuner import Tuner
import cPickle as pkl

true_parameters = []

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('reference_9ml', type=str,
                       help="The path of the 9ml cell model to be used as a "
                            "reference")
parser.add_argument('to_tune_9ml', type=str,
                       help="The path of the 9ml cell to tune")
parser.add_argument('--build', type=str, default='lazy',
                    help="Option to build the NMODL files before running (can "
                         "be one of {})".format(BUILD_MODE_OPTIONS))
parser.add_argument('--timestep', type=float, default=0.025,
                    help="The timestep used for the simulation "
                         "(default: %(default)s)")
parser.add_argument('--time', type=float, default=2000.0,
                       help="Recording time")
parser.add_argument('--output', type=outputpath,
                    default=os.path.join(os.environ['HOME'], 'tuned.pkl'),
                    help="The path to the output file where the grid will be "
                         "written (default: %(default)s)")
parser.add_argument('-o', '--objective', type=str, nargs='+',
                    default=[], action='append',
                    help="Selects which objective function to use "
                         "out of 'histogram', 'pointwise', 'frequency', "
                         "'spike_times' or a combination (potentially "
                         "weighted) of them (default: 'pointwise')")
parser.add_argument('-p', '--parameter', nargs=4, default=[], action='append',
                    metavar=('NAME', 'LBOUND', 'UBOUND', 'LOG_SCALE'),
                    help="Sets a parameter to tune and its lower and upper "
                         "bounds")
parser.add_argument('--parameter_set', type=str, default=[], nargs='+',
                    help="Select which parameter set to tune from a few "
                         "descriptions")
parser.add_argument('--num_generations', type=int, default=100,
                    help="The number of generations (iterations) to run the "
                         "algorithm for")
parser.add_argument('--population_size', type=int, default=100,
                    help="The number of genomes in a generation")
parser.add_argument('--algorithm', type=str, default='eda',
                    help="The type of algorithm used for the tuning. Can be "
                         "one of '{}' (default: %(default)s)"
                         .format("', '". join(algorithm_types.keys())))
parser.add_argument('-a', '--optimize_argument', nargs=2, action='append',
                    default=[], metavar=('KEY', 'ARG'),
                    help="Extra arguments to be passed to the algorithm")
parser.add_argument('-b', '--objective_argument', nargs=3, action='append',
                    metavar=('KEY', 'ARG', 'OBJECTIVE_INDEX'), default=[],
                    help="Extra keyword arguments to pass to the objective "
                         "function (can specify which objective in a "
                         "objective method by the third \"index\" argument")
parser.add_argument('--action', type=str, nargs='+', default=['tune'],
                    help="The action used to run the script, can be 'tune', "
                         "'plot'")
parser.add_argument('--verbose', action='store_true', default=False,
                    help="Whether to print out which candidates are being "
                    "evaluated on which nodes")
parser.add_argument('--replacer', type=str, default=None,
                    help="The replacement component of the evolutionary "
                         "algorithm. Can be one of ('{}')"
                         .format("', '". join(replacer_types.keys())))

obj_dict = {'histogram': PhasePlaneHistObjective,
            'pointwise': PhasePlanePointwiseObjective,
            'frequency': SpikeFrequencyObjective,
            'spike_times': SpikeTimesObjective}


def get_objective(args):
    # Generate the reference trace from the original class
    cell = NineCellMetaClass(args.reference_9ml, build_mode=args.build)()
    cell.record('v')
    simulation_controller.run(simulation_time=args.time,
                              timestep=args.timestep)
    reference = cell.get_recording('v')
    # Distribute the objective arguments between the (possibly) multiple
    # objectives
    objective_args = [{} for _ in xrange(len(args.objective))]
    for oa in args.objective_argument:
        try:
            index = oa[2]
        except IndexError:
            index = 0
        objective_args[index][oa[0]] = oa[1]
    try:
        # Use a mult-objective, or weighted sum objective depending on the
        # algorithm selected
        if len(args.objective) > 1:
            objs = [obj_dict[o[0]](reference, **kwargs)
                    for (o, kwargs) in zip(args.objective, objective_args)]
            if args.algorithm in ('nsga2', 'pareto_archived'):
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
            objective = WeightedSumObjective(
                                 (1.0, PhasePlanePointwiseObjective(reference),
                                 (75.0, SpikeFrequencyObjective(reference))))
    except KeyError as e:
        raise Exception("Unrecognised objective '{}' passed to '--objective' "
                        "option".format(e))
    return objective


def _get_algorithm(args):
    try:
        Algorithm = algorithm_types[args.algorithm]
    except KeyError:
        raise Exception("Unrecognised algorithm '{}'".format(args.algorithm))
    kwargs = dict(args.optimize_argument)
    if args.replacer:
        kwargs['replacer'] = replacer_types[args.replacer]
    return Algorithm(args.population_size,
                     max_generations=args.num_generations,
                     observer=[ec.observers.population_observer],
                     output_dir=os.path.dirname(args.output), **kwargs)


def _get_parameters(args):
    if args.parameter and args.parameter_set:
        raise Exception("Cannot use --parameter and --parameter_set options "
                        "simulataneously")
    if args.parameter:
        parameters = [Parameter(p[0], 'S/cm^2', p[1], p[2], p[3])
                      for p in args.parameter]
    # The parameters to be tuned by the tuner
    elif args.parameter_set[0] == 'all-gmaxes':
        bound_range = float(args.parameter_set[1])
        from nineml.extensions.biophysics import parse
        bio_model = next(parse(args.reference_9ml).itervalues())
        parameters = []
        for comp in bio_model.components.itervalues():
            if comp.type == 'ionic-current':
                gbar = float(comp.parameters['g'].value)
                gbar_log = math.log(gbar)
                lbound = gbar_log - bound_range
                ubound = gbar_log + bound_range
                parameters.append(Parameter('soma.{}.gbar'.format(comp.name),
                                            'S/cm^2', lbound, ubound,
                                            log_scale=True))
                true_parameters.append(gbar)
    elif args.parameter_set:
        raise Exception("Unrecognised name '{}' passed to '--parameter_set' "
                        "option. Can be one of ('original', 'all-gmaxes')."
                        .format(args.parameter_set))
    else:
        raise Exception("No --parameter or --parameter set arguments passed "
                        "to tuning script")
    return parameters


def _get_simulation(args, parameters=None, objective=None):
    simulation = NineLineSimulation(args.to_tune_9ml, build_mode=args.build)
    if parameters is not None:
        simulation.set_tune_parameters(parameters)
    if objective is not None:
        simulation._process_requests(objective.get_recording_requests())
    return simulation


def run(args):
    # Instantiate the tuner
    parameters = _get_parameters(args)
    algorithm = _get_algorithm(args)
    objective = get_objective(args)
    simulation = _get_simulation(args)
    tuner = Tuner(parameters,
                  objective,
                  algorithm,
                  simulation,
                  verbose=args.verbose)
    tuner.true_candidate = true_parameters
    # Run the tuner
    try:
        pop, _ = tuner.tune()
    except EvaluationException as e:
        e.save(os.path.join(os.path.dirname(args.output),
                            'evaluation_exception.pkl'))
        raise
    # Save the file if the tuner is the master
    if tuner.is_master():
        fittest_individual = min(pop, key=lambda c: c.fitness)
        print ("Fittest candidate (fitness {}): {}"
              .format(fittest_individual.fitness,
                      fittest_individual.candidate))
        # Save the grid to file
        with open(args.output, 'w') as f:
            pkl.dump((fittest_individual.candidate, fittest_individual.fitness,
                      pop), f)


def record_candidate(candidate_path, filepath, args):
    with open(candidate_path) as f:
        candidate, _, _ = pkl.load(f)
    parameters = _get_parameters(args)
    objective = get_objective(args)
    simulation = _get_simulation(args, parameters=parameters,
                                 objective=objective)
    recordings = simulation.run_all(candidate[:len(parameters)])
    with open(filepath, 'w') as f:
        pkl.dump((recordings.segments[0].analogsignals[0], objective), f)


def plot(recordings_path):
    from matplotlib import pyplot as plt
    with open(recordings_path) as f:
        recording, objective = pkl.load(f)
    plt.plot(recording)
    plt.plot(objective.reference_traces[0])
    objective.plot_hist(recording, diff=True, show=False)
    plt.show()


def prepare_work_dir(submitter, args):
    os.mkdir(os.path.join(submitter.work_dir, '9ml'))
    copied_reference = os.path.join(submitter.work_dir, '9ml',
                                    os.path.basename(args.reference_9ml))
    shutil.copy(args.reference_9ml, copied_reference)
    copied_to_tune = os.path.join(submitter.work_dir, '9ml',
                                  os.path.basename(args.to_tune_9ml))
    shutil.copy(args.to_tune_9ml, copied_to_tune)
    NineCellMetaClass(copied_reference, build_mode='build_only')
    NineCellMetaClass(copied_to_tune, build_mode='build_only')
    args.reference_9ml = copied_reference
    args.to_tune_9ml = copied_to_tune


if __name__ == '__main__':
    args = parser.parse_args()
    if (Tuner.num_processes - 1) > args.population_size:
        args.population_size = Tuner.num_processes - 1
        print ("Warning population size was automatically increased to {} in "
               "order to match the number of processes"
               .format(args.population_size))
    if args.action[0] == 'plot':
        plot(args.action[1])
    elif args.action[0] == 'record':
        record_candidate(args.action[1], args.action[2], args)
    else:
        run(args)
