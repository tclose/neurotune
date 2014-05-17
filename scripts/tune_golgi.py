#!/usr/bin/env python
"""
Evaluates objective functions on a grid of positions in parameter space
"""
import argparse
import shutil
import math
from nineline.cells.neuron import NineCellMetaClass, simulation_controller
from nineline.cells.build import BUILD_MODE_OPTIONS
from neurotune import Parameter
from neurotune.tuner import EvaluationException
# from neurotune.objective.multi import MultiObjective
from neurotune.objective.phase_plane import (PhasePlaneHistObjective, 
                                             ConvPhasePlaneHistObjective, 
                                             PhasePlanePointwiseObjective)
from neurotune.algorithm.inspyred import *  # @UnusedWildImport

from neurotune.simulation.nineline import NineLineSimulation
try:
    from neurotune.tuner.mpi import MPITuner as Tuner
except ImportError:
    from neurotune.tuner import Tuner
import cPickle as pkl

algorithm_types = ['genetic', 'estimation_distr', 'evolution_strategy', 
                   'differential', 'simulated_annealing', 'nsga2', 
                   'pareto_archived']

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
parser.add_argument('--disable_resampling', action='store_true', 
                    help="Disables the resampling of the traces before the "
                         "histograms are calcualted")
parser.add_argument('--timestep', type=float, default=0.025, 
                    help="The timestep used for the simulation "
                         "(default: %(default)s)")
parser.add_argument('--time', type=float, default=2000.0,
                       help="Recording time")
parser.add_argument('--output', type=str, 
                    default=os.path.join(os.environ['HOME'], 'tuned.pkl'),
                    help="The path to the output file where the grid will be "
                         "written (default: %(default)s)")
parser.add_argument('--objective', type=str, default='convolved',
                    help="Selects which objective function to use "
                         "('vanilla', 'convolved', 'pointwise')")
parser.add_argument('--parameter_set', type=str, default=['all-gmaxes', 3.0], 
                    nargs='+', help="Select which parameter set to tune from a"
                                    " few descriptions")
parser.add_argument('--num_generations', type=int, default=100,
                    help="The number of generations (iterations) to run the "
                         "algorithm for")
parser.add_argument('--population_size', type=int, default=100,
                    help="The number of genomes in a generation")
parser.add_argument('--algorithm', type=str, default='evolution_strategy', 
                    help="The type of algorithm used for the tuning. Can be "
                         "one of '{}' (default: %(default)s)"
                         .format("', '". join(algorithm_types)))
parser.add_argument('-a', '--optimize_argument', nargs=2, action='append', 
                    default=[],
                    help="Extra arguments to be passed to the algorithm")
parser.add_argument('--action', type=str, nargs='+', default=['tune'],
                    help="The action used to run the script, can be 'tune', "
                         "'plot'")
parser.add_argument('--verbose', action='store_true', default=False,
                    help="Whether to print out which candidates are being "
                    "evaluated on which nodes") 
 
#objective_names = ['Phase-plane original', 'Convolved phase-plane', 
#                   'Pointwise phase-plane']

def _get_objective(args):
    # Generate the reference trace from the original class
    cell = NineCellMetaClass(args.reference_9ml, build_mode=args.build)()
    cell.record('v')
    simulation_controller.run(simulation_time=args.time, timestep=args.timestep)
    reference_trace = cell.get_recording('v')
    obj_kwargs =  {}
    if args.disable_resampling:
        obj_kwargs['sample_to_bin_ratio'] = False
    if args.objective == 'vanilla':
        objective = PhasePlaneHistObjective(reference_trace, **obj_kwargs)
    elif args.objective == 'convolved':
        objective = ConvPhasePlaneHistObjective(reference_trace, **obj_kwargs)
    elif args.objective == 'pointwise':
        objective = PhasePlanePointwiseObjective(reference_trace, (20, -20), 
                                                 100, **obj_kwargs)
    else:
        raise Exception("Unrecognised objective '{}' passed to '--objective' "
                        "option".format(args.objective))
    return objective

def _get_algorithm(args):
    if args.algorithm == 'genetic':
        Algorithm = GAAlgorithm
    elif args.algorithm == 'estimation_distr':
        Algorithm = EDAAlgorithm
    elif args.algorithm == 'evolution_strategy':
        Algorithm = ESAlgorithm
    elif args.algorithm == 'differential':
        Algorithm = DEAAlgorithm
    elif args.algorithm == 'simulated_annealing':
        Algorithm = SAAlgorithm
    elif args.algorithm == 'nsga2':
        Algorithm = NSGA2Algorithm
    elif args.algorithm == 'pareto_archived':
        Algorithm = PAESAlgorithm
    else:
        raise Exception("Unrecognised algorithm '{}'".format(args.algorithm))
    return Algorithm(args.population_size,
                     max_generations=args.num_generations,
                     terminator=ec.terminators.generation_termination,
                     observer=[ec.observers.best_observer, 
                               ec.observers.file_observer],
                     variator=[ec.variators.blend_crossover, 
                               ec.variators.gaussian_mutation],
                     output_dir=os.path.dirname(args.output), 
                     **dict(args.optimize_argument))
        
def _get_parameters(args):
    # The parameters to be tuned by the tuner
    if args.parameter_set[0] == 'original':
        parameters = [Parameter('soma.Lkg.gbar', 'S/cm^2', 20.0, 40.0),
                      ] #1e-5, 3e-5)]
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
                
    else:
        raise Exception("Unrecognised name '{}' passed to '--parameter_set' "
                        "option. Can be one of ('original', 'all-gmaxes')."
                        .format(args.parameter_set))
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
    objective = _get_objective(args)
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
              .format(fittest_individual.fitness, fittest_individual.candidate))
        # Save the grid to file
        with open(args.output, 'w') as f:
            pkl.dump((fittest_individual.candidate, fittest_individual.fitness,
                      pop), f)
         
def record_candidate(candidate_path, filepath, args):
    with open(candidate_path) as f:
        candidate, _, _ = pkl.load(f)
    parameters = _get_parameters(args)
    objective = _get_objective(args)
    simulation = _get_simulation(args, parameters=parameters, 
                                 objective=objective)
    recordings = simulation.run_all(candidate[:len(parameters)])
    with open(filepath, 'w') as f:
        pkl.dump((recordings.segments[0].analogsignals[0], objective) , f)
    
def plot(recordings_path):
    from matplotlib import pyplot as plt
    with open(recordings_path) as f:
        recording, objective = pkl.load(f)
    plt.plot(recording)
    plt.plot(objective.reference_traces[0])
    objective.plot_hist(recording, diff=True, show=False)
    plt.show()
    
def prepare_work_dir(work_dir, args):
    os.mkdir(os.path.join(work_dir, '9ml'))
    copied_reference = os.path.join(work_dir, '9ml', 
                                    os.path.basename(args.reference_9ml))
    shutil.copy(args.reference_9ml, copied_reference)
    copied_to_tune = os.path.join(work_dir, '9ml', 
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
