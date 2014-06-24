#!/usr/bin/env python
"""
Evaluates objective functions on a grid of positions in parameter space
"""
import os.path
import argparse
import sys
import shutil
import cPickle as pkl
from nineline.cells.neuron import NineCellMetaClass
from nineline.cells.build import BUILD_MODE_OPTIONS
from nineline.arguments import outputpath
from neurotune import Parameter
from neurotune.tuner import EvaluationException
from neurotune.algorithm.grid import GridAlgorithm
from neurotune.simulation.nineline import NineLineSimulation
try:
    from neurotune.tuner.mpi import MPITuner as Tuner
except ImportError:
    from neurotune.tuner import Tuner  # @Reimport
# Import parser from evaluate_grid
sys.path.insert(0, os.path.dirname(__file__))
from tune_9ml import get_objective
sys.path.pop(0)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('reference_9ml', type=str,
                    help="The path of the 9ml cell to test the objective "
                         "function on")
parser.add_argument('--build', type=str, default='lazy',
                    help="Option to build the NMODL files before running (can "
                         "be one of {})".format(BUILD_MODE_OPTIONS))
parser.add_argument('--timestep', type=float, default=0.025,
                    help="The timestep used for the simulation "
                         "(default: %(default)s)")
parser.add_argument('--time', type=float, default=2000.0,
                       help="Recording time")
parser.add_argument('--output', type=outputpath,
                    default=os.path.join(os.environ['HOME'], 'grid.pkl'),
                    help="The path to the output file where the grid will be "
                         "written (default: %(default)s)")
parser.add_argument('-p', '--parameter', nargs=5, default=[], action='append',
                    metavar=('NAME', 'LBOUND', 'UBOUND', 'NUM_STEPS',
                             'LOG_SCALE'),
                    help="Sets a parameter to tune and its lower and upper "
                         "bounds")
parser.add_argument('-o', '--objective', type=str, nargs='+',
                    default=[], action='append',
                    help="Selects which objective function to use "
                         "out of 'histogram', 'pointwise', 'frequency', "
                         "'spike_times' or a combination (potentially "
                         "weighted) of them (default: 'pointwise')")
parser.add_argument('--verbose', action='store_true', default=False,
                    help="Print out which candidates are being evaluated")
parser.add_argument('--save_recordings', type=outputpath, default=None,
                    metavar='DIRECTORY',
                    help="Save recordings to file")
parser.add_argument('--algorithm', type=str, default='grid',
                    help="Can be either 'grid' for all in one grid or"
                         "'multi-grid' for separate grids for each "
                         "objective function")
objective_names = ['Phase-plane Histogram', 'Phase-plane Pointwise',
                   'Spike Frequency', 'Spike Times']


def get_parameters(args):
    if not args.parameter:
        raise Exception("At least one parameter argument '--parameter' needs "
                        "to be supplied")
    parameters = [Parameter(name, 'S/cm^2', lbound, ubound, log_scale)
                  for name, lbound, ubound, _, log_scale in args.parameter]
    return parameters


def run(args):
    parameters = get_parameters(args)
    args.algorithm = 'eda'
    objective = get_objective(args)
    # Generate the reference trace from the original class
    # Instantiate the tuner
    tuner = Tuner(parameters,
                  objective,
                  GridAlgorithm(num_steps=[p[3] for p in args.parameter]),
                  NineLineSimulation(args.reference_9ml),
                  verbose=args.verbose,
                  save_recordings=args.save_recordings)
    # Run the tuner
    try:
        candidate, fitness, grid = tuner.tune()
    except EvaluationException as e:
        e.save(os.path.join(os.path.dirname(args.output),
                            'evaluation_exception.pkl'))
        raise
    # Save the file if the tuner is the master
    if tuner.is_master():
        print "Fittest candidate ({}) {}".format(fitness, candidate)
        # Save the grid to file
        with open(args.output, 'w') as f:
            pkl.dump(grid, f)
        print ("Saved grid file '{out}' can be plotted using the command: "
               "\n {script_name} {cell9ml} {params} --plot_saved {out}"
               .format(cell9ml=args.reference_9ml,
                       script_name=os.path.basename(__file__),
                       params=' '.join(['-p ' + ' '.join(p)
                                        for p in args.parameter]),
                       out=args.output))


def prepare_work_dir(submitter, args):
    os.mkdir(os.path.join(submitter.work_dir, '9ml'))
    copied_9ml = os.path.join(submitter.work_dir, '9ml',
                              os.path.basename(args.reference_9ml))
    shutil.copy(args.reference_9ml, copied_9ml)
    NineCellMetaClass(copied_9ml, build_mode='build_only')
    args.reference_9ml = copied_9ml


if __name__ == '__main__':
    args = parser.parse_args()
    args.algorithm = 'mult-grid'
    run(args)
