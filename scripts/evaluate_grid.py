#!/usr/bin/env python
"""
Evaluates objective functions on a grid of positions in parameter space
"""
import os.path
import argparse
import numpy.ma
import shutil
import cPickle as pkl
import quantities as pq
from nineline.cells.neuron import NineCellMetaClass, simulation_controller
from nineline.cells.build import BUILD_MODE_OPTIONS
from nineline.arguments import outputpath
from neurotune import Parameter
from neurotune.tuner import EvaluationException
from neurotune.objective.multi import MultiObjective
from neurotune.objective.phase_plane import (PhasePlaneHistObjective,
                                             PhasePlanePointwiseObjective)
from neurotune.objective.spike import (SpikeFrequencyObjective,
                                       SpikeTimesObjective)
from neurotune.algorithm.grid import GridAlgorithm
from neurotune.simulation.nineline import NineLineSimulation
from neurotune.analysis import AnalysedSignal
try:
    from neurotune.tuner.mpi import MPITuner as Tuner
except ImportError:
    from neurotune.tuner import Tuner  # @Reimport

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('cell_9ml', type=str,
                    help="The path of the 9ml cell to test the objective "
                         "function on")
parser.add_argument('--disable_mpi', action='store_true',
                    help="Disable MPI tuner and replace with basic tuner")
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
parser.add_argument('--plot', action='store_true',
                    help="Plot the grid on a 1-2d mesh")
parser.add_argument('--plot_saved', nargs='*', default=[],
                    help="Plot a file that has been saved to file already")
parser.add_argument('--verbose', action='store_true', default=False,
                    help="Print out which candidates are being evaluated")
parser.add_argument('--save_recordings', type=outputpath, default=None,
                    metavar='DIRECTORY',
                    help="Save recordings to file")

# # The parameters to be tuned by the tuner
# parameters = [Parameter('diam', 'um', 20.0, 40.0),
#               Parameter('soma.Lkg.gbar', 'S/cm^2', -6, -4, log_scale=True)]
# 1e-5, 3e-5)]

objective_names = ['Phase-plane Histogram', 'Phase-plane Pointwise',
                   'Spike Frequency', 'Spike Times']


def run(parameters, args):
    # Generate the reference trace from the original class
    cell = NineCellMetaClass(args.cell_9ml)()
    cell.record('v')
    simulation_controller.run(simulation_time=args.time,
                              timestep=args.timestep)
    reference = AnalysedSignal(cell.get_recording('v'))
    sliced_reference = reference.slice(500 * pq.ms, 2000 * pq.ms)
    # Instantiate the multi-objective objective from 3 phase-plane objectives
    objective = MultiObjective(PhasePlaneHistObjective(reference),
                               PhasePlanePointwiseObjective(reference, 100,
                                                            (20, -20)),
                               SpikeFrequencyObjective(sliced_reference.\
                                                       spike_frequency()),
                               SpikeTimesObjective(sliced_reference.\
                                                   spikes()))
    # Instantiate the tuner
    tuner = Tuner(parameters,
                  objective,
                  GridAlgorithm(num_steps=[p[3] for p in args.parameter]),
                  NineLineSimulation(args.cell_9ml),
                  verbose=args.verbose,
                  save_recordings=args.save_recordings)
    # Run the tuner
    try:
        pop, grid = tuner.tune()
    except EvaluationException as e:
        e.save(os.path.join(os.path.dirname(args.output),
                            'evaluation_exception.pkl'))
        raise
    # Save the file if the tuner is the master
    if tuner.is_master():
        print "Fittest candidate {}".format(pop)
        # Save the grid to file
        with open(args.output, 'w') as f:
            pkl.dump(grid, f)
        # Plot the grid if asked
        if args.plot:
            plot(grid)
        else:
            print ("Saved grid file '{out}' can be plotted using the command: "
                   "\n {script_name} {cell9ml} {params} --plot_saved {out}"
                   .format(cell9ml=args.cell_9ml,
                           script_name=os.path.basename(__file__),
                           params=' '.join(['-p ' + ' '.join(p)
                                            for p in args.parameter]),
                           out=args.output))


def plot(grids, parameters, plot_type='image', trim_factor=None):
    # Import the plotting modules here so they are not imported unless plotting
    # is required
    from mpl_toolkits.mplot3d import Axes3D  # @UnusedImport
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    import matplotlib.pyplot as plt
    import matplotlib
    # If using a non-multi-objective reshape the grid into a 1-?-? so it fits
    # the looping structure
    if grids.ndim == len(parameters):
        grids = [grids]
    # Loop through all grids and plot a surface mesh
    for grid, title in zip(grids, objective_names):
        fig = plt.figure()
        if len(parameters) == 1:
            x_range = numpy.linspace(parameters[0].lbound,
                                     parameters[0].ubound,
                                     grid.shape[0])
            if parameters[0].log_scale:
                x_range = 10 ** x_range
            plt.plot(x_range, grid)
            plt.xlabel('{} ({})'.format(parameters[0].name,
                                        parameters[0].units))
            plt.ylabel('Objective')
        elif len(parameters) == 2:
            x_range = numpy.linspace(parameters[0].lbound,
                                     parameters[0].ubound,
                                     grid.shape[0])
            y_range = numpy.linspace(parameters[1].lbound,
                                     parameters[1].ubound,
                                     grid.shape[1])
            if parameters[0].log_scale:
                x_range = 10 ** x_range
            if parameters[1].log_scale:
                y_range = 10 ** y_range
            kwargs = {}
            max_under_trim = None
            if trim_factor is not None:
                trim_value = (float(trim_factor) *
                              numpy.percentile(grid, 95))
                if numpy.max(grid) > trim_value:
                    over_trim = grid > trim_value
                    max_under_trim = numpy.ma.masked_array(grid,
                                                          mask=over_trim).max()
            if plot_type == 'surf':
                if max_under_trim is not None:
                    grid[numpy.where(over_trim)] = float('nan')
                    lev = numpy.linspace(0, max_under_trim, 1000)
                    kwargs['norm'] = matplotlib.colors.BoundaryNorm(lev, 256)

                ax = fig.gca(projection='3d')
                X, Y = numpy.meshgrid(x_range, y_range)
                surf = ax.plot_surface(X, Y, grid, rstride=1, cstride=1,
                                       cmap=cm.jet, linewidth=0,
                                       antialiased=False, **kwargs)
                if max_under_trim is not None:
                    ax.set_zlim(0, max_under_trim)
                ax.zaxis.set_major_locator(LinearLocator(10))
                fig.colorbar(surf, shrink=0.5, aspect=5)
                ax.set_zlabel('Objective')
            elif plot_type == 'image':
                plt.imshow(grid, interpolation='nearest', vmax=max_under_trim,
                           origin='lower', aspect='auto',
                           extent=(parameters[0].lbound, parameters[0].ubound,
                                   parameters[1].lbound, parameters[1].ubound))
                plt.grid()
                plt.colorbar()
                plt.clim(0, max_under_trim)
            else:
                raise Exception("Unrecognised plot_type '{}'"
                                .format(plot_type))
            plt.xlabel('{} ({})'.format(parameters[0].name,
                                        parameters[0].units))
            plt.ylabel('{} ({})'.format(parameters[1].name,
                                        parameters[1].units))
        else:
            raise Exception("Cannot plot grids with dimensions greater than 2")
        plt.title('{} objective'.format(title))
    plt.show()


def prepare_work_dir(submitter, args):
    os.mkdir(os.path.join(submitter.work_dir, '9ml'))
    copied_9ml = os.path.join(submitter.work_dir, '9ml',
                              os.path.basename(args.cell_9ml))
    shutil.copy(args.cell_9ml, copied_9ml)
    NineCellMetaClass(copied_9ml, build_mode='build_only')
    args.cell_9ml = copied_9ml


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.parameter:
        raise Exception("At least one parameter argument '--parameter' needs "
                        "to be supplied")
    parameters = [Parameter(name, 'S/cm^2', lbound, ubound, log_scale)
                  for name, lbound, ubound, _, log_scale in args.parameter]
    if args.plot_saved:
        with open(args.plot_saved[0]) as f:
            plot(pkl.load(f), parameters, *args.plot_saved[1:])
    else:
        run(parameters, args)
