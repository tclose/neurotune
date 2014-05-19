#!/usr/bin/env python
"""
Evaluates objective functions on a grid of positions in parameter space
"""
import os.path
import argparse
import numpy.ma
import shutil
from nineline.cells.neuron import NineCellMetaClass, simulation_controller
from nineline.cells.build import BUILD_MODE_OPTIONS
from neurotune import Parameter
from neurotune.tuner import EvaluationException
from neurotune.objective.multi import MultiObjective
from neurotune.objective.phase_plane import (PhasePlaneHistObjective,
                                             PhasePlanePointwiseObjective)
from neurotune.objective.spike import (SpikeFrequencyObjective,
                                       SpikeTimesObjective)
from neurotune.algorithm import GridAlgorithm
from neurotune.simulation.nineline import NineLineSimulation
from neurotune.analysis import AnalysedSignal
import cPickle as pkl

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('cell_9ml', type=str,
                       help="The path of the 9ml cell to test the objective "
                            "function on (default: %(default)s)")
parser.add_argument('--num_steps', type=int, default=2,
                    help="The number of grid steps to take along each "
                         "dimension (default: %(default)s)")
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
parser.add_argument('--output', type=str,
                    default=os.path.join(os.environ['HOME'], 'grid.pkl'),
                    help="The path to the output file where the grid will be "
                         "written (default: %(default)s)")
parser.add_argument('--plot', action='store_true',
                    help="Plot the grid on a 1-2d mesh")
parser.add_argument('--plot_saved', nargs='*', default=False,
                    help="Plot a file that has been saved to file already")

# The parameters to be tuned by the tuner
parameters = [Parameter('diam', 'um', 20.0, 40.0),
              Parameter('soma.Lkg.gbar', 'S/cm^2', -6, -4, log_scale=True)]
# 1e-5, 3e-5)]

objective_names = ['Phase-plane original', 'Pointwise phase-plane']


def run(args):
    if args.disable_mpi:
        from neurotune import Tuner  # @UnusedImport
    else:
        from neurotune.tuner.mpi import MPITuner as Tuner  # @Reimport
    # Generate the reference trace from the original class
    cell = NineCellMetaClass(args.cell_9ml)()
    cell.record('v')
    simulation_controller.run(simulation_time=args.time,
                              timestep=args.timestep)
    reference = AnalysedSignal(cell.get_recording('v'))
    # Instantiate the multi-objective objective from 3 phase-plane objectives
    objective = MultiObjective(PhasePlaneHistObjective(reference),
                               PhasePlanePointwiseObjective(reference,
                                                            (20, -20), 100),
                               SpikeFrequencyObjective(reference.\
                                                       spike_frequency),
                               SpikeTimesObjective(reference.spike_times))
    # Instantiate the tuner
    tuner = Tuner(parameters,
                  objective,
                  GridAlgorithm(num_steps=args.num_steps),
                  NineLineSimulation(args.cell_9ml))
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
                   "\n {script_name} {cell9ml} --plot_saved {out}"
                   .format(cell9ml=args.cell_9ml,
                           script_name=os.path.basename(__file__),
                           out=args.output))


def plot(grids, plot_type='surf', trim_factor=None):
    # Import the plotting modules here so they are not imported unless plotting
    # is required
    from mpl_toolkits.mplot3d import Axes3D  # @UnusedImport
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    import matplotlib.pyplot as plt
    import matplotlib
    # If using a non-multi-objective reshape the grid into a 1-?-? so it fits
    # the looping structure
    if grids.ndim == 2:
        grids.reshape(1, grids.shape[0], grids.shape[1])
    # Loop through all grids and plot a surface mesh
    for grid, title in zip(grids, objective_names):
        fig = plt.figure()
        x_range = numpy.linspace(parameters[0].lbound, parameters[0].ubound,
                                 grid.shape[0])
        y_range = numpy.linspace(parameters[1].lbound, parameters[1].ubound,
                                 grid.shape[1])
        kwargs = {}
        max_under_trim = None
        if trim_factor is not None:
            trim_value = (trim_factor * numpy.percentile(grid, 95))
            if numpy.max(grid) > trim_value:
                over_trim = grid > trim_value
                max_under_trim = numpy.ma.masked_array(grid,
                                                       mask=over_trim).max()
                grid[numpy.where(over_trim)] = float('nan')
                lev = numpy.linspace(0, max_under_trim, 1000)
                kwargs['norm'] = matplotlib.colors.BoundaryNorm(lev, 256)
        if plot_type == 'surf':
            ax = fig.gca(projection='3d')
            X, Y = numpy.meshgrid(x_range, y_range)
            surf = ax.plot_surface(X, Y, grid, rstride=1, cstride=1,
                                   cmap=cm.jet, linewidth=0,
                                   antialiased=False, **kwargs)
            if max_under_trim is not None:
                ax.set_zlim(0, max_under_trim)
            ax.zaxis.set_major_locator(LinearLocator(10))
            fig.colorbar(surf, shrink=0.5, aspect=5)
        elif plot_type == 'image':
            plt.imshow(grid, interpolation='nearest', vmax=max_under_trim,
                       origin='lower', aspect='auto',
                       extent=(parameters[0].lbound, parameters[0].ubound,
                               parameters[1].lbound, parameters[1].ubound))
            plt.grid()
            plt.colorbar()
        else:
            raise Exception("Unrecognised plot_type '{}'".format(plot_type))
        plt.xlabel('{}{} ({})'.format('log_10 '
                                      if parameters[0].log_scale else '',
                                      parameters[0].name, parameters[0].units))
        plt.ylabel('{}{} ({})'.format('log_10 '
                                      if parameters[1].log_scale else '',
                                      parameters[1].name, parameters[1].units))
        plt.title('{} objective'.format(title))
    plt.show()


def prepare_work_dir(work_dir, args):
    os.mkdir(os.path.join(work_dir, '9ml'))
    copied_9ml = os.path.join(work_dir, '9ml', os.path.basename(args.cell_9ml))
    shutil.copy(args.cell_9ml, copied_9ml)
    NineCellMetaClass(copied_9ml, build_mode='build_only')
    args.cell_9ml = copied_9ml


if __name__ == '__main__':
    args = parser.parse_args()
    if args.plot_saved is not False:
        with open(args.cell_9ml) as f:
            plot(pkl.load(f), *args.plot_saved)
    else:
        run(args)
