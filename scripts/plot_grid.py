import sys
import os.path
from copy import copy
import numpy
try:
    from mpl_toolkits.mplot3d import Axes3D  # @UnusedImport
except ImportError:
    pass
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import matplotlib
import cPickle as pkl
# Import parser from evaluate_grid
sys.path.insert(0, os.path.dirname(__file__))
from evaluate_grid import (parser as grid_parser, get_parameters,
                           objective_names)
sys.path.pop(0)

parser = copy(grid_parser)
parser.add_argument('--trim_factor', default=None, type=float,
                    help="The factor of the 95th percentile at which to "
                         "truncate the plot")
parser.add_argument('--plot_type', default='image',
                    help="The type of the plot can be either 'image' or 'surf'"
                         "(default: %(default)s)")

# Remove uneeded arguments
for argname in ('output', 'build', 'timestep', 'verbose', 'save_recordings'):
    try:
        parser._remove_action(next(a for a in parser._actions
                                   if a.dest == argname))
    except StopIteration:
        pass


def plot(args):
    with open(args.plot_saved[0]) as f:
        grids = pkl.load(f)
    parameters = get_parameters(args)
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
            if args.trim_factor is not None:
                trim_value = (float(args.trim_factor) *
                              numpy.percentile(grid, 95))
                if numpy.max(grid) > trim_value:
                    over_trim = grid > trim_value
                    max_under_trim = numpy.ma.masked_array(grid,
                                                          mask=over_trim).max()
            if args.plot_type == 'surf':
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
            elif args.plot_type == 'image':
                plt.imshow(grid, interpolation='nearest', vmax=max_under_trim,
                           origin='lower', aspect='auto',
                           extent=(parameters[0].lbound, parameters[0].ubound,
                                   parameters[1].lbound, parameters[1].ubound))
                plt.grid()
                plt.colorbar()
                plt.clim(0, max_under_trim)
            else:
                raise Exception("Unrecognised plot_type '{}'"
                                .format(args.plot_type))
            plt.xlabel('{} ({})'.format(parameters[0].name,
                                        parameters[0].units))
            plt.ylabel('{} ({})'.format(parameters[1].name,
                                        parameters[1].units))
        else:
            raise Exception("Cannot plot grids with dimensions greater than 2")
        plt.title('{} objective'.format(title))
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    plot(args)
