#!/usr/bin/env python
"""
Evaluates objective functions on a grid of positions in parameter space
"""
import os.path
import argparse
import numpy
try:
    from nineline.cells.build import BUILD_MODE_OPTIONS
except ImportError:
    BUILD_MODE_OPTIONS = []

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument('cell_9ml', type=str, 
                       default=os.path.join('/home', 'tclose', 'git', 'kbrain', '9ml', 'neurons', 
                                            'Golgi_Solinas08.9ml'),
                       help="The path of the 9ml cell to test the objective function on"
                            "(default: %(default)s)") 
argparser.add_argument('--num_steps', type=int, default=2, 
                    help="The number of grid steps to take along each dimension "
                         "(default: %(default)s)")
argparser.add_argument('--disable_mpi', action='store_true', 
                    help="Disable MPI tuner and replace with basic tuner")
argparser.add_argument('--simulator', type=str, default='neuron', 
                    help="simulator for NINEML+ (either 'neuron' or 'nest')")
argparser.add_argument('--build', type=str, 'lazy', 
                    help="Option to build the NMODL files before running (can be one of {})"
                         .format(BUILD_MODE_OPTIONS))
argparser.add_argument('--timestep', type=float, default=0.025, 
                    help="The timestep used for the simulation (default: %(default)s)")
argparser.add_argument('--time', type=float, default=2000.0,
                       help="Recording time")
argparser.add_argument('--output', type=str, default=os.path.join(os.environ['HOME'], 'grid.pkl'),
                       help="The path to the output file where the grid will be written "
                            "(default: %(default)s)")
argparser.add_argument('--plot', action='store_true', help="Plot the grid on a 1-2d mesh")

def main():
    args = argparser.parse_args()    
    if args.disable_mpi:
        from neurotune import Tuner  # @UnusedImport 
    else:
        from neurotune.tuner.mpi import MPITuner as Tuner # @Reimport  
    from nineline.cells.neuron import NineCellMetaClass, simulation_controller
    from neurotune import Parameter
    from neurotune.objective.multi import MultiObjective
    from neurotune.objective.phase_plane import (PhasePlaneHistObjective, 
                                                 ConvPhasePlaneHistObjective, 
                                                 PhasePlanePointwiseObjective)
    from neurotune.algorithm.grid import GridAlgorithm
    from neurotune.simulation.nineline import NineLineSimulation
    import cPickle as pkl
    from matplotlib import pyplot as plt
    
    # Generate the reference trace from the original class
    cell = NineCellMetaClass(args.cell_9ml)()
    cell.record('v')
    simulation_controller.run(simulation_time=args.time, timestep=args.timestep)
    reference_trace = cell.get_recording('v')
    
    parameters = [Parameter('diam', 'um', 10.0, 40.0),
                  Parameter('soma.Lkg.gbar', 'S/cm^2', 1e-5, 3e-5)]
    
    objective = MultiObjective(PhasePlaneHistObjective(reference_trace), 
                               ConvPhasePlaneHistObjective(reference_trace),
                               PhasePlanePointwiseObjective(reference_trace, (20, -20), 100))
    
    # Instantiate the tuner
    tuner = Tuner(parameters,
                  objective,
                  GridAlgorithm(num_steps=args.num_steps),
                  NineLineSimulation(args.cell_9ml))

    # Run the tuner
    pop, grid = tuner.tune()
    
    if tuner.is_master():
        print "Fittest candidate {}".format(pop)
        
        # Save the grid to file
        with open(args.output, 'w') as f:
            pkl.dump(grid, f)
            
        # Plot the grid if asked
        if args.plot:
            if len(parameters) == 1:
                plt.plot(numpy.linspace(parameters[0].lbound, parameters[0].ubound, args.num_steps),
                         grid)
            elif len(parameters) == 2:
                plt.imshow(grid, interpolation='nearest', origin='lower', aspect='auto',
                           extent=(parameters[0].lbound, parameters[0].ubound, 
                                   parameters[1].lbound, parameters[1].ubound))
            else:
                raise Exception("Plot is only supported number of parameters <= 2 (found {})"
                                .format(len(parameters)))
            plt.show()

if __name__ == '__main__':
    main()
