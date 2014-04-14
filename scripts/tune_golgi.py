#!/usr/bin/env python
"""
Evaluates objective functions on a grid of positions in parameter space
"""
import os.path
import argparse
import shutil
from nineline.cells.neuron import NineCellMetaClass, simulation_controller
from nineline.cells.build import BUILD_MODE_OPTIONS
from neurotune import Parameter
from neurotune.tuner import EvaluationException
# from neurotune.objective.multi import MultiObjective
from neurotune.objective.phase_plane import (PhasePlaneHistObjective, 
                                             ConvPhasePlaneHistObjective, 
                                             PhasePlanePointwiseObjective)
from neurotune.algorithm.evolutionary import EDAAlgorithm
from neurotune.simulation.nineline import NineLineSimulation
import cPickle as pkl

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('cell_9ml', type=str,
                       help="The path of the 9ml cell to test the objective function on"
                            "(default: %(default)s)") 
parser.add_argument('--build', type=str, default='lazy', 
                    help="Option to build the NMODL files before running (can be one of {})"
                         .format(BUILD_MODE_OPTIONS))
parser.add_argument('--timestep', type=float, default=0.025, 
                    help="The timestep used for the simulation (default: %(default)s)")
parser.add_argument('--time', type=float, default=2000.0,
                       help="Recording time")
parser.add_argument('--output', type=str, default=os.path.join(os.environ['HOME'], 'tuned.pkl'),
                       help="The path to the output file where the grid will be written "
                            "(default: %(default)s)")
parser.add_argument('--objective', type=str, default='convolved',
                    help="Selects which objective function to use "
                         "('vanilla', 'convolved', 'pointwise')")

# The parameters to be tuned by the tuner
parameters = [Parameter('diam', 'um', 20.0, 40.0),
              Parameter('soma.Lkg.gbar', 'S/cm^2', -6, -4, log_scale=True),
              Parameter('soma.Lkg.e_rev', 'mV', -70, 45)] #1e-5, 3e-5)]

objective_names = ['Phase-plane original', 'Convolved phase-plane', 'Pointwise phase-plane']

def run(args):
    from neurotune.tuner.mpi import MPITuner as Tuner # @Reimport  
    # Generate the reference trace from the original class
    cell = NineCellMetaClass(args.cell_9ml)()
    cell.record('v')
    simulation_controller.run(simulation_time=args.time, timestep=args.timestep)
    reference_trace = cell.get_recording('v')
    # Select which objective function to use
    if args.objective == 'vanilla':
        objective = PhasePlaneHistObjective(reference_trace)
    elif args.objective == 'convolved':
        objective = ConvPhasePlaneHistObjective(reference_trace)
    elif args.objective == 'pointwise':
        objective = PhasePlanePointwiseObjective(reference_trace, (20, -20), 100)
    else:
        raise Exception("Unrecognised objective '{}' passed to '--objective' option"
                        .format(args.objective))
    # Instantiate the tuner
    tuner = Tuner(parameters,
                  objective,
                  EDAAlgorithm(),
                  NineLineSimulation(args.cell_9ml))
    # Run the tuner
    try:
        pop, ea = tuner.tune()
    except EvaluationException as e:
        e.save(os.path.join(os.path.dirname(args.output), 'evaluation_exception.pkl'))
        raise
    # Save the file if the tuner is the master
    if tuner.is_master():
        print "Fittest candidate {}".format(pop)
        # Save the grid to file
        with open(args.output, 'w') as f:
            pkl.dump((pop,ea), f)
         

def prepare_work_dir(work_dir, args):
    os.mkdir(os.path.join(work_dir, '9ml'))
    copied_9ml = os.path.join(work_dir, '9ml', os.path.basename(args.cell_9ml))
    shutil.copy(args.cell_9ml, copied_9ml)
    NineCellMetaClass(copied_9ml, build_mode='build_only')
    args.cell_9ml = copied_9ml

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
