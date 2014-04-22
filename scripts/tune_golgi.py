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
try:
    from neurotune.tuner.mpi import MPITuner as Tuner
except ImportError:
    from neurotune.tuner import Tuner
import cPickle as pkl

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('reference_9ml', type=str,
                       help="The path of the 9ml cell model to be used as a reference")
parser.add_argument('to_tune_9ml', type=str,
                       help="The path of the 9ml cell to tune") 
parser.add_argument('--build', type=str, default='lazy', 
                    help="Option to build the NMODL files before running (can be one of {})"
                         .format(BUILD_MODE_OPTIONS))
parser.add_argument('--disable_resampling', action='store_true', 
                    help="Disables the resampling of the traces before the histograms are calcualted")
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
parser.add_argument('--parameter_set', type=str, default='all-gmaxes',
                    help="Select which parameter set to tune from a few descriptions")
objective_names = ['Phase-plane original', 'Convolved phase-plane', 'Pointwise phase-plane']

def run(args):
    # Generate the reference trace from the original class
    cell = NineCellMetaClass(args.reference_9ml)()
    cell.record('v')
    simulation_controller.run(simulation_time=args.time, timestep=args.timestep)
    reference_trace = cell.get_recording('v')
    # Select which objective function to use
    obj_kwargs =  {}
    if args.disable_resampling:
        obj_kwargs['resample'] = False
    if args.objective == 'vanilla':
        objective = PhasePlaneHistObjective(reference_trace, **obj_kwargs)
    elif args.objective == 'convolved':
        objective = ConvPhasePlaneHistObjective(reference_trace, **obj_kwargs)
    elif args.objective == 'pointwise':
        objective = PhasePlanePointwiseObjective(reference_trace, (20, -20), 100, **obj_kwargs)
    else:
        raise Exception("Unrecognised objective '{}' passed to '--objective' option"
                        .format(args.objective))
    # The parameters to be tuned by the tuner
    if args.parameter_set == 'original':
        parameters = [Parameter('soma.Lkg.gbar', 'S/cm^2', 20.0, 40.0),
                      ] #1e-5, 3e-5)]
    elif args.parameter_set == 'all-gmaxes':
        parameters = [Parameter('soma.KA.g', 'S/cm^2', 0.0008, 0.08),
                      Parameter('soma.HCN2.g', 'S/cm^2', 8e-06, 0.0008),
                      Parameter('soma.KCa.g', 'S/cm^2', 0.0003, 0.03),
                      Parameter('soma.Lkg.g', 'S/cm^2', 2.1e-06, 0.00021),
                      Parameter('soma.SK2.g', 'S/cm^2', 0.0038, 0.38),
                      Parameter('soma.HCN1.g', 'S/cm^2', 5e-06, 0.0005),
                      Parameter('soma.NaBase.g', 'S/cm^2', 0.0048, 0.48),
                      Parameter('soma.KM.g', 'S/cm^2', 0.0001, 0.01),
                      Parameter('soma.NaR.g', 'S/cm^2', 0.00017, 0.017),
                      Parameter('soma.NaP.g', 'S/cm^2', 1.9e-05, 0.0019),
                      Parameter('soma.KV.g', 'S/cm^2', 0.0032, 0.32),
                      Parameter('soma.CaHVA.g', 'S/cm^2', 4.6e-05, 0.0046),
                      Parameter('soma.CaLVA.g', 'S/cm^2', 2.5e-05, 0.0025)]
    else:
        raise Exception("Unrecognised name '{}' passed to '--parameter_set' option. Can be one of "
                        "('original', 'all-gmaxes').".format(args.parameter_set))
    # Instantiate the tuner
    tuner = Tuner(parameters,
                  objective,
                  EDAAlgorithm(),
                  NineLineSimulation(args.to_tune_9ml))
    # Run the tuner
    try:
        pop, _ = tuner.tune()
    except EvaluationException as e:
        e.save(os.path.join(os.path.dirname(args.output), 'evaluation_exception.pkl'))
        raise
    # Save the file if the tuner is the master
    if tuner.is_master():
        print "Fittest candidate {}".format(pop)
        # Save the grid to file
        with open(args.output, 'w') as f:
            pkl.dump(pop, f)
         

def prepare_work_dir(work_dir, args):
    os.mkdir(os.path.join(work_dir, '9ml'))
    copied_9ml = os.path.join(work_dir, '9ml', os.path.basename(args.cell_9ml))
    shutil.copy(args.cell_9ml, copied_9ml)
    NineCellMetaClass(copied_9ml, build_mode='build_only')
    args.cell_9ml = copied_9ml

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
