#!/usr/bin/env python
"""
Reduces a model in 9ml format, by iteratively merging distal tree branches
and retuning the parameters
"""
import sys
import os.path
import argparse
import numpy
from copy import deepcopy
from lxml import etree
from nineml.extensions.biophysical_cells import parse as parse_nineml
from nineline.cells.build import BUILD_MODE_OPTIONS
from nineline.arguments import outputpath
from nineline.cells import (Model, IrreducibleMorphologyException,
                            IonChannelModel)
from neurotune import Parameter
from neurotune.tuner import EvaluationException
from neurotune.simulation.nineline import NineLineSimulation
try:
    from neurotune.tuner.mpi import MPITuner as Tuner
except ImportError:
    from neurotune.tuner import Tuner
from neurotune.algorithm import algorithm_factory
sys.path.insert(0, os.path.dirname(__file__))
from tune_9ml import run as tune, add_tune_arguments, get_objective
sys.path.pop(0)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('nineml', type=str,
                    help="The path of the 9ml cell to tune")
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
parser.add_argument('--parameter_range', type=float, default=0.5,
                    help="The range of the parameters about the previous "
                         "best fit to vary")
parser.add_argument('--fitness_bounds', nargs='+', type=float, default=None,
                    help="The bounds on the fitnesses below which the tree "
                         "will continue to be reduced")
# Add tuning arguments
add_tune_arguments(parser)


def run(args):
    # Get original model to be reduced
    model = Model.from_9ml(next(parse_nineml(args.nineml).itervalues()))
    # Create a copy to hold the reduced model
    reduced_model = deepcopy(model)
    # Get the objective functions based on the provided arguments and the
    # original model
    active_objective = get_objective(args, reference=args.nineml)
    passive_objective = get_objective(args, reference=args.nineml)
    algorithm = algorithm_factory(args)
    active_parameters = []
    new_states = []
    for comp in model.components:
        if isinstance(comp, IonChannelModel):
            value = numpy.log(float(comp.parameters['g']))
            # Initialise parameters with infinite bounds to begin
            # with their actual bounds will vary with each
            # iteration
            active_parameters.append(Parameter('{}.gbar'
                                        .format(comp.name),
                                        str(comp.parameters['g'].units[4:]),
                                        float('-inf'),
                                        float('inf'),
                                        log_scale=True,
                                        initial_value=value))
            new_states.append(value)
#     for comp in model.biophysics.components.itervalues():
#         if comp.type == 'ionic-current':
#             for mapping in model.mappings:
#                 if comp.name in mapping.components:
#                     for group in mapping.segments:
#                         value = numpy.log(float(comp.parameters['g'].value))
#                         # Initialise active_parameters with infinite bounds to begin
#                         # with their actual bounds will vary with each
#                         # iteration
#                         active_parameters.append(Parameter('{}.{}.gbar'
#                                                     .format(group, comp.name),
#                                                     'S/cm^2',
#                                                     float('-inf'),
#                                                     float('inf'),
#                                                     log_scale=True,
#                                                     initial_value=value))
#                         new_states.append(value)
    fitnesses = numpy.zeros(len(active_objective))
    # Initialise the tuner so I can use the 'set' method in the loop. The
    # arguments aren't actually used in their current form but are required by
    # the Tuner constructor
    tuner = Tuner(active_parameters,
                  active_objective,
                  algorithm,
                  NineLineSimulation(reduced_model),
                  verbose=args.verbose)
    # Keep reducing the model until it can't be reduced any further
    try:
        while all(fitnesses < args.fitness_bounds):
            states = new_states
            require_tuning = model.merge_leaves()
            passive_parameters = []
            for comp in require_tuning:
                value = float(comp.value)
                passive_parameters.append(Parameter('{}.Ra'.format(comp.name),
                                                    str(comp.value.units)[4:],
                                                    value - args.ra_range,
                                                    value + args.ra_range,
                                                    log_scale=False,
                                                    initial_value=value))
            simulation = NineLineSimulation(reduced_model)
            tuner.set(passive_parameters, passive_objective, algorithm,
                      simulation, verbose=args.verbose)
            new_states, fitnesses, _ = tuner.tune()
            for param, state in zip(active_parameters, states):
                param.lbound = state - args.parameter_range
                param.ubound = state + args.parameter_range
            tuner.set(active_parameters, active_objective, algorithm, simulation,
                      verbose=args.verbose)
            new_states, fitnesses, _ = tuner.tune()
    except IrreducibleMorphologyException:
        pass
    except EvaluationException as e:
        e.save(os.path.join(os.path.dirname(args.output),
                            'evaluation_exception.pkl'))
        raise
    for state, param in zip(states, active_parameters):
        setattr(model, param.name, state)
    # Write final model to file
    etree.ElementTree(model.to_xml()).write(args.output, encoding="UTF-8",
                                            pretty_print=True,
                                            xml_declaration=True)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
