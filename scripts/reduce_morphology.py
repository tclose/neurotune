#!/usr/bin/env python
"""
Reduces a model in 9ml format, by iteratively merging distal tree branches
and retuning the parameters
"""
import sys
import os.path
import numpy
import argparse
from nineml.extensions.biophysical_cells import parse as parse_nineml
from nineline.cells.build import BUILD_MODE_OPTIONS
from nineline.arguments import outputpath
from neurotune import Parameter
from neurotune.simulation.nineline import NineLineSimulation
from neurotune.morphology import (reduce_morphology,
                                  rationalise_spatial_sampling,
                                  merge_morphology_classes,  # @UnusedImport
                                  IrreducibleMorphologyException)
from lxml import etree
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
parser.add_argument('--fitness_bounds', nargs='+', type=float,
                    help="The bounds on the fitnesses below which the tree "
                         "will continue to be reduced")
# Add tuning arguments
add_tune_arguments(parser)


def run(args):
    # Get original model to be reduced
    model = next(parse_nineml(args.nineml).itervalues())
    # Get the objective functions based on the provided arguments and the
    # original model
    objective = get_objective(reference=args.nineml)
    parameters = []
    new_states = []
    for comp in model.biophysics.components.itervalues():
        if comp.type == 'ionic-current':
            value = numpy.log(float(comp.parameters['g'].value))
            # Initialise parameters with infinite bounds to begin with.
            # Their actual bounds will vary with each iteration
            parameters.append(Parameter('soma.{}.gbar'.format(comp.name),
                                        'S/cm^2', float('-inf'), float('inf'),
                                        log_scale=True,
                                        initial_value=value))
            new_states.append(value)
    fitnesses = numpy.zeros(len(objective))
    # Keep reducing the model until it can't be reduced any further
    try:
        while all(fitnesses < args.fitness_bounds):
            states = new_states
            model.morphology = reduce_morphology(model.morphology)
            model = rationalise_spatial_sampling(model)
            simulation = NineLineSimulation(model)
            for param, state in zip(parameters, states):
                param.lbound = state - args.parameter_range
                param.ubound = state + args.parameter_range
            new_states, fitnesses, _ = tune(args, parameters=parameters,
                                            objective=objective,
                                            simulation=simulation)
    except IrreducibleMorphologyException:
        pass
    for state, param in zip(states, parameters):
        setattr(model, param.name, state)
    # Write final model to file
    etree.ElementTree(model.to_xml()).write(args.output, encoding="UTF-8",
                                            pretty_print=True,
                                            xml_declaration=True)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
