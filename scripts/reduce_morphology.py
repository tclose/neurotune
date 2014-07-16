#!/usr/bin/env python
"""
Reduces a model in 9ml format, by iteratively merging distal tree branches
and retuning the parameters
"""
import sys
import os.path
import argparse
import numpy
from collections import namedtuple
from itertools import chain, groupby
from copy import deepcopy
from lxml import etree
from nineml.extensions.biophysical_cells import parse as parse_nineml
from nineline.cells.build import BUILD_MODE_OPTIONS
from nineline.arguments import outputpath
from nineline.cells import (Model, SegmentModel, IonChannelModel,
                            AxialResistanceModel,
                            IrreducibleMorphologyException)
from neurotune import Parameter
from neurotune.tuner import EvaluationException
from neurotune.objective.multi import MultiObjective
from neurotune.objective.passive import (RCCurveObjective,
                                         SteadyStateVoltagesObjective)
from neurotune.simulation.nineline import NineLineSimulation
try:
    from neurotune.tuner.mpi import MPITuner as Tuner
except ImportError:
    from neurotune.tuner import Tuner
from neurotune.algorithm import algorithm_factory
sys.path.insert(0, os.path.dirname(__file__))
from tune_9ml import add_tune_arguments, get_objective
sys.path.pop(0)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('nineml', type=str,
                    help="The path of the 9ml cell to tune")
parser.add_argument('leak_components', nargs='+', type=str,
                    help="The names of the component classes of the leak "
                         "components used in the model (eg. 'Lkg')")
parser.add_argument('--build', type=str, default='lazy',
                    help="Option to build the NMODL files before "
                         "running (can be one of {})"
                         .format(BUILD_MODE_OPTIONS))
parser.add_argument('--time', type=float, default=2000.0,
                   help="Recording time")
parser.add_argument('--output', type=outputpath,
                    default=os.path.join(os.environ['HOME'], 'grid.pkl'),
                    help="The path to the output file where the grid will "
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
parser.add_argument('--ra_range', type=float, default=0.5,
                    help="Fraction of combined axial resistance of merged "
                         "branches over which the optimimum axial resistance "
                         "will be searched for")
# Add tuning arguments
add_tune_arguments(parser)


def merge_leaves(tree, only_most_distal=False, ancestry=None):
    """
    Reduces a 9ml morphology, starting at the most distal branches and
    merging them with their siblings.
    """
    # Helper function with which to compare segments on all of their components
    # except axial resistance
    def get_non_Ra_comps(seg):
        return set(c for c in seg.components
                     if not isinstance(c, AxialResistanceModel))
    tree = deepcopy(tree)
    if only_most_distal:
        # Get the branches at the maximum depth
        max_branch_depth = max(seg.branch_depth for seg in tree.segments)
        candidates = [branch for branch in tree.branches
                      if branch[0].branch_depth == max_branch_depth]
    else:
        candidates = [branch for branch in tree.branches
                      if not branch[-1].children]
    # Only include branches that have consistent segment_classes
    candidates = [branch for branch in candidates
                  if all(get_non_Ra_comps(b) ==
                         get_non_Ra_comps(branch[0]) for b in branch)]
    if not candidates:
        raise IrreducibleMorphologyException("Cannot reduce the morphology "
                                             "further without merging "
                                             "segment_classes")
    needs_tuning = []
    # Group together candidates that are "siblings", i.e. have the same
    # parent and also the same components (excl. Ra)
    sibling_seg_classes = groupby(candidates,
                                  key=lambda b: (b[0].parent,
                                                 get_non_Ra_comps(b[0])))
    for (parent, non_Ra_components), siblings_iter in sibling_seg_classes:
        siblings = list(siblings_iter)
        if len(siblings) > 1:
            # Get the combined properties of the segments to be merged
            average_length = (numpy.sum(seg.length
                                        for seg in chain(*siblings)) /
                              len(siblings))
            total_surface_area = numpy.sum(seg.length * seg.diameter
                                           for seg in chain(*siblings))
            # Calculate the (in-parallel) axial resistance of the branches
            # to be merged as a starting point for the subsequent tuning
            # step (see the returned 'needs_tuning' list)
            axial_cond = 0.0
            for branch in siblings:
                axial_cond += 1.0 / numpy.array([seg.Ra
                                                 for seg in branch]).sum()
            axial_resistance = (1.0 / axial_cond) * branch[0].Ra.units
            # Get the diameter of the merged segment so as to conserve
            # total membrane surface area given that the length of the
            # segment is the average of the candidates to be merged.
            diameter = total_surface_area / average_length
            # Get a unique name for the generated segments
            # FIXME: this is not guaranteed to be unique (but should be in
            # most cases given a sane naming convention)
            sorted_names = sorted([s[0].name for s in siblings])
            name = sorted_names[0]
            if len(branch) > 1:
                name += '_' + sorted_names[-1]
            # Extend the new get_segment in the same direction as the
            # parent get_segment
            #
            # If the classes are the same between parent and the new
            # segment treat them as one
            disp = parent.disp * (average_length / parent.length)
            segment = SegmentModel(name, parent.distal + disp, diameter)
            # Add dynamic components to segment
            for comp in non_Ra_components:
                segment.set_component(comp)
            # Create new Ra comp to hold the adjusted axial resistance
            Ra_comp = AxialResistanceModel(name + '_Ra', axial_resistance)
            # FIXME: Remove adding of components to trees. Components should
            #        be able to be reused across multiple trees and
            #        therefore only stored at segment levels
            tree.add_component(Ra_comp)
            segment.set_component(Ra_comp)
            # Add new segment to tree
            tree.add_node_with_parent(segment, parent)
            # Remove old branches from list
            for branch in siblings:
                parent.remove_child(branch[0])
            if ancestry:
                ancestry.record_merger(segment, siblings)
            # Append the Ra component to the 'needs_tuning' list for subsequent
            # retuning along with name of the new segment and the names of the
            # middle segments of the old branches that have been merged
            biggest_branch = siblings[numpy.argmax(numpy.sum(s.surface_area
                                                            for seg in branch),
                                                   for branch in siblings)]
            middle_segment = biggest_branch[len(biggest_branch) // 2]
            Ra_to_tune.append(Ra_comp)
            needs_tuning.append((Ra_comp, middle_segment, segment.name))
    ancestry, Ra_to_tune = tree.normalise_spatial_sampling(ancestry,
                                                           Ra_to_tune)
    return tree, ancestry, Ra_to_tune


def run(args):
    # Get original model to be reduced
    model = Model.from_9ml(next(parse_nineml(args.nineml).itervalues()))
    # Create a copy to hold the reduced model
    reduced_model = deepcopy(model)
    # Get copy of model with active components removed for tuning axial
    # resistances
    passive_model = model.passive_model(leak_components=['Lkg'])  # FIXME: This name shouldn't be hardcoded @IgnorePep8
    # Get the objective functions based on the provided arguments and the
    # original model
    active_objective = get_objective(args, reference=args.nineml)
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
#                         # Initialise active_parameters with infinite bounds to begin @IgnorePep8
#                         # with their actual bounds will vary with each
#                         # iteration
#                         active_parameters.append(Parameter('{}.{}.gbar'
#                                                     .format(group, comp.name), @IgnorePep8
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
    input_synapses = []
    # Keep reducing the model until it can't be reduced any further
    new_model = model  # Start with the full model
    try:
        while all(fitnesses < args.fitness_bounds):
            # If the best fitnesses are below the fitness bounds update the
            # model
            states = new_states
            reduced_model = new_model
            # Merge the leaves of the outmost branches of the tree
            (new_model, requires_tuning,
                        mapped_synapses) = merge_leaves(reduced_model,
                                                        input_synapses)
            # Create the parameters that are required to be retuned for the
            # correct axial resistance for the merged branches
            ra_parameters = []
            reference_inject_locations = []
            tune_inject_locations = []
            for comp, old_segname, new_segname in requires_tuning:
                value = float(comp.value)
                ra_parameters.append(Parameter(comp.name,
                                               str(comp.value.units)[4:],
                                               value * (1.0 - args.ra_range),
                                               value + (1.0 + args.ra_range),
                                               log_scale=False,
                                               initial_value=value))
                reference_inject_locations.append(old_segname)
                tune_inject_locations.append(new_segname)
            # Set up the objectives for the passive tuning of axial resistances
            new_passive_model = new_model.passive_model(['Lkg'])
            rc_obj = RCCurveObjective(NineLineSimulation(passive_model))
            ss_obj = SteadyStateVoltagesObjective(
                                             NineLineSimulation(passive_model))
            passive_objective = MultiObjective([rc_obj, ss_obj])
            # Run the tuner to tune the axial resistances of the merged tree
            tuner.set(ra_parameters, passive_objective, algorithm,
                      NineLineSimulation(new_model.passive_model()),
                      verbose=args.verbose)
            tuned_Ras, passive_fitnesses, _ = tuner.tune()
            # Check to see if the passive tuning was successful
            if not all(passive_fitnesses < args.passive_tolerance):
                raise Exception("Could not find passive tuning for reduced "
                                "model that is less than specified tolerance")
            # Set the axial resistances to the tuned values for the new model
            for (comp, _, _), tuned_Ra in zip(requires_tuning, tuned_Ras):
                comp.value = tuned_Ra
            # Set the bounds on the active parameters for the active tuning
            for param, state in zip(active_parameters, states):
                param.lbound = state - args.parameter_range
                param.ubound = state + args.parameter_range
            # Create new simulation object for latest model
            active_simulation = NineLineSimulation(new_model)
            tuner.set(active_parameters, active_objective, algorithm,
                      active_simulation, verbose=args.verbose)
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
