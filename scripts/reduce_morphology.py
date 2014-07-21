#!/usr/bin/env python
"""
Reduces a model in 9ml format, by iteratively merging distal tree branches
and retuning the parameters
"""
import sys
import os.path
import argparse
import numpy
from itertools import chain, groupby
from copy import deepcopy
from lxml import etree
from nineml.extensions.biophysical_cells import parse as parse_nineml
import quantities as pq
from nineline.cells.build import BUILD_MODE_OPTIONS
from nineline.arguments import outputpath
from nineline.cells import (Model, SegmentModel, IonChannelModel,
                            BranchAncestry, AxialResistanceModel,
                            IrreducibleMorphologyException,
                            DistributedParameter, DummyNinemlModel,
                            PointProcessModel)
from nineline.cells.neuron import NineCellMetaClass
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

home_dir = os.environ['HOME']

cn_dir = os.path.join(home_dir, 'git', 'cerebellarnuclei')
psection_fn = os.path.join(cn_dir, 'extracted_data', 'psections.txt')
mechs_fn = os.path.join(cn_dir, 'extracted_data', 'mechanisms.txt')

alpha = 0.2


def load_dcn_model():
    model = Model.from_psections(psection_fn, mechs_fn)
    for comp in chain(model.components_of_class('CaConc'),
                      model.components_of_class('CalConc')):
        segments = list(model.component_segments(comp))
        if len(segments) == 1:
            assert segments[0].name == 'soma'
            comp.parameters['depth'] = DistributedParameter(
                        lambda seg: alpha - 2 * alpha ** 2 / seg.diam + \
                                    4 * alpha ** 3 / (3 * seg.diam ** 2))
        else:
            comp.parameters['depth'] = DistributedParameter(
                                lambda seg: alpha - alpha ** 2 / seg.diam)
    nineml_model = DummyNinemlModel('CerebellarNuclei', cn_dir, model)
    model._source = nineml_model
    celltype = NineCellMetaClass(nineml_model, standalone=True)
    return celltype, model


def plot_branches(tree, branches):
    from btmorph.btviz import plot_3D_SWC
    import matplotlib.pyplot as plt
    for b in branches:
        for seg in b:
            seg.diameter = 10
    tree.write_SWC_tree_to_file('/home/tclose/Desktop/'
                                 'reduced.swc')
    plot_3D_SWC('/home/tclose/Desktop/reduced.swc')
    plt.show()


def merge_leaves(tree, only_most_distal=False, ancestry=None,
                 num_merges=1, normalise=True):
    """
    Reduces a 9ml morphology, starting at the most distal branches and
    merging them with their siblings.
    """
    # Helper function with which to compare segments on all of their components
    # except axial resistance
    def get_non_Ra_comps(seg):
        return set(c for c in seg.components
                     if not (isinstance(c, AxialResistanceModel) or
                             isinstance(c, PointProcessModel)))
    tree = deepcopy(tree)
    for _ in xrange(num_merges):
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
            raise IrreducibleMorphologyException("Cannot reduce the morphology"
                                                 " further without merging "
                                                 "segment_classes")
        # Keep track of the newly created axial resistance components that will
        # need to be tuned so that the passive properties of the cell are
        # consistent
        Ra_to_tune = []
        # Group together candidates that are "siblings", i.e. have the same
        # parent and also the same components (excl. Ra)
        merged_branches = []
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
                total_surface_area = numpy.sum(seg.length * float(seg.diameter)
                                               for seg in chain(*siblings))
                # Calculate the (in-parallel) axial resistance of the branches
                # to be merged as a starting point for the subsequent tuning
                # step (see the returned 'needs_tuning' list)
                axial_cond = 0.0
                for branch in siblings:
                    axial_cond += 1.0 / numpy.array([seg.Ra
                                                     for seg in branch]).sum()
                axial_resistance = (1.0 / axial_cond) # branch[0].Ra.units FIXME: shouldn't be hard-coded
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
                Ra_to_tune.append(Ra_comp)
                merged_branches.append([segment])
            plot_branches(tree, merged_branches)
        if normalise:
            ancestry, Ra_to_tune = tree.normalise_spatial_sampling(ancestry,
                                                                   Ra_to_tune)
        return tree, ancestry, Ra_to_tune


def tune_passive_model(tuner, algorithm, reference_sim, celltype, model,
                       Ra_to_tune, ancestry, tuning_range, tolerances,
                       leak_components):
    # Create the parameters that are required to be retuned for the
    # correct axial resistance for the merged branches
    ra_parameters = []
    for Ra_comp in Ra_to_tune:
        value = float(Ra_comp.value)
        ra_parameters.append(Parameter(Ra_comp.name,
                                       str(Ra_comp.value.units)[4:],
                                       value * (1.0 - tuning_range),
                                       value + (1.0 + tuning_range),
                                       log_scale=False,
                                       initial_value=value))
    # Set up the objectives for the passive tuning of axial resistances
    passive_model = model.passive_model(leak_components=['Lkg'])
    objectives = []
    for Ra_comp in Ra_to_tune:
        # Get segments that have this component
        segments = [seg for seg in model.segments if Ra_comp in seg.components]
        # Find the root of the branch (should be a single branch)
        root = [seg for seg in segments if seg.parent not in segments]
        leaf = [seg for seg in segments if seg.is_leaf()]
        assert len(root) == 1, "There should only be one root in Ra segments"
        assert len(leaf) == 1, "There should only be one leaf in Ra segments"
        root = root[0]
        leaf = leaf[0]
        segment_path = reversed(leaf.path_to_ancestor(root))
        # Get the equivalent subtree in the original model
        orig_root = ancestry.get_original(root)
        # Loop through all the paths from the root to the leaves in the
        # original subtree and pick out the one with the largest surface
        # area
        largest_surface_area = -float('inf')
        largest_path_to_leaf = None
        for orig_leaf in [c for c in orig_root.allchildren if c.is_leaf()]:
            path_to_leaf = list(orig_leaf.path_to_ancestor(orig_root))
            surface_area = numpy.sum([seg.surface_area
                                      for seg in path_to_leaf])
            if surface_area > largest_surface_area:
                largest_surface_area = surface_area
                largest_path_to_leaf = reversed(path_to_leaf)
        # Get the middle of the largest path from the root
        path_length = numpy.sum([seg.length for seg in segment_path])
        orig_path_length = numpy.sum([seg.length
                                      for seg in largest_path_to_leaf])
        path_fraction = (path_length / 2.0) / orig_path_length
        if path_fraction > 1.0:
            path_fraction = 1.0
        middle_segment = segment_path[len(segment_path) // 2]
        middle_orig_segment = largest_path_to_leaf[
                                                int(len(largest_path_to_leaf) *
                                                path_fraction)]
        # Create a dummy RC-curve objective for the reference simulation in
        # order to generate the correct reference traces
        ref_rc_obj = RCCurveObjective(reference_sim,
                                      inject_location=middle_orig_segment.name)
        # Append RC-curve objective to the list of multiple objectives
        objectives.append(RCCurveObjective(ref_rc_obj.reference,
                                  inject_location=middle_segment.name))
        # Create a dummy Steady-state objective for the reference simulation in
        # order to generate the correct reference traces
        ref_inject_dists = [numpy.sum(s.length
                                      for s in seg.path_to_ancestor(orig_root))
                            for seg in orig_root.allchildren]
        ref_record_sites = [seg.name for seg in orig_root.allchildren]
        ref_ss_obj = SteadyStateVoltagesObjective(
                                        reference_sim,
                                        ref_record_sites,
                                        ref_inject_dists,
                                        ref_inject_dists,
                                        inject_location=orig_root.name)
        rec_inject_dists = [numpy.sum(s.length
                                      for s in seg.path_to_ancestor(root))
                            for seg in segment_path]
        record_sites = [seg.name for seg in segment_path]
        objectives.append(SteadyStateVoltagesObjective(
                                        ref_ss_obj.reference,
                                        record_sites,
                                        ref_inject_dists,
                                        rec_inject_dists,
                                        inject_location=root.name))
    # Run the tuner to tune the axial resistances of the merged tree
    tuner.set(ra_parameters, MultiObjective(objectives), algorithm,
              NineLineSimulation(celltype, model=passive_model))
    tuned_Ras, passive_fitnesses, _ = tuner.tune()
    # Check to see if the passive tuning was successful
    if not all(passive_fitnesses < tolerances):
        raise Exception("Could not find passive tuning for reduced "
                        "model that is less than specified tolerance")
    # Set the axial resistances to the tuned values for the new model
    for Ra_comp, tuned_Ra in zip(Ra_to_tune, tuned_Ras):
        Ra_comp.value = tuned_Ra


def run(args):
    # Get original model to be reduced
    if args.nineml == 'dcn':
        celltype, model = load_dcn_model()
    else:
        nineml_model = next(parse_nineml(args.nineml).itervalues())
        model = Model.from_9ml(nineml_model)
        celltype = NineCellMetaClass(nineml_model)
    # Create an ancestry object to enable tracing back to the segments in the
    # the original model from segments in the reduced model
    ancestry = BranchAncestry(model)
    # Create a copy to hold the reduced model
    reduced_model = deepcopy(model)
    # Get copy of model with active components removed for tuning axial
    # resistances
    passive_model = model.passive_model(leak_components=args.leak_components)
    passive_sim = NineLineSimulation(celltype, model=passive_model)
    # Get the objective functions based on the provided arguments and the
    # original model
    active_objective = get_objective(args, reference=celltype)
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
    fitnesses = numpy.zeros(len(active_objective))
    # Initialise the tuner so I can use the 'set' method in the loop. The
    # arguments aren't actually used in their current form but are required by
    # the Tuner constructor
    tuner = Tuner(active_parameters,
                  active_objective,
                  algorithm,
                  NineLineSimulation(celltype, model=reduced_model),
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
            new_model, Ra_to_tune, ancestry = merge_leaves(reduced_model,
                                                           input_synapses,
                                                           ancestry)
            # Tune the axial resistances of the new model to match the
            # passive properties of the original model
            new_model = tune_passive_model(tuner, algorithm, passive_sim,
                                           new_model, Ra_to_tune, ancestry,
                                           args.ra_range,
                                           args.passive_tolerances,
                                           args.leak_components)
            # Set the bounds on the active parameters for the active tuning
            for param, state in zip(active_parameters, states):
                param.lbound = state - args.parameter_range
                param.ubound = state + args.parameter_range
            # Create new simulation object for latest model
            active_simulation = NineLineSimulation(celltype, model=new_model)
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


# # Append the Ra component to the 'needs_tuning' list for subsequent
# # retuning along with name of the new segment and the names of the
# # middle segments of the old branches that have been merged
# biggest_branch = siblings[numpy.argmax(numpy.sum(s.surface_area
#                                                 for seg in branch),
#                                        for branch in siblings)]
# middle_segment = biggest_branch[len(biggest_branch) // 2]

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

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
