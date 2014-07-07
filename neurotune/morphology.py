from itertools import groupby, chain
from copy import deepcopy
import numpy
from lxml import etree
import quantities as pq
import nineml.extensions.biophysical_cells
from nineml.extensions.morphology import Segment, DistalPoint, ParentReference


class IrreducibleException(Exception):
    pass


def reduce_morphology(morph, only_most_distal=False):
    """
    Reduces a 9ml morphology, starting at the most distal branches and merging
    them with their siblings.
    """
    # Create a complete copy of the morphology to allow it to be reduced
    morph = deepcopy(morph)
    if only_most_distal:
        # Get the branches
        max_branch_depth = max(seg.branch_depth for seg in morph.segments)
        candidates = [branch for branch in morph.branches
                      if branch[0].branch_depth == max_branch_depth]
    else:
        candidates = [branch for branch in morph.branches
                      if not branch[-1].children]
    # Only include branches that have consistent classes
    candidates = [branch for branch in candidates
                  if all(b.classes == branch[0].classes for b in branch)]
    if not candidates:
        raise IrreducibleException("Cannot reduce the morphology further{}."
                                   "without merging classes")
    sibling_groups = groupby(candidates,
                             key=lambda b: (b[0].parent, b[0].classes))
    for (parent, classes), siblings_iter in sibling_groups:
        siblings = list(siblings_iter)
        if len(siblings) > 1:
            average_length = (numpy.sum(seg.length
                                        for seg in chain(*siblings)) /
                              len(siblings))
            total_surface_area = numpy.sum(seg.length * seg.diameter()
                                           for seg in chain(*siblings))
            diameter = total_surface_area / average_length
            name = siblings[0].name
            if len(branch) > 1:
                name += '_' + siblings[-1].name
            parent_ref = ParentReference(parent.name, 1.0)
            parent_ref.segment = parent
            distal = (parent.distal + parent.disp *
                      (average_length / parent.length))
            segment = Segment(name, distal=DistalPoint(distal[0], distal[1],
                                                       distal[2], diameter),
                              parent_ref=parent_ref)
            segment.classes = classes
            segment.reduced = True
            for branch in siblings:
                parent.children.remove(branch[0])
            parent.children.append(segment)
    return morph


def rationalise_spatial_sampling(model9ml, **d_lambda_kwargs):
    model9ml = deepcopy(model9ml)
    default_params = model.biophysics.components['__NO_COMPONENT__'].parameters
    Ra = pq.Quantity(float(default_params['Ra'].value),
                     default_params['Ra'].unit)
    cm = pq.Quantity(float(default_params['C_m'].value), 'uF/cm^2')  # default_params['C_m'].unit) @IgnorePep8
    to_replace = []
    for branch in model9ml.morphology.branches:
        parent = branch[0].parent
        if parent:
            branch_length = numpy.sum(seg.length for seg in branch) * pq.um
            diameter = (numpy.sum(seg.diameter() for seg in branch) /
                        len(branch) * pq.um)
            num_segments = d_lambda_rule(branch_length, diameter, Ra, cm,
                                         **d_lambda_kwargs)
            base_name = branch[0].name
            if len(branch) > 1:
                base_name += '_' + branch[-1].name
            # Get the direction of the branch
            classes = branch[0].classes
            direction = branch[-1].distal - branch[0].proximal
            direction *= (branch_length /
                          numpy.sqrt(numpy.sum(direction ** 2)))
            # Temporarily add the parent to the new_branch to allow it to be
            # linked to the new segments
            new_branch = [parent]
            for i, seg_length in enumerate(
                                    numpy.linspace(0.0,
                                                   float(branch_length),
                                                   num_segments)):
                name = base_name + '_' + str(i)
                parent_ref = ParentReference(new_branch[-1].name, 1.0)
                parent_ref.segment = new_branch[-1]
                distal = branch[0].proximal + direction * seg_length
                segment = Segment(name, parent_ref=parent_ref,
                                  distal=DistalPoint(distal[0], distal[1],
                                                     distal[2], diameter))
                segment.classes = classes
                new_branch.append(segment)
            new_branch.pop(0)  # remove the parent from the new_branch
            to_replace.append((parent, branch[0], new_branch[0]))
    for parent, orig_branch_start, new_branch_start in to_replace:
        parent.children.remove(orig_branch_start)
        parent.children.append(new_branch_start)
    return model9ml


def d_lambda_rule(length, diam, Ra, cm, freq=(100.0 * pq.Hz), d_lambda=0.3):
    """
    Calculates the number of segments required for a straight branch section so
    that its segments are no longer than d_lambda x the AC length constant at
    frequency freq in that section.

    See Hines, M.L. and Carnevale, N.T.
       NEURON: a tool for neuroscientists.
       The Neuroscientist 7:123-135, 2001.

    `length`     -- length of the branch section
    `diameter`   -- diameter of the branch section
    `Ra`         -- Axial resistance (Ohm cm)
    `cm`         -- membrane capacitance (uF cm^(-2))
    `freq`       -- frequency at which AC length constant will be computed (Hz)
    `d_lambda`   -- fraction of the wavelength
    """
    # Convert all units to the correct values
    length = float(pq.Quantity(length, 'um'))
    diam = float(pq.Quantity(diam, 'um'))
    Ra = float(pq.Quantity(Ra, 'ohm.cm'))
    cm = float(pq.Quantity(cm, 'uF/cm^2'))
    freq = float(pq.Quantity(freq, 'Hz'))
    # Calculate the wavelength for the segment
    lambda_f = 1e5 * numpy.sqrt(diam / (4 * numpy.pi * freq * Ra * cm))
    length_frac = length / (d_lambda * lambda_f)
    return int((length_frac + 0.9) / 2) * 2 + 1
    # above was too inaccurate with large variation in 3d diameter
    # so now we use all 3-d points to get a better approximate lambda
#     x1 = arc3d(0)
#     d1 = diam3d(0)
#     lam = 0
#     for i in xrange(1, n3d() - 1):
#         x2 = arc3d(i)
#         d2 = diam3d(i)
#         lam += (x2 - x1) / numpy.sqrt(d1 + d2)
#         x1 = x2
#         d1 = d2
#     #  length of the section in units of lambda
#     lam *= numpy.sqrt(2) * 1e-5 * numpy.sqrt(4 * numpy.pi * freq * Ra * cm)
#     lambda_f =  L / lam


if __name__ == '__main__':
    nineml_file = '/home/tclose/git/kbrain/9ml/neurons/Golgi_Solinas08.9ml'
    models = nineml.extensions.biophysical_cells.parse(nineml_file)
    model = next(models.itervalues())
    model.morphology = reduce_morphology(model.morphology)
    reduced_model = rationalise_spatial_sampling(model)
    etree.ElementTree(reduced_model.to_xml()).write(
             '/home/tclose/git/kbrain/9ml/neurons/Golgi_Solinas08-reduced.9ml',
             encoding="UTF-8",
             pretty_print=True,
             xml_declaration=True)
