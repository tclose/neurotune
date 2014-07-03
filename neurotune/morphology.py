from operator import itemgetter

def reduce_morphology(morph, num_levels=1, only_most_distal=False,
                      respect_classes=True):
    """
    Reduces a 9ml morphology, starting at the most distal branches and merging
    them with their siblings.
    """
    # Get the segments to reduce
    if only_most_distal:
        max_depth = max([seg.depth for seg in morph.segments])
        to_reduce = [seg for seg in morph.segments if seg.depth == max_depth]
    else:
        to_reduce = [seg for seg in morph.segments if not seg.children]
    # Filter out segments that are in different classes to their parents if
    # respect_classes flag is set
    if respect_classes:
        to_reduce = [seg for seg in to_reduce
                     if sorted(seg.classes) == sorted(seg.parent.classes)]
    