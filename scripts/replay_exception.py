#!/usr/bin/env python
"""
Loads a saved evaluation exception and replays the error
"""

import argparse
import cPickle as pkl
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('exception_file',
                    help="The path of the saved evaluation exception file"
                         " ('evaluation_exception.pkl')")
args = parser.parse_args()

with open(args.exception_file, 'rb') as f:
    tuner, candidate, analysis = pkl.load(f)

print "Replaying evaluation exception for {}".format(candidate)

if analysis is None:
    tuner.simulation.run_all(candidate)
else:
    print "fitness: {}".format(tuner.objective.fitness(analysis))
