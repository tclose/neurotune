#!/usr/bin/env python
"""
Loads a saved evaluation exception and replays the error
"""

import argparse
import cPickle as pkl
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('exception_file',
                    help="The path of the saved evaluation exception file")
args = parser.parse_args()

with open(args.exception_file, 'rb') as f:
    objective, candidate, analysis = pkl.load(f)

print "Replaying objective fitness calculation for {}".format(candidate)

if analysis is None:
    print ("Simulation failed for candidate ({}), cannot automatically replay"
           .format(candidate))

print objective.fitness(analysis)
