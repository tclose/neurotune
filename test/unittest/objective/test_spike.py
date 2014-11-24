# -*- coding: utf-8 -*-
"""
Tests of the objective package
"""

# needed for python 3 compatibility
from __future__ import division
import os.path
import neo
import sys
import quantities as pq

# Sometimes it is convenient to run it outside of the unit-testing framework
# in which case the ng module is not imported
if __name__ == '__main__':
    from neurotune.utilities import DummyTestCase as TestCase  # @UnusedImport
else:
    try:
        from unittest2 import TestCase
    except ImportError:
        from unittest import TestCase
from neurotune.objective.spike import (SpikeFrequencyObjective,
                                       SpikeTimesObjective,
                                       MinCurrentToSpikeObjective,
                                       SpikeAmplitudeObjective)
from neurotune.analysis import AnalysedRecordings
try:
    from matplotlib import pyplot as plt
except:
    plt = None


# Load testing traces into analysed recordings
data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                        'traces')
soma_seg = neo.PickleIO(os.path.join(data_dir, 'purkinje_soma.pkl')).read()
dend_seg = neo.PickleIO(os.path.join(data_dir, 'purkinje_dendrite.pkl')).read()
soma_analysis = AnalysedRecordings(soma_seg)
dend_analysis = AnalysedRecordings(dend_seg)
reference = soma_analysis.get_analysed_signal()


class TestObjectiveBase(TestCase):

    def test_fitness(self):
        fitness = self.objective.fitness(soma_analysis)
        print ("Fitness value: {} (reference amplitude {})"
               .format(fitness, self.amplitude))
        #self.assertEqual(fitness, self.target_fitness)


class TestSpikeFrequencyObjective(TestObjectiveBase):

    target_fitnesses = 0.0

    def setUp(self):
        self.objective = SpikeFrequencyObjective(reference.spike_frequency(),
                                                 time_start=reference.t_start,
                                                 time_stop=reference.t_stop)


class TestSpikeTimesObjective(TestObjectiveBase):

    target_fitness = 0.0

    def setUp(self):
        self.objective = SpikeTimesObjective(reference.spikes(),
                                             time_start=reference.t_start,
                                             time_stop=reference.t_stop)


class TestMinCurrentToSpikeObjective(TestObjectiveBase):

    target_fitness = 0.0

    def setUp(self):
        self.objective = MinCurrentToSpikeObjective(time_start=reference.t_start,
                                                    time_stop=reference.t_stop)


class TestSpikeAmplitudeObjective(TestObjectiveBase):

    target_fitness = 1.0

    def __init__(self, amplitude=10 * pq.mV):
        super(TestSpikeAmplitudeObjective, self).__init__()
        self.amplitude = amplitude

    def setUp(self):
        self.objective = SpikeAmplitudeObjective(self.amplitude,
                                                 time_start=reference.t_start,
                                                 time_stop=reference.t_stop)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--amplitude', type=float, default=10.0 * pq.mV,
                        help="The reference spike amplitude")
    args = parser.parse_args()
    test = TestSpikeAmplitudeObjective(amplitude=args.amplitude)
    test.setUp()
    test.test_fitness()
