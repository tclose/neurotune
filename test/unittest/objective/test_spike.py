# -*- coding: utf-8 -*-
"""
Tests of the objective package
"""

# needed for python 3 compatibility
from __future__ import division
import os.path
import neo
import numpy
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
avg_reference_amp = numpy.average(reference.spike_amplitudes())
avg_reference_freq = numpy.average(reference.spike_frequency())


class TestObjectiveBase(TestCase):

    def test_fitness(self):
        fitnesses = []
        for objective in self.objectives:
            fitnesses.append(objective.fitness(soma_analysis))
        # self.assertEqual(fitnesses, self.target_fitness)
        return fitnesses


class TestSpikeFrequencyObjective(TestObjectiveBase):

    target_fitnesses = 0.0
    
    references = numpy.arange(20, 70, 5) * pq.Hz

    def setUp(self):
        self.objectives = [SpikeFrequencyObjective(frequency,
                                                 time_start=reference.t_start,
                                                 time_stop=reference.t_stop)
                          for frequency in self.references]


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

    references = numpy.arange(-20, 30, 5) * pq.mV

    def setUp(self):
        self.objectives = [SpikeAmplitudeObjective(amplitude,
                                                   time_start=reference.t_start,
                                                   time_stop=reference.t_stop)
                           for amplitude in self.references]
"""
if __name__ == '__main__':
    test = TestSpikeAmplitudeObjective()
    test.setUp()
    fitnesses = test.test_fitness()
    plt.plot(test.references, fitnesses)
    plt.xlabel('Target amplitude (mV)')
    plt.ylabel('Fitness')
    plt.title("Objective function (avg. amp.={})"
              .format(avg_reference_amp))
    plt.show()
"""

if __name__ == '__main__':
    test = TestSpikeFrequencyObjective()
    test.setUp()
    fitnesses = test.test_fitness()
    plt.plot(test.references, fitnesses)
    plt.xlabel('Target frequecy (Hz)')
    plt.ylabel('Fitness')
    plt.title("Objective function (avg. freq.={})"
              .format(avg_reference_freq))
    plt.show()