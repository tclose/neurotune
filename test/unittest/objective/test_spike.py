# -*- coding: utf-8 -*-
"""
Tests of the objective package
"""

# needed for python 3 compatibility
from __future__ import division
import cPickle as pkl
from abc import ABCMeta  # Metaclass for abstract base classes

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
                                       MinCurrentToSpikeObjective)
from neurotune.analysis import AnalysedSignal, AnalysedRecordings
try:
    from matplotlib import pyplot as plt
except:
    plt = None


with open('spiking_neuron_analysis.pkl', 'w') as f:
    reference = pkl.load(f)

reference = AnalysedSignal(reference_block.segments[0].analogsignals[0]).\
                                                   slice(time_start, time_stop)
analyses = [AnalysedRecordings(r, simulation.setups) for r in recordings]
analyses_dict = dict([(str(r.annotations['candidate'][0]),
                       AnalysedRecordings(r, simulation.setups))
                      for r in recordings])


class TestObjective(object):

    # Declare this class abstract to avoid accidental construction
    __metaclass__ = ABCMeta

    def plot(self):
        if not plt:
            raise Exception("Matplotlib not imported properly")
        plt.plot(parameter_range,
                 [self.objective.fitness(a) for a in analyses])
        plt.xlabel('soma.KA.gbar')
        plt.ylabel('fitness')
        plt.title(self.__class__.__name__)
        plt.show()

    def test_fitness(self):
        fitnesses = [self.objective.fitness(a) for a in analyses]
        self.assertEqual(fitnesses, self.target_fitnesses)


class TestSpikeFrequencyObjective(TestObjective, TestCase):

    target_fitnesses = [0.3265306122448987, 0.3265306122448987,
                        0.3265306122448987, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.32653061224489766, 0.32653061224489766,
                        0.32653061224489766, 0.32653061224489766,
                        1.3061224489795906, 1.3061224489795906,
                        1.3061224489795906]

    def setUp(self):
        self.objective = SpikeFrequencyObjective(reference.spike_frequency(),
                                                 time_start=time_start,
                                                 time_stop=time_stop)


class TestSpikeTimesObjective(TestObjective, TestCase):

    target_fitnesses = [48861.63264168518, 42461.31814161993,
                        45899.285983621434, 71791.87749344285,
                        72317.99719666546, 43638.346161592424,
                        11543.74327161325, 2.6999188118427894e-20,
                        24167.5639638691, 51168.20605556744, 68990.99639960933,
                        54978.101362379784, 60117.67140614826,
                        55935.42039310986, 58535.24894951394]

    def setUp(self):
        self.objective = SpikeTimesObjective(reference.spikes(),
                                             time_start=time_start,
                                             time_stop=time_stop)


class TestMinCurrentToSpikeObjective(TestObjective, TestCase):

    target_fitnesses = []

    def setUp(self):
        self.objective = MinCurrentToSpikeObjective(time_start=time_start,
                                                    time_stop=time_stop)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', default="amplitude",
                        help="Which objective to test")
    

    test = TestMinCurrentToSpikeObjective()
    test.setUp()
    test.test_fitness()
