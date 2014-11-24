# -*- coding: utf-8 -*-
"""
Tests of the objective package
"""

# needed for python 3 compatibility
from __future__ import division
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
import os.path
import numpy
import shutil
import quantities as pq
import neo
from nineline.cells.neuron import NineCellMetaClass, simulation_controller
from neurotune import Parameter, Tuner
from neurotune.objective.phase_plane import (PhasePlaneHistObjective,
                                             PhasePlanePointwiseObjective)
from neurotune.objective.spike import (SpikeFrequencyObjective,
                                       SpikeTimesObjective,
                                       MinCurrentToSpikeObjective)
from neurotune.algorithm.grid import GridAlgorithm
from neurotune.simulation.nineline import NineLineSimulation
from neurotune.analysis import AnalysedSignal, AnalysedRecordings
try:
    from matplotlib import pyplot as plt
except:
    plt = None

time_start = 250 * pq.ms
time_stop = 2000 * pq.ms

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..', '..', 'data', 'objective'))
nineml_file = os.path.join(os.path.join(os.path.dirname(__file__),
                                        '..', '..', 'data', '9ml',
                                        'Golgi_Solinas08.9ml'))
#                                         'Granule_DeSouza10.9ml'))

parameter = Parameter('soma.KA.gbar', 'nS', 0.001, 0.015, False)
parameter_range = numpy.linspace(parameter.lbound, parameter.ubound, 15)
simulation = NineLineSimulation(nineml_file)
# Create a dummy tuner to generate the simulation 'setups'
tuner = Tuner([parameter], SpikeFrequencyObjective(1, time_start=time_start,
                                                   time_stop=time_stop),
              GridAlgorithm(num_steps=[10]),
              simulation)

cache_dir = os.path.join(data_dir, 'cached')
reference_path = os.path.join(cache_dir, 'reference.neo.pkl')
try:
    reference_block = neo.PickleIO(reference_path).read()[0]
    recordings = []
    for p in parameter_range:
        recording = neo.PickleIO(os.path.join(cache_dir,
                                              '{}.neo.pkl'.format(p))).read()[0]
        recordings.append(recording)
except:
    try:
        shutil.rmtree(cache_dir)
    except:
        pass
    print ("Generating test recordings, this may take some time (but will be "
           "cached for future reference)...")
    os.makedirs(cache_dir)
    cell = NineCellMetaClass(nineml_file)()
    cell.record('v')
    print "Simulating reference trace"
    simulation_controller.run(simulation_time=time_stop, timestep=0.025)
    reference_block = cell.get_recording('v', in_block=True)
    neo.PickleIO(reference_path).write(reference_block)
    recordings = []
    for p in parameter_range:
        print "Simulating candidate parameter {}".format(p)
        recording = simulation.run_all([p])
        neo.PickleIO(os.path.join(cache_dir,
                                  '{}.neo.pkl'.format(p))).write(recording)
        recordings.append(recording)
    print "Finished regenerating test recordings"

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


class TestPhasePlaneHistObjective(TestObjective, TestCase):

    target_fitnesses = [0.02015844682551193, 0.018123409598981708,
                        0.013962311575888967, 0.0069441036552407784,
                        0.0023839335328775684, 0.0011445239578201732,
                        0.00030602120322790186, 3.0887216189148659e-06,
                        0.006547149370465518, 0.0076745489061881287,
                        0.0099491858088049737, 0.013118872960859328,
                        0.033487424019271739, 0.036565367845945843,
                        0.039124238259558256]

    def setUp(self):
        self.objective = PhasePlaneHistObjective(reference,
                                                 time_start=time_start,
                                                 time_stop=time_stop)


class TestPhasePlanePointwiseObjective(TestObjective, TestCase):

    target_fitnesses = [791688.05737917486, 417417.7464231535,
                        193261.77390985735, 74410.720655699188,
                        22002.548124354013, 3708.9743763776464,
                        181.9806266596876, 1.6331237044696623e-33,
                        178.61862070496014, 3635.6582762656681,
                        20987.897851732858, 71968.838988315663,
                        187877.22081095798, 403248.13347720244,
                        761436.21907631645]

    def setUp(self):
        self.objective = PhasePlanePointwiseObjective(reference,
                                                      (20, -20), 100,
                                                      time_start=time_start,
                                                      time_stop=time_stop)


if __name__ == '__main__':
    print "didn't run any tests"
