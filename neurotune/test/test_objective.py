# -*- coding: utf-8 -*-
"""
Tests of the objective module
"""

# needed for python 3 compatibility
from __future__ import division
from abc import ABCMeta  # Metaclass for abstract base classes

try:
    import unittest2 as unittest
except ImportError:
    import unittest  # @UnusedImport

import os.path
import shutil
import numpy
import quantities as pq
import neo
from nineline.cells.neuron import NineCellMetaClass, simulation_controller
from neurotune import Parameter, Tuner
from neurotune.objective.phase_plane import (PhasePlaneHistObjective,
                                             PhasePlanePointwiseObjective)
from neurotune.objective.spike import (SpikeFrequencyObjective,
                                       SpikeTimesObjective)
from neurotune.algorithm import GridAlgorithm
from neurotune.simulation.nineline import NineLineSimulation
from neurotune.analysis import AnalysedSignal, Analysis
try:
    from matplotlib import pyplot as plt
except:
    plt = None

time_start = 500 * pq.ms
time_stop = 2000 * pq.ms

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        'test_objective_data'))
nineml_file = os.path.join(data_dir, 'Golgi_Solinas08.9ml')

parameter = Parameter('soma.KA.gbar', 'nS', 0.001, 0.015, False)
parameter_range = numpy.linspace(parameter.lbound, parameter.ubound, 15)
simulation = NineLineSimulation(nineml_file)
# Create a dummy tuner to generate the simulation 'setups'
tuner = Tuner([parameter],
              SpikeFrequencyObjective(1, time_start=time_start,
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
                                              str(p) + '.neo.pkl')).read()[0]
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
    for param in parameter_range:
        print "Simulating candidate parameter {}".format(param)
        recording = simulation.run_all([param])
        neo.PickleIO(os.path.join(cache_dir,
                                  str(param) + '.neo.pkl')).write(recording)
        recordings.append(recording)
    print "Finished regenerating test recordings"

reference = AnalysedSignal(reference_block.segments[0].analogsignals[0]).\
                                                   slice(time_start, time_stop)
analyses = [Analysis(r, simulation.setups) for r in recordings]


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
        plt.show()

    def test_fitness(self):
        fitnesses = [self.objective.fitness(a) for a in analyses]
        self.assertEqual(self, fitnesses, self.target_fitnesses)


class TestPhasePlaneHistObjective(TestObjective, unittest.TestCase):

    target_fitnesses = []

    def setUp(self):
        self.objective = PhasePlaneHistObjective(reference)


class TestPhasePlanePointwiseObjective(TestObjective, unittest.TestCase):

    target_fitnesses = []

    def setUp(self):
        self.objective = PhasePlanePointwiseObjective(reference, (20, -20),
                                                      100)


class TestSpikeFrequencyObjective(TestObjective, unittest.TestCase):

    target_fitnesses = []

    def setUp(self):
        self.objective = SpikeFrequencyObjective(reference.spike_frequency())


class TestSpikeTimesObjective(TestObjective, unittest.TestCase):

    target_fitnesses = []

    def setUp(self):
        self.objective = SpikeTimesObjective(reference.spike_times())
