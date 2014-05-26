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

data_dir = os.path.abspath(os.path.join(__file__, 'test_objective_data'))
nineml_file = os.path.join(data_dir, 'Golgi_Solinas08.9ml')

parameter = Parameter('soma.KA.gbar', 0.001, 0.008, False)
parameter_range = numpy.linspace(parameter.lbound, parameter.ubound, 10)
simulation = NineLineSimulation(nineml_file)
# Create a dummy tuner to generate the simulation 'setups'
tuner = Tuner([parameter],
              SpikeFrequencyObjective(1, time_start=time_start,
                                            time_stop=time_stop),
              GridAlgorithm(num_steps=[10]),
              simulation)

cache_dir = os.path.join(data_dir, 'cached')
reference_path = os.path.join(cache_dir, 'reference.neo.pkl')
recordings_path = os.path.join(cache_dir, 'recordings.neo.pkl')
try:
    reference = (neo.PickleIO(reference_path).read()[0].segments[0]
                 .analogsignals[0])
    recordings = neo.PickleIO(recordings_path).read()
except:
    try:
        os.removedirs(cache_dir)
    except:
        pass
    print ("Generating test recordings, this may take some time (but will be "
           " cached for future reference)...")
    os.makedirs(cache_dir)
    cell = NineCellMetaClass(nineml_file)()
    cell.record('v')
    simulation_controller.run(simulation_time=time_stop, timestep=0.025)
    reference = cell.get_recording('v', in_block=True)
    neo.PickleIO(reference_path).write(reference)
    recordings = []
    for candidate in parameter_range:
        recordings.append(simulation.run_all([candidate]))
    neo.PickleIO(recordings_path).write(recordings)
    print "Finished regenerating test recordings"

reference = AnalysedSignal(reference).slice(time_start, time_stop)
analyses = [Analysis(r, simulation.setup) for r in recordings]


class TestObjective(object):

    # Declare this class abstract to avoid accidental construction
    __metaclass__ = ABCMeta

    @unittest.skip
    def _get_fitnesses(self):
        try:
            return self._fitnesses
        except AttributeError:
            self._fitnesses = [self.objective.fitness(a) for a in analyses]
            return self._fitnesses

    @unittest.skip
    def plot(self):
        if not plt:
            raise Exception("Matplotlib not imported properly")
        plt.plot(parameter_range, self._get_fitnesses())
        plt.xlabel('soma.KA.gbar')
        plt.ylable('fitness')
        plt.show()

    def test_fitness(self):
        self.assertEqual(self, self.get_fitnesses(), self.target_fitnesses)


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


if __name__ == '__main__':

    test = TestPhasePlaneHistObjective()
    print test.test_fitness()
    test.plot()
