# -*- coding: utf-8 -*-
"""
Tests of the objective module
"""

# needed for python 3 compatibility
from __future__ import division

# Sometimes it is convenient to run it outside of the unit-testing framework
# in which case the unittesting module is not imported
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
from pype9.cells.neuron import NineCellMetaClass, simulation_controller
from neurotune import Parameter, Tuner
from neurotune.objective.phase_plane import (PhasePlaneHistObjective,
                                             PhasePlanePointwiseObjective)
from neurotune.objective.spike import (SpikeFrequencyObjective,
                                       SpikeTimesObjective)
from neurotune.objective.multi import MultiObjective
from neurotune.algorithm.grid import GridAlgorithm
from neurotune.simulation.pype9 import NineLineSimulation
from neurotune.analysis import AnalysedSignal, AnalysedRecordings
try:
    from matplotlib import pyplot as plt
except:
    plt = None
import pickle


time_start = 250 * pq.ms
time_stop = 2000 * pq.ms

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..', 'data', 'objective'))
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

objective = MultiObjective(PhasePlaneHistObjective(reference),
                           PhasePlanePointwiseObjective(reference, 100,
                                                        (20, -20)),
                           SpikeFrequencyObjective(reference.\
                                                   spike_frequency()),
                           SpikeTimesObjective(reference.spikes()))
simulation = NineLineSimulation(nineml_file)

parameters = [Parameter('soma.KA.gbar', 'nS', 0.001, 0.015, False),
               Parameter('soma.SK2.gbar', 'nS', 0.001, 0.015, False)]


class TestTunerFunctions(TestCase):

    def test_pickle(self):
        tuner1 = Tuner(parameters,
                       objective,
                       GridAlgorithm([10, 10]),
                       simulation)
        with open('./pickle', 'wb') as f:
            pickle.dump(tuner1, f)
        with open('./pickle', 'rb') as f:
            try:
                tuner2 = pickle.load(f)
            except ValueError:
                tuner2 = None
        os.remove('./pickle')
        self.assertEqual(tuner1, tuner2)

if __name__ == '__main__':
    test = TestTunerFunctions()
    test.test_pickle()
