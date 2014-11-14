# -*- coding: utf-8 -*-
"""
Tests of the analysis module
"""

# needed for python 3 compatibility
from __future__ import division

import os
import pickle
import numpy

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import quantities as pq
import neo
from neo.core import AnalogSignal
from neurotune.analysis import AnalysedSignal, AnalysedSignalSlice


class TestAnalysedSignalFunctions(unittest.TestCase):

    test_data_file = os.path.join(os.path.dirname(__file__), '..', 'data',
                                  'signals', 'current_clamp.neo.pkl')

    def test_pickle(self):
        signal = AnalogSignal(range(20), sampling_period=1 * pq.ms,
                              units=pq.mV)
        analysed_signal1 = AnalysedSignal(signal)
        with open('./pickle', 'wb') as f:
            pickle.dump(analysed_signal1, f)
        with open('./pickle', 'rb') as f:
            try:
                analysed_signal2 = pickle.load(f)
            except ValueError:
                analysed_signal2 = None
        os.remove('./pickle')
        self.assertEqual(analysed_signal1, analysed_signal2)

    def test_spike_amplitudes(self):
        ref_amps = [7.65625, 7.56875, 7.25, 7.48125, 7.753125, 7.425, 7.715625,
                    7.746875, 7.675, 7.665625, 7.88125, 7.859374, 7.5625,
                    7.859374, 7.5625, 7.85, 7.834375, 7.70625, 7.56875, 7.525,
                    7.659375, 7.178125, 7.3, 7.4125, 7.584375, 7.721875,
                    7.621875, 7.33125, 7.31875, 7.53125, 7.475, 7.359375,
                    7.515625, 7.35, 7.203125, 7.459375, 7.28125, 7.3, 7.321875,
                    7.48125, 7.828125, 7.259375, 7.165625, 7.6375, 7.078125,
                    7.575]
        ref_amps = numpy.array(ref_amps)
        block = neo.PickleIO(self.test_data_file).read()[0]
        inject0 = next(s for s in block.segments
                       if s.name == '0.0 nA injection')
        sig = inject0.analogsignals[0]
        analysed_sig = AnalysedSignal(sig)
        amps = analysed_sig.spike_amplitudes()
        total_diff = numpy.sum(ref_amps - amps)
        self.assertAlmostEquals(total_diff, 0.0, 12)


class TestAnalysedSignalSliceFunctions(unittest.TestCase):

    def test_pickle(self):
        signal = AnalogSignal(range(20), sampling_period=1 * pq.ms,
                              units=pq.mV)
        analysed_signal = AnalysedSignal(signal)
        sliced_signal1 = AnalysedSignalSlice(analysed_signal,
                                             t_start=5 * pq.ms,
                                             t_stop=15 * pq.ms)
        with open('./pickle', 'wb') as f:
            pickle.dump(sliced_signal1, f)
        with open('./pickle', 'rb') as f:
            try:
                sliced_signal2 = pickle.load(f)
            except ValueError:
                sliced_signal2 = None
        os.remove('./pickle')
        self.assertEqual(sliced_signal1, sliced_signal2)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    block = neo.PickleIO(TestAnalysedSignalFunctions.test_data_file).read()[0]
    inject0 = next(s for s in block.segments
                   if s.name == '0.0 nA injection')
    sig = inject0.analogsignals[0]
    analysed_sig = AnalysedSignal(sig)
    periods = analysed_sig._spike_period_indices()
    amps = analysed_sig.spike_amplitudes()
    print amps
    plt.plot(sig.times, sig)
    plt.show()