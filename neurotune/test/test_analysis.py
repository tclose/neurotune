# -*- coding: utf-8 -*-
"""
Tests of the neo.core.analogsignal.AnalogSignal class and related functions
"""

# needed for python 3 compatibility
from __future__ import division

import os
import pickle

try:
    import unittest2 as unittest
except ImportError:
    import unittest

import quantities as pq
from neo.core import AnalogSignal
from neurotune.analysis import AnalysedSignal, SlicedAnalysedSignal
from neo.test.tools import assert_arrays_equal


class TestAnalysedSignalFunctions(unittest.TestCase):

    def test_pickle(self):
        signal = AnalogSignal([1, 2, 3, 4], sampling_period=1 * pq.ms,
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
        assert_arrays_equal(analysed_signal1, analysed_signal2)


class TestSlicedAnalysedSignalFunctions(unittest.TestCase):

    def test_pickle(self):
        signal = AnalogSignal(range(20), sampling_period=1 * pq.ms,
                               units=pq.mV)
        analysed_signal = AnalysedSignal(signal)
        sliced_signal1 = SlicedAnalysedSignal(analysed_signal,
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
        assert_arrays_equal(sliced_signal1, sliced_signal2)
