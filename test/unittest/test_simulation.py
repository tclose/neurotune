# -*- coding: utf-8 -*-
"""
Tests of the analysis module
"""

# needed for python 3 compatibility
from __future__ import division

import os
import pickle
from neurotune.simulation.nineline import NineLineSimulation
if __name__ == '__main__':

    class unittest(object):

        class TestCase(object):

            def __init__(self):
                try:
                    self.setUp()
                except AttributeError:
                    pass

            def assertEqual(self, first, second):
                print 'are{} equal'.format(' not' if first != second else '')
else:
    try:
        import unittest2 as unittest
    except ImportError:
        import unittest


data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..', 'data', 'objective'))
nineml_file = os.path.join(data_dir, 'Golgi_Solinas08.9ml')


class TestNineLineSimulationFunctions(unittest.TestCase):

    def test_pickle(self):
        simulation1 = NineLineSimulation(nineml_file)
        with open('./pickle', 'wb') as f:
            pickle.dump(simulation1, f)
        with open('./pickle', 'rb') as f:
            try:
                simulation2 = pickle.load(f)
            except ValueError:
                simulation2 = None
        os.remove('./pickle')
        self.assertEqual(simulation1, simulation2)

if __name__ == '__main__':
    test = TestNineLineSimulationFunctions()
    test.test_pickle()