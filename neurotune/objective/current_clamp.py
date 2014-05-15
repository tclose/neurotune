from __future__ import absolute_import
from abc import ABCMeta  # Metaclass for abstract base classes
from collections import namedtuple
import numpy
import scipy.signal
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize
import neo.io
from .__init__ import Objective
from ..simulation.__init__ import RecordingRequest


class CurrentClampObjective(Objective):
    pass
