from __future__ import absolute_import
import quantities as pq
from .tuner import Tuner


class Parameter(object):

    def __init__(self, name, units, lbound, ubound, log_scale=False):
        self.name = name
        self.units = pq.Quantity(1.0, units)
        self.lbound = float(lbound)
        self.ubound = float(ubound)
        self.log_scale = bool(int(log_scale))
