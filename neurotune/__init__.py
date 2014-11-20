from __future__ import absolute_import
import quantities as pq
from .tuner import Tuner


class Parameter(object):

    def __init__(self, name, units, lbound, ubound, log_scale=False,
                 initial_value=None):
        """
        `name`            -- used to look up the variable in the model
        `units`           -- units of the parameter, also used to set the model
                             [pq.Quantity]
        `lbound`          -- the lower bound placed on the parameter
        `ubound`          -- the upper bound placed on the parameter
        `log_scale`       -- whether the value is log_scaled or not (is
                             converted to non-log scale when setting the value
                             of the model)
        `initial_value`   -- optionally the initial value can be provided as
                             well to be used as a reference
        """
        self.name = name
        self.units = pq.Quantity(1.0, units) if units is not None else 1.0
        self.lbound = float(lbound)
        self.ubound = float(ubound)
        self.log_scale = bool(int(log_scale))
        self.initial_value = initial_value
