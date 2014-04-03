from __future__ import absolute_import
from collections import namedtuple
from .tuner.__init__ import Tuner

Parameter = namedtuple('Parameter', 'name units lbound ubound')
       

