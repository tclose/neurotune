from __future__ import absolute_import 
from collections import namedtuple
from abc import ABCMeta # Metaclass for abstract base classes
from .simulation import __init__ as simulation
from .algorithm import __init__ as algorithm
from .objective import __init__ as objective


Parameter = namedtuple('Parameter', 'name units lbound ubound')
       

