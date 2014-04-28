"""
Run the simulation 
"""
from __future__ import absolute_import
from collections import namedtuple
from itertools import groupby
from abc import ABCMeta # Metaclass for abstract base classes
import neo
import quantities as pq
from neurotune.conditions import ExperimentalConditions


class RecordingRequest(object):
    """"
    RecordingRequests are raised by objective functions and are passed to Simulation objects so they
    can set up the required simulation conditions (eg IClamp, VClamp, spike input) and recorders
    """
    
    def __init__(self, record_time=2000.0, record_variable=None, conditions=None):
        """
        `record_time` -- the length of the recording required by the simulation
        `record_variable` -- the name of the section/synapse/current to record from (simulation specific)
        `conditions`  -- the experimental conditions required (eg. initial voltage, current clamp)
        """
        self.record_time = record_time
        self.record_variable = record_variable
        self.conditions = conditions
        

class Simulation():
    "Base class of Simulation objects"
    
    __metaclass__ = ABCMeta # Declare this class abstract to avoid accidental construction
    
    ## Groups together all the information to interface to and from a requested simulation
    Setup = namedtuple('Setup', "time conditions record_variables request_keys")
    
    supported_clamp_types = []
        
    def _process_requests(self, recording_requests):
        """
        Merge recording requests so that the same recording/simulation doesn't get performed 
        multiple times
        
        `recording_requests`  -- a list of recording requests from the objective functions [.RecordingRequest]
        """
        # Group into requests by common experimental conditions
        try:
            request_items = recording_requests.items()
        except AttributeError:
            request_items = [(None, recording_requests)]
        request_items.sort(key=lambda x: x[1].conditions)
        common_conditions = groupby(request_items, key=lambda x: x[1].conditions)
        # Merge the common requests into simulation setups
        self.simulation_setups = []
        for conditions, requests_iter in common_conditions:
            for c in conditions.clamps:
                if type(c) not in self.supported_clamp_types:
                    raise Exception("Condition of type {} is not supported by this Simulation "
                                    "class ({})".format(type(c), self.__class__))
            requests = [r for r in requests_iter]
            # Get the maxium record time in the group
            record_time = max([r[1].record_time for r in requests])
            # Group the requests by common recording sites
            requests.sort(key=lambda x: x[1].record_variable)
            common_record_variables = groupby(requests, key=lambda x: x[1].record_variable)
            # Get the common recording sites
            record_variables, requests_iters = zip(*[(rv, [r for r in requests]) 
                                                      for rv, requests in common_record_variables])
            # Get the list of request keys for each requested recording
            request_keys = [zip(*com_record)[0] for com_record in requests_iters]
            # Append the simulation request to the 
            self.simulation_setups.append(self.Setup(record_time, conditions, 
                                                     list(record_variables), request_keys))
        # Do initial preparation for simulation (how much preparation can be done depends on whether
        # the same experimental conditions are used throughout the evaluation process.
        self.prepare_simulations()
        
    def _get_requested_recordings(self, candidate):
        """
        Return the recordings in a dictionary to be returned to the objective functions, so each
        objective function can access the recording it requested
        
        `candidate` -- a list of parameters [list(float)]
        """
        recordings = self.run(candidate)
        requests_dict = {}
        for seg, setup in zip(recordings.segments, self.simulation_setups):
            assert len(seg.analogsignals) == len(setup.request_keys)
            for signal, request_keys in zip(seg.analogsignals, setup.request_keys):
                requests_dict.update([(key, signal) for key in request_keys])
        return recordings, requests_dict   
        
    def prepare_simulations(self):
        """
        Allows subclasses to prepares the simulations that are required by the chosen objective
        functions after they have been initially processed
        """
        pass
            
    def set_tune_parameters(self, tune_parameters):
        """
        Sets the parameters in which the candidate arrays passed to the 'run' method will map to in 
        respective order
        
        `tune_parameters` -- list of parameter names which correspond to the candidate order
        """
        raise NotImplementedError("Derived Simulation class '{}' does not implement "
                                  "'set_tune_parameters' method"
                                  .format(self.__class__.__name__))

    def run(self, candidate):
        """
        At a high level - accepts a candidate (a list of cell parameters that are being tuned)
        
        `candidate` -- a list of parameters [list(float)]
        
        Returns:
            A Neo block object with a segment for each Setup in self.simulation_setups containing
            an analogsignal for each requested recording 
        """
        raise NotImplementedError("Derived Simulation class '{}' does not implement _run_simulation"
                                  " method".format(self.__class__.__name__))
