"""
Run the simulation 
"""
from __future__ import absolute_import
from collections import namedtuple
from itertools import groupby
from abc import ABCMeta # Metaclass for abstract base classes
import neo
import quantities as pq


class ExperimentalConditions(object):
    """
    Defines the experimental conditions an objective function requires to make its evaluation. Can
    be extended for specific conditions required by novel objective functions but may not be 
    supported by all simulations, especially custom ones
    """
    
    class NotSupportedException(Exception): pass
    
    def __init__(self, initial_v=None):
        """
        `initial_v` -- the initial voltage of the membrane
        """
        self.initial_v = initial_v
        
    def __eq__(self, other):
        return self.initial_v == other.initial_v
    

class RecordingRequest(object):
    """"
    RecordingRequests are raised by objective functions and are passed to Simulation objects so they
    can set up the required simulation conditions (eg IClamp, VClamp, spike inumpyut) and recorders
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
        
    def _prepare_simulations(self):
        """
        Prepares the simulations that are required by the chosen objective functions
        
        `simulation_setups` -- a of simulation setups [list(Simulation.Setup)]
        """
        raise NotImplementedError("Derived Simulation class '{}' does not implement "
                                  "'_setup_a_simulation' method" .format(self.__class__.__name__))
            
    def set_tuneable_parameters(self, tuneable_parameters):
        """
        Sets the parameters in which the candidate arrays passed to the 'run' method will map to in 
        respective order
        
        `tuneable_parameters` -- list of parameter names which correspond to the candidate order
        """
        raise NotImplementedError("Derived Simulation class '{}' does not implement "
                                  "'set_tuneable_parameters' method"
                                  .format(self.__class__.__name__))
    
    def process_requests(self, recording_requests):
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
        self._prepare_simulations()
        
    def _run_all(self, candidate):
        """
        At a high level - accepts a candidate (a list of cell parameters that are being tuned)
        
        `candidate` -- a list of parameters [list(float)]
        """
        raise NotImplementedError("Derived Simulation class '{}' does not implement _run_simulation"
                                  " method".format(self.__class__.__name__))
                  
    def run(self, candidate):
        """
        Return the recordings in a dictionary to be returned to the objective functions, so each
        objective function can access the recording it requested
        
        `candidate` -- a list of parameters [list(float)]
        """
        all_recordings = self._run_all(candidate)
        requests_dict = {}
        for recordings, setup in zip(all_recordings, self.simulation_setups):
            assert len(recordings) == len(setup.request_keys)
            for recording, request_keys in zip(recordings, setup.request_keys):
                requests_dict.update([(key, recording) for key in request_keys])
        if len(requests_dict) == 1 and requests_dict.keys()[0] is None:
            requests_dict = requests_dict.values()[0]
        return requests_dict


class CustomSimulation(Simulation):
    """
    A convenient base class for custom simulation objects. Provides record time from requested 
    recordings
    """
    
    __metaclass__ = ABCMeta # Declare this class abstract to avoid accidental construction
        
    def set_tuneable_parameters(self, tuneable_parameters):
        """
        The body of this method is just here for convenience can be overridden if required.
        """
        self.tuneable_parameters = tuneable_parameters
            
    def _prepare_simulations(self):
        """
        Prepares the simulations that are required by the chosen objective functions
        
        `simulation_setups` -- a of simulation setups [list(Simulation.Setup)]
        """
        if (len(self.simulation_setups) != 1 or 
            self.simulation_setups[0].record_variables != [None] or
            self.simulation_setups[0].conditions is not None):
            raise Exception("This custom simulation '{}' can only handle default recordings "
                            "(typically voltage traces from the soma)"
                            .format(self.__class__.__name__))
        self.record_time = self.simulation_setups[0].time
        
    def _run_all(self, candidate):
        """
        Wraps the 'simulate' method in a list to be returned to the Simulation.run method
        
        `candidate` -- a list of parameters [list(float)]
        """
        volt_traces, times = self.simulate(candidate)
        if not isinstance(volt_traces, list):
            volt_traces = [volt_traces]
            times = [times]
        recordings = []
        for v, t in volt_traces, times:
            # If t is a timestep rather than a time vector
            if isinstance(t, float):
                sampling_period = t * pq.ms
                t_start = 0.0 * pq.ms
                t_stop = t * len(v) * pq.ms
                rec = neo.AnalogSignal(v, sampling_period=sampling_period, t_start=t_start, 
                                       t_stop=t_stop, name='custom_simulation', units='mV')
            else:
                rec = neo.IrregularlySampledSignal(t, v, units='mV', time_units='ms')
            recordings.append(rec) 
        return recordings
            
    def simulate(self, candidate):
        """
        At a high level - accepts a candidate (a list of cell parameters that are being tuned)
        
        `candidate` -- a list of parameters [list(float)]
        returns     -- a tuple consisting of a voltage vector and a time vector or timestep
        """
        raise NotImplementedError("Derived SimpleCustomSimulation class '{}' does not implement "
                                  "the 'simulate' method".format(self.__class__.__name__))                    
