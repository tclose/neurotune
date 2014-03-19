"""
Run the simulation 
"""
from itertools import groupby
from abc import ABCMeta # Metaclass for abstract base classes
from nineline.cells.neuron import NineCellMetaClass


class ExperimentalConditions(object):
    
    def __init__(self, initial_v=None):
        self.initial_v = initial_v
        

class RecordingRequest(object):
    
    def __init__(self, key, record_site, record_time, conditions=None):
        self.keys = [key]
        self.record_times = [record_time]
        self.record_site = record_site
        self.conditions = conditions
        
    @property
    def key(self):
        return self.keys[0]
    
    @property
    def record_time(self):
        return max(self.record_times)
    
    @classmethod
    def matching_conditions(cls, a, b):
        return a.record_site == b.record_site and a.conditions == b.conditions


class _Controller():
    """
    _Controller base class
    """
    
    __metaclass__ = ABCMeta # Declare this class abstract to avoid accidental construction

    def _merge_recording_requests(self, recording_requests):
        """
        Merge recording requests so that the same recording/simulation doesn't get performed multiple times
        
        `recording_requests`  -- a list of recording requests from the objective functions
        """
        self.requested_recordings = []
        matched_groups = groupby(recording_requests, key=RecordingRequest.matching_conditions)
        for group in matched_groups:
            group_request = group[0]
            for req in group[1:]:
                group_request.keys.append(req.key)
                group_request.record_times.append(req.record_time)
            self.requested_recordings.append(group_request)
            
    def _return_requested_recordings(self, recordings):
        """
        Return the recordings in a dictionary to be returned to the objective functions, so each
        objective function can access the recording it requested
        """
        request_dict = {}
        assert len(recordings) == len(self.requested_recordings)
        for recording, request in zip(recordings, self.requested_recordings):
            for key in request.keys:
                #TODO: trim recordings that don't require the fully recorded time
                if request_dict.has_key(key):
                    raise Exception("Duplicate keys '{}' found in recording request".format(key))
                request_dict[key] = recording
        return request_dict
            
    def set_recording_requests(self, recording_requests):
        """
        Sets the recordings that are required from the simulation
        """
        raise NotImplementedError("Derived Controller class '{}' does not implement "
                                  "'set_recording_request' method" .format(self.__class__.__name__))

    def run(self, candidate):
        """
        At a high level - accepts a list of parameters and chromosomes
        and (usually) returns corresponding simulation data. This is
	    implemented polymporphically in subclasses.
        """
        raise NotImplementedError("Derived Controller class '{}' does not implement run method"
                                  .format(self.__class__.__name__))
    
    
class NineLineController(_Controller):
    
    def __init__(self, nineml_filename, genome_keys):
        # Generate the NineLine class from the nineml file and initialise a single cell from it
        self.cell = NineCellMetaClass('TestCell', nineml_filename)()
        # Translate the genome keys into attribute names for NineLine cells
        self.genome_keys = ['{soma}' + k if isinstance(k, basestring) else '{' + k[0] + '}' + k[2] 
                            for k in genome_keys]
        # Check to see if any of the keys are missing
        missing_keys = [k for k in self.genome_keys if not hasattr(self.cell, k)]
        if missing_keys:
            raise Exception("The following genome keys were not attributes of test cell: '{}'"
                            .format("', '".join(missing_keys)))

    

    def run(self, candidate):
        raise NotImplementedError

