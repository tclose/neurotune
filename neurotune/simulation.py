"""
Run the simulation 
"""
from collections import namedtuple
from itertools import groupby
from abc import ABCMeta # Metaclass for abstract base classes
import nineline.pyNN.neuron


class ExperimentalConditions(object):
    
    def __init__(self, initial_v=None):
        self.initial_v = initial_v
        

class RecordingRequest(object):
    
    def __init__(self, record_time, record_site=None, conditions=None):
        """
        `record_time` -- the length of the recording required by the simulation
        `record_site` -- the name of the section/synapse/current to record from (controller specific)
        `conditions`  -- the experimental conditions required (eg. initial voltage, current clamp)
        """
        self.record_time = record_time
        self.record_site = record_site
        self.conditions = conditions
        

class _Simulation():
    """
    _Simulation base class
    """
    
    __metaclass__ = ABCMeta # Declare this class abstract to avoid accidental construction
    
    SimulationSetup = namedtuple('SimulationSetup', "time conditions recording_sites request_keys")
        
    def _prepare_all(self, simulation_setups):
        """
        Prepares the simulations that are required by the chosen objective functions
        """
        raise NotImplementedError("Derived Simulation class '{}' does not implement "
                                  "'_setup_a_simulation' method" .format(self.__class__.__name__))
    
    def process_requests(self, recording_requests):
        """
        Merge recording requests so that the same recording/simulation doesn't get performed 
        multiple times
        
        `recording_requests`  -- a list of recording requests from the objective functions
        """
        # Group into requests by common experimental conditions
        common_conditions = groupby(recording_requests.items(), 
                                    key=lambda x, y: x[1].conditions == y[1].conditions)
        self.simulation_setups = []
        for com_cond in common_conditions:
            # Get the conditions object which is common to the group
            conditions = com_cond[0][1].conditions
            # Get the maxium record time in the group
            record_time = max([r[1].record_time for r in com_cond])
            # Group the requests by common recording sites
            common_record_sites = groupby(com_cond, 
                                          key=lambda x, y: x[1].record_site == y[1].record_site)
            # Get the common recording sites
            recording_sites = [com_record[0][1].record_site for com_record in common_record_sites]
            # Get the list of request keys for each requested recording
            request_keys = [[key for key, val in com_record] for com_record in common_record_sites] #@UnusedVariable
            # Append the simulation request to the 
            self.simulation_setups.append(self.SimulationSetup(record_time, conditions, 
                                                               recording_sites, request_keys))
        self._prepare_all()
        
    def _run_all(self, candidate):
        """
        At a high level - accepts a candidate (a list of cell parameters that are being tuned)
        """
        raise NotImplementedError("Derived Simulation class '{}' does not implement _run_simulation"
                                  " method".format(self.__class__.__name__))
                  
    def run(self, candidate):
        """
        Return the recordings in a dictionary to be returned to the objective functions, so each
        objective function can access the recording it requested
        """
        simulations = self._run_all(candidate)
        requests_dict = {}
        for simulation, setup in zip(simulations, self.simulation_setups):
            recordings = simulation.segments[0].analogsignalarrays
            assert len(recordings) == len(setup.request_keys)
            for recording, request_keys in zip(recordings, setup.request_keys):
                requests_dict.update([(key, recording) for key in request_keys])
        return requests_dict

    
class NineLineSimulation(_Simulation):
    
    @classmethod
    def _add_default_segment(cls, keys):
        return ['{soma}' + k if isinstance(k, basestring) else '{' + k[0] + '}' + k[1] for k in keys]
    
    def __init__(self, cell_9ml, genome_keys):
        # Generate the NineLine class from the nineml file and initialise a single cell from it
        self.cell_9ml = cell_9ml #NineCellMetaClass('TestCell', nineml_filename)
        # Translate the genome keys into attribute names for NineLine cells
        self.genome_keys = self._add_default_segment(genome_keys)
        # Check to see if any of the keys are missing
        missing_keys = [k for k in self.genome_keys if not hasattr(self.cell, k)]
        if missing_keys:
            raise Exception("The following genome keys were not attributes of test cell: '{}'"
                            .format("', '".join(missing_keys)))

    def _prepare_all(self):
        # Check to see if there are multiple setups, because if there aren't the cell can be 
        # initialised (they can't in general if there are multiple as there is only ever one 
        # instance of NEURON running)
        if len(self.simulation_setups) == 1:
            self._prepare(self.simulation_setups[0])            

    def _run_all(self, candidate):
        recordings = []
        for setup in self.simulation_setups:
            nineline.pyNN.neuron.reset()
            if len(self.simulation_setups) != 1:
                self._prepare(setup)
            self._set_candidate_params(candidate)
            nineline.pyNN.neuron.run(setup.time)
            recordings.append(self.pop.get_data())
        return recordings
        
    def _prepare(self, simulation_setup):
        #Initialise cell
        self.pop, self.cell = nineline.pyNN.neuron.create_singleton_population(self.cell_9ml)
        for record_site in self._add_default_segment(simulation_setup.record_sites):
            self.pop.record(record_site)
            
    def _set_candidate_params(self, candidate):
        for key, val in zip(self.genome_keys, candidate):
            setattr(self.cell, key, val)

