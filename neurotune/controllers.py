"""
Run the simulation 
"""
from collections import namedtuple
from itertools import groupby
from abc import ABCMeta # Metaclass for abstract base classes
from nineline.cells.neuron import NineCellMetaClass


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
        

class _Controller():
    """
    _Controller base class
    """
    
    __metaclass__ = ABCMeta # Declare this class abstract to avoid accidental construction
    
    SimulationSetup = namedtuple('SimulationSetup', "time conditions recording_sites request_keys")
        
    def _prepare_simulations(self, simulation_setups):
        """
        Prepares the simulations that are required by the chosen objective functions
        """
        raise NotImplementedError("Derived Controller class '{}' does not implement "
                                  "'_setup_a_simulation' method" .format(self.__class__.__name__))
    
    def process_requests(self, recording_requests):
        """
        Merge recording requests so that the same recording/simulation doesn't get performed 
        multiple times
        
        `recording_requests`  -- a list of recording requests from the objective functions
        """
        # Group into requests by common experimental conditions
        common_conditions = groupby(recording_requests.items(), 
                                    key=lambda r1, r2: r1[1].conditions == r2[1].conditions)
        self.simulation_setups = []
        for com_cond in common_conditions:
            # Get the conditions object which is common to the group
            conditions = com_cond[0][1].conditions
            # Get the maxium record time in the group
            record_time = max([r[1].record_time for r in com_cond])
            # Group the requests by common recording sites
            common_record_sites = groupby(com_cond, 
                                          key=lambda r1, r2: r1[1].record_site == r2[1].record_site)
            # Get the common recording sites
            recording_sites = [com_record[0][1].record_site for com_record in common_record_sites]
            # Get the list of request keys for each requested recording
            request_keys = [[kv[0] for kv in com_record] for com_record in common_record_sites]
            # Append the simulation request to the 
            self.simulation_setups.append(self.SimulationSetup(record_time, conditions, 
                                                               recording_sites, request_keys))
        self._prepare_simulations(self.simulation_setups)
            
        
    def _run_simulations(self, candidate):
        """
        At a high level - accepts a candidate (a list of cell parameters that are being tuned)
        """
        raise NotImplementedError("Derived Controller class '{}' does not implement _run_simulation"
                                  " method".format(self.__class__.__name__))
                  
    def run(self, candidate):
        """
        Return the recordings in a dictionary to be returned to the objective functions, so each
        objective function can access the recording it requested
        """
        simulations = self._run_simulations(candidate)
        requests_dict = {}
        for simulation, setup in zip(simulations, self.simulation_setups):
            assert len(simulation.recordings) == len(setup.request_keys)
            for recording, request_keys in zip(simulation.recordings, setup.request_keys):
                requests_dict.update([(key, recording) for key in request_keys])
        return requests_dict

    
class NineLineController(_Controller):
    
    def __init__(self, nineml_filename, genome_keys):
        # Generate the NineLine class from the nineml file and initialise a single cell from it
        self.celltype = NineCellMetaClass('TestCell', nineml_filename)
        self.cell = self.celltype()
        # Translate the genome keys into attribute names for NineLine cells
        self.genome_keys = ['{soma}' + k if isinstance(k, basestring) else '{' + k[0] + '}' + k[2] 
                            for k in genome_keys]
        # Check to see if any of the keys are missing
        missing_keys = [k for k in self.genome_keys if not hasattr(self.cell, k)]
        if missing_keys:
            raise Exception("The following genome keys were not attributes of test cell: '{}'"
                            .format("', '".join(missing_keys)))

    def _prepare_simulations(self, simulation_setups):
        pass

    def _run_simulation(self, candidate, simulation_setup):
        pass

