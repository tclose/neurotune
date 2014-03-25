"""
Run the simulation 
"""
from collections import namedtuple
from itertools import groupby
from abc import ABCMeta # Metaclass for abstract base classes
from nineline.cells.neuron import NineCellMetaClass, simulation_controller as nineline_controller

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
        

class RecordingRequest(object):
    """"
    RecordingRequests are raised by objective functions and are passed to Simulation objects so they
    can set up the required simulation conditions (eg IClamp, VClamp, spike input) and recorders
    """
    
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
    "Base class of Simulation objects"
    
    __metaclass__ = ABCMeta # Declare this class abstract to avoid accidental construction
    
    ## Groups together all the information to interface to and from a requested simulation
    SimulationSetup = namedtuple('SimulationSetup', "time conditions record_variables request_keys")
        
    def _prepare_all(self, simulation_setups):
        """
        Prepares the simulations that are required by the chosen objective functions
        
        `simulation_setups` -- a of simulation setups [list(_Simulation.SimulationSetup)]
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
            common_record_vars = groupby(com_cond, 
                                          key=lambda x, y: x[1].record_site == y[1].record_site)
            # Get the common recording sites
            record_variables = [com_record[0][1].record_site for com_record in common_record_vars]
            # Get the list of request keys for each requested recording
            request_keys = [[key for key, val in com_record] for com_record in common_record_vars] #@UnusedVariable
            # Append the simulation request to the 
            self.simulation_setups.append(self.SimulationSetup(record_time, conditions, 
                                                               record_variables, request_keys))
        self._prepare_all()
        
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
        simulations = self._run_all(candidate)
        requests_dict = {}
        for simulation, setup in zip(simulations, self.simulation_setups):
            recordings = simulation.segments[0].analogsignalarrays
            assert len(recordings) == len(setup.request_keys)
            for recording, request_keys in zip(recordings, setup.request_keys):
                requests_dict.update([(key, recording) for key in request_keys])
        return requests_dict

    
class NineLineSimulation(_Simulation):
    "A simulation class for 9ml descriptions"
    
    def __init__(self, cell_9ml, genome_keys):
        """
        `cell_9ml`    -- A 9ml file [str]
        `genome_keys` -- A list of genome keys which are used to map the candidate parameters to 
                         parameters of the model [list(str)]
        """
        # Generate the NineLine class from the nineml file and initialise a single cell from it
        self.cell_9ml = cell_9ml
        self.celltype = NineCellMetaClass('TestCell', cell_9ml)
        self.genome_keys = ['source_section.' + k if isinstance(k, basestring) else '.'.join(k) 
                            for k in genome_keys]

    def _prepare_all(self):
        """
        Prepare all simulations (eg. create cells and set recorders if possible)
        """
        # Parse all recording sites into a tuple containing the variable name, segment name and 
        # component names
        for setup in self.simulation_setups:
            for i, rec in enumerate(setup.record_variables):
                parts = rec.split('.')
                if len(parts) == 1:
                    var = parts[0]
                    segname, component = None
                elif len(parts) == 2:
                    segname, var = parts
                    component = None
                else:
                    segname, component, var = parts
                setup.record_variables[i] = (var, segname, component)
        # Check to see if there are multiple setups, because if there aren't the cell can be 
        # initialised (they can't in general if there are multiple as there is only ever one 
        # instance of NEURON running)        
        if len(self.simulation_setups) == 1:
            self._prepare(self.simulation_setups[0])            

    def _run_all(self, candidate):
        """
        Run all simulations required to assess the candidate
        
        `candidate` -- a list of parameters [list(float)]
        """
        recordings = []
        for setup in self.simulation_setups:
            # If there aren't multiple simulation setups the same setup can be reused with just the
            # recorders being reset
            if len(self.simulation_setups) != 1:
                self._prepare(setup)
            else:
                self.cell.reset_recordings()
            self._set_candidate_params(candidate)
            nineline_controller.run(setup.time)
            recordings.append(self.cell.get_recording(*zip(setup.record_variables)))
        return recordings
        
    def _prepare(self, simulation_setup):
        """
        Initialises cell and sets recording sites. Record sites are delimited by '.'s into segment 
        names, component names and variable names. Sitenames without '.'s are interpreted as 
        properties of the default segment and site-names with only one '.' are interpreted as 
        (segment name - property) pairs. Therefore in order to record from component states you must
        also provide the segment name to disambiguate it from the segment name - property case. 
        
        `simulation_setup` -- A set of simulation setup instructions [_Simulation.SimulationSetup] 
        """
        #Initialise cell
        self.cell = self.celltype()
        for rec in simulation_setup.record_variables:
            self.cell.record(*rec)
            
    def _set_candidate_params(self, candidate):
        """
        Set the parameters of the candidate
        
        `candidate` -- a list of parameters [list(float)]
        """
        assert len(candidate) == len(self.genome_keys), "length of candidate and genome keys do not match"
        for key, val in zip(self.genome_keys, candidate):
            setattr(self.cell, key, val)

