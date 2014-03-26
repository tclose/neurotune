"""
Run the simulation 
"""
from collections import namedtuple
from itertools import groupby
from abc import ABCMeta # Metaclass for abstract base classes
import numpy
from nineline.cells.neuron import NineCellMetaClass, simulation_controller as nineline_controller
import neuron
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
        `record_variable` -- the name of the section/synapse/current to record from (controller specific)
        `conditions`  -- the experimental conditions required (eg. initial voltage, current clamp)
        """
        self.record_time = record_time
        self.record_variable = record_variable
        self.conditions = conditions
        

class _Simulation():
    "Base class of Simulation objects"
    
    __metaclass__ = ABCMeta # Declare this class abstract to avoid accidental construction
    
    ## Groups together all the information to interface to and from a requested simulation
    SimulationSetup = namedtuple('SimulationSetup', "time conditions record_variables request_keys")
        
    def _prepare_all(self):
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
        
        `recording_requests`  -- a list of recording requests from the objective functions [.RecordingRequest]
        """
        # Group into requests by common experimental conditions
        request_items = recording_requests.items()
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
            record_variables, requests_iters = zip(*common_record_variables)
            # Get the list of request keys for each requested recording
            request_keys = [zip(*com_record)[0] for com_record in requests_iters]
            # Append the simulation request to the 
            self.simulation_setups.append(self.SimulationSetup(record_time, conditions, 
                                                               list(record_variables),
                                                               request_keys))
        # Do initial preparation for simulation (how much preparation can be done depends on whether
        # the same experimental conditions are used throughout the evaluation process.
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
        all_recordings = self._run_all(candidate)
        requests_dict = {}
        for recordings, setup in zip(all_recordings, self.simulation_setups):
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
        if isinstance(genome_keys, basestring):
            raise Exception("'genome_keys' argument should be a list of keys not a single key")
        # Generate the NineLine class from the nineml file and initialise a single cell from it
        self.cell_9ml = cell_9ml
        self.celltype = NineCellMetaClass(cell_9ml)
        self.default_seg = self.celltype().source_section.name
        self.genome_keys = [self.default_seg + '.' + k if isinstance(k, basestring) else '.'.join(k) 
                            for k in genome_keys]

    def _prepare_all(self):
        """
        Prepare all simulations (eg. create cells and set recorders if possible)
        """
        # Parse all recording sites into a tuple containing the variable name, segment name and 
        # component names
        for setup in self.simulation_setups:
            for i, rec in enumerate(setup.record_variables):
                if rec is None:
                    var = 'v' # Records the voltage in the default segment by default
                    segname = self.default_seg
                    component = None
                else:
                    parts = rec.split('.')
                    if len(parts) == 1:
                        var = parts[0]
                        segname = self.default_seg
                        component = None
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
            recordings.append(self.cell.get_recording(*zip(*setup.record_variables)))
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


class _CustomSimulation(_Simulation):
    """
    A convenient base class for custom simulation objects. Provides record time from requested 
    recordings
    """
    
    __metaclass__ = ABCMeta # Declare this class abstract to avoid accidental construction
    
            
    def _prepare_all(self):
        """
        Prepares the simulations that are required by the chosen objective functions
        
        `simulation_setups` -- a of simulation setups [list(_Simulation.SimulationSetup)]
        """
        if (len(self.simulation_setups) != 1 or self.simulation_setups.record_variable is not None 
            or self.simulation_setups.conditions is not None):
            raise Exception("Custom simulation '{}' can only handle default recordings (typically "
                            "voltage traces from the soma)".format(self.__class__.__name__))
        self.record_time = self.simulation_setups.record_time
        
    def _run_all(self, candidate):
        """
        Wraps the _run method in a list to be returned to the _Simulation.run method
        
        `candidate` -- a list of parameters [list(float)]
        """
        t, v = self._run(candidate)
        return [neo.AnalogSignal(v, sampling_period = t[1] - t[0] * pq.ms, 
                                 t_start=t[0]* pq.ms, name='custom_simulation', units='ms')]
            
    def _run(self, candidate):
        """
        At a high level - accepts a candidate (a list of cell parameters that are being tuned)
        
        `candidate` -- a list of parameters [list(float)]
        returns     -- a time and voltage vector
        """
        raise NotImplementedError("Derived Simulation class '{}' does not implement '_run'"
                                  " method".format(self.__class__.__name__))            
            

class MasasIOSimulation(_CustomSimulation):
    
    class singleIO :
        def __init__(self, g_l, g_ca, e_l):
            soma = neuron.h.Section()
            soma.diam=25
            soma.L=25
            soma.nseg = 1
            soma.insert('leak')
            soma.insert('stoca')
    ##        soma.insert('iona')
    ##        soma.insert('iokdr')
            for seg in soma:
                seg.gbar_leak = g_l
                seg.el_leak = e_l
                seg.gbar_stoca = g_ca
    ##            seg.gbar_iona = g_na
    ##            seg.gbar_iokdr = g_k
            self.soma = soma #put soma on the class
            vec1, vec2 = self.record()
            self.runsim()
            self.recordedt, self.recordedV = self.getRecordedVm(vec1,vec2)
    
        def record(self): #record t and Vm from hoc to vector
            vec = {}
            for var in 'v_soma', 'i_stoca','t':
                vec[var] = neuron.h.Vector()
            vec['v_soma'].record(self.soma(0.5)._ref_v) #record Vm of soma
            vec['t'].record(neuron.h._ref_t) #record time
    
            return vec['v_soma'],vec['t']
            
        def getRecordedVm(self,vec1,vec2) : # traslate hoc data to array
            rec = {}
            rec['t'] = numpy.array(vec2)
            rec['vm'] = numpy.array(vec1)
            tsize = len(vec2)
        ##    print 'Vm recorded'
            return rec['t'], rec['vm']
    
        def runsim(self):    # run simulation in neuron simlator
            #h.v_init = -55
            #h.tstop = 3000.0
            #h.init()
            neuron.h.finitialize(-55)
            neuron.run(3000)
            
    def _run(self, candidate):
        io_cell = self.singleIO(*candidate)
        return io_cell.recordedt, io_cell.recordedV
        
