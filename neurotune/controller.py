"""
Run the simulation 
"""
from abc import ABCMeta # Metaclass for abstract base classes
from nineline.cells.neuron import 

class _Controller():
    """
    _Controller base class
    """
    
    __metaclass__ = ABCMeta # Declare this class abstract to avoid accidental construction

    def set_objective(self, objective):
        """
        Sets the up the simulation to return the data required by the objective function
        """
        raise NotImplementedError("Derived Controller class does not implement set_objective method")

    def run(self, candidate):
        """
        At a high level - accepts a list of parameters and chromosomes
        and (usually) returns corresponding simulation data. This is
	    implemented polymporphically in subclasses.
        """
        raise NotImplementedError("Derived Controller class does not implement run method")
    
    
class NineLineController(_Controller):
    
    def __init__(self, nineml_filename, genome_keys, objective):
        CellClass = self._NineCellMetaClass('TestCell', nineml_filename)
        self.cell = CellClass()
        self.genome_keys = ['{soma}' + k if isinstance(k, basestring) else '{' + k[0] + '}' + k[2] 
                            for k in genome_keys]
        missing_keys = [k for k in self.genome_keys if not hasattr(self.cell, k)]
        if missing_keys:
            raise Exception("The following genome keys were not attributes of test cell: '{}'"
                            .format("', '".join(missing_keys)))

    def set_objective(self, objective):
        """
        Initialises the simulator object and is passed the objective function to set any requirements
        such as recorders
        """
        self.recording_sites = objective.recording_sites

    def run(self, candidate):
        raise NotImplementedError

