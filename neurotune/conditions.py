
class ExperimentalConditions(object):
    """
    Defines the experimental conditions an objective function requires to make its evaluation. Can
    be extended for specific conditions required by novel objective functions but may not be 
    supported by all simulations, especially custom ones
    """
    
    def __init__(self, initial_v=None, clamps=[]):
        """
        `initial_v` -- the initial voltage of the membrane
        """
        self.initial_v = initial_v
        self.clamps = set(clamps)
        
    def __eq__(self, other):
        return self.initial_v == other.initial_v and self.clamps == other.clamps
    
    
class StepCurrentSource(object):
    
    def __init__(self, amplitudes, times):
        self.amplitudes = amplitudes
        self.times = times
        
    def __eq__(self, other):
        return self.amplitudes == other.amplitudes and self.times == other.times