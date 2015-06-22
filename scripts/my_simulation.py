import neo
from copy import copy
import quantities as pq
try:
    from neurotune.tuner.mpi import MPITuner as Tuner  # @UnusedImport
except ImportError:
    from neurotune import Tuner  # @Reimport
from neurotune.simulation import Simulation


class MySimulation(Simulation):

    def __init__(self):
        pass

    def set_tune_parameters(self, tune_parameters):
        super(MySimulation, self).set_tune_parameters(tune_parameters)
        self.tuneable_parameter_names = [p.name for p in tune_parameters]

    def run(self, candidate, setup):
        params = copy(self.all_parameters)
        params.update(zip(self.tuneable_parameter_names, candidate))
        
        
        recordings = neo.Segment()
        sig = neo.AnalogSignal(signal, sampling_period=dt * pq.s,
                               units='mV')
        recordings.analogsignals.append(sig)
        return recordings

if __name__ == '__main__':
    simulation = MySimulation()
    simulation.set_tune_parameters([])
    recordings = simulation.run([], None)
