from __future__ import absolute_import
import numpy
import quantities as pq
import neo.core
from . import Objective


class SpikeFrequencyObjective(Objective):
    """
    A simple objective based on the squared difference between the spike
    frequencies
    """

    def __init__(self, frequency, time_start=500.0, time_stop=2000.0):
        """
        `frequency`  -- the desired spike frequency [quantities.Quantity]
        `time_start` -- the time from which to start calculating the frequency
        `time_stop`  -- the length of time to run the simulation
        """
        super(SpikeFrequencyObjective, self).__init__(time_start, time_stop)
        self.frequency = pq.Quantity(frequency, units='Hz')

    def fitness(self, analysis):
        signal = analysis.get_signal(t_start=self.time_start,
                                     t_stop=self.time_stop)
        frequency = signal.spike_frequency()
        return (self.frequency - frequency) ** 2


class SpikeTimesObjective(Objective):
    """
    A simple objective based on the squared difference spike times and the
    nearest spike in the reference set and vice versa.
    """

    def __init__(self, spikes, time_start=500.0, time_stop=2000.0):
        """
        `spikes`    -- the reference spike train [neo.SpikeTrain]
        `time_start` -- the time from which to start including spikes [float]
        `time_stop` -- the length of time to run the simulation [float]
        """
        super(SpikeTimesObjective, self).__init__(time_start, time_stop)
        if not isinstance(spikes, neo.core.SpikeTrain):
            raise Exception("Spikes must be a neo.core.SpikeTrain object not "
                            "{}".format(type(spikes)))
        self.reference_spikes = spikes

    def fitness(self, analysis):
        """
        Calculates the sum squared difference between each spike in the
        signal and the closest spike in the reference spike train, plus the
        vice-versa case

        `signal` -- the recorded signal
        """
        signal = analysis.get_signal(t_start=self.time_start,
                                     t_stop=self.time_stop)
        spikes = signal.spike_times()
        fitness = 0.0
        for spike in spikes:
            fitness += float(numpy.square(self.reference_spikes - spike).min())
        for ref_spike in self.reference_spikes:
            fitness += float(numpy.square(spikes - ref_spike).min())
        return fitness
