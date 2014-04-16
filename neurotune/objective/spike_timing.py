from __future__ import absolute_import
from .__init__ import Objective
from ..simulation.__init__ import RecordingRequest


class SpikeFrequencyObjective(Objective):
    """
    A simple objective based on the squared difference between the spike frequencies
    """

    def __init__(self, frequency, time_start, time_stop):
        """
        `frequency`  -- the desired spike frequency
        `time_stop` -- the length of time to run the simulation
        """
        super(SpikeFrequencyObjective, self).__init__(time_start, time_stop)
        self.frequency = frequency

    def fitness(self, recordings):
        recording_frequency = len(recordings[recordings < self.time_stop]) / self.time_stop
        return (self.frequency - recording_frequency) ** 2

    def get_recording_requests(self):
        """
        Returns a RecordingRequest object or a dictionary of RecordingRequest objects with unique keys
        representing the recordings that are required from the simulation controller
        """
        return RecordingRequest(time_stop=self.time_stop, record_variable='spikes')
