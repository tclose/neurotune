from __future__ import absolute_import
from .__init__ import Objective


class MultiObjective(Objective):
    """
    A container class for multiple objectives, to be used with multiple
    objective optimisation algorithms
    """

    def __init__(self, *objectives):
        """
        A list of objectives that are to be combined

        `objectives` -- a list of Objective objects [list(Objective)]
        """
        self.objectives = list(objectives)

    def append(self, objective):
        self.objectives.append(objective)

    def __getitem__(self, i):
        return self.objectives[i]

    def __setitem__(self, i, val):
        self.objectives[i] = val

    def __len__(self):
        return len(self.objectives)

    def fitness(self, analysis):
        """
        Returns a inspyred.ec.emo.Pareto list of the fitness functions in the
        order the objectives were passed to the __init__ method
        """
        fitnesses = []
        for obj in self.objectives:
            fitnesses.append(obj.fitness(analysis.objective_specific(obj)))
        return fitnesses

    def get_recording_requests(self):
        # Zip the recording requests keys with objective object in a tuple to
        # guarantee unique keys
        recordings_request = {}
        for obj in self.objectives:
            # Get the recording requests from the sub-objective function
            obj_requests = obj.get_recording_requests()
            # Add the recording request to the collated dictionary
            recordings_request.update([((obj, key), val)
                                       for key, val in obj_requests.items()])
        return recordings_request


class WeightedSumObjective(MultiObjective):
    """
    Combines multiple objectives into a single weighted sum between objectives.
    Useful for optimization algorithms that require a single objective
    """

    def __init__(self, *weighted_objectives):
        """
        `weighted_objectives` -- a list of weight-objective pairs
                                 [list((float, Objective))]
        """
        self.weights, self.objectives = zip(*weighted_objectives)

    def fitness(self, analysis):
        """
        Returns a inspyred.ec.emo.Pareto list of the fitness functions in the
        order the objectives were passed to the __init__ method
        """
        weighted_sum = 0.0
        for weight, obj in zip(self.weights, self.objectives):
            weighted_sum += (weight *
                             obj.fitness(analysis.objective_specific(obj)))
        return weighted_sum
