from __future__ import absolute_import
import numpy
from .__init__ import Algorithm


class GridAlgorithm(Algorithm):
    
    def __init__(self, num_steps):
        self.num_steps = num_steps
        
    def _set_tuneable_parameters(self, tuneable_parameters):
        super(GridAlgorithm, self)._set_tuneable_parameters(tuneable_parameters)
        if not isinstance(self.num_steps, int):
            if len(self.num_steps) != self.num_dims:
                raise Exception("Number of tuneable parameters ({}) does not match number num_steps"
                                " provided to GridAlgorithm constructor ({})"
                                .format(len(self.num_steps) != self.num_dims))
    
    @property                
    def num_dims(self):
        try:
            return len(self.constraints)
        except AttributeError:
            raise Exception("Tuneable parameters have not been set for grid algorithm so it doesn't"
                            " have a dimension")
    
    def optimize(self, evaluator, **kwargs):
        
        # Convert number of steps into a list with a step number for each dimension if it is not
        # already
        if isinstance(self.num_steps, int):
            num_steps = numpy.empty(self.num_dims, dtype=int)
            num_steps.fill(self.num_steps)
        else:
            num_steps = self.num_steps
        # Get the ranges of the parameters using the number of steps
        param_ranges = [numpy.linspace(l, u, n) for (l, u), n in zip(self.constraints, num_steps)]
        # Get all permutations of candidates given parameter ranges
        meshes = numpy.meshgrid(*param_ranges)
        cand_mesh = numpy.concatenate([mesh.reshape([1] + list(mesh.shape)) for mesh in meshes])
        candidates = cand_mesh.reshape((self.num_dims, -1)).T
        # Evaluate fitnesses
        fitnesses = evaluator(candidates)
        # Get fittest candidate
        fittest_candidate = candidates[numpy.argmin(fitnesses), :]
        # return fittest candidate and grid of fitnesses (for plotting potentially)
        return fittest_candidate, numpy.reshape(fitnesses, num_steps)
        