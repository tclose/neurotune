import os.path
import numpy
import quantities as pq
import neo
import neuron
import neurotune as nt

neuron.load_mechanisms(os.path.join(os.path.dirname(__file__), 'masas_io'))

class MasasIOSimulation(nt.simulation.SimpleCustomSimulation):
    
    class SingleIO(object):
        def __init__(self, g_l, g_ca, e_l):
            soma = neuron.h.Section()
            soma.diam=25
            soma.L=25
            soma.nseg = 1
            soma.insert('leak')
            soma.insert('stoca')
##            soma.insert('iona')
##            soma.insert('iokdr')
            for seg in soma:
                seg.gbar_leak = g_l
                seg.el_leak = e_l
                seg.gbar_stoca = g_ca
##                seg.gbar_iona = g_na
##                seg.gbar_iokdr = g_k
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
            return vec['v_soma'], vec['t']
            
        def getRecordedVm(self,vec1,vec2) : # traslate hoc data to array
            rec = {}
            rec['t'] = numpy.array(vec2)
            rec['vm'] = numpy.array(vec1)
            return rec['t'], rec['vm']
    
        def runsim(self):    # run simulation in neuron simlator
            neuron.h.finitialize(-55)
            neuron.run(3000)
            
    def simulate(self, candidate):
        io_cell = self.SingleIO(*candidate)
        return io_cell.recordedV, io_cell.recordedt

# genome = [0.1275, 0.16153846, 0.65]
genome = [0.3745782626692236, 0.3393107817168788, 0.635192963911868]
genome_test = [0.5208486893578104, 0.8059478259307578, 0.6659825491055114]
# genome=[0.9289666791770207, 0.8142125763855085, 0.6273299121718918]
# genome = [0.7606730549817866, 0.5701091010683264, 0.6248730834352871]
# genome = [0.7603275525543237, 0.5701001144176526, 0.6248729123236844]
genome = [0.5458158539772086, 0.4456162523542067, 0.6287524521307443]
# genome_centroid = (0.3, 0.85, -60)
# genome_test =  (0.4, 1.0, -55)
simulation = MasasIOSimulation()
v, _ = simulation.simulate(genome)
reference_trace = neo.AnalogSignal(v, sampling_rate=1.0 / (neuron.h.dt * pq.ms), units='ms')
objective = nt.objective.PhasePlaneHistObjective(reference_trace, time_start=500.0, 
                                                 time_stop=2000.0)    
#objective = MasasIOObjective()
print "Centroid error (should be 0): {}".format(objective.fitness(reference_trace))

test_v, _ = simulation.simulate(genome_test)
test_trace = neo.AnalogSignal(test_v, sampling_rate=1.0 / (neuron.h.dt * pq.ms), units='ms')
print "Test error: {}".format(objective.fitness(test_trace))

tuner = nt.Tuner([nt.Parameter('g_l', 'uS', 0.1, 0.5),
                  nt.Parameter('g_ca', 'uS', 0.2, 1.5),
                  nt.Parameter('e_l', 'mV', -70, -50)], 
                 objective, 
                 nt.algorithm.inspyred.EDAAlgorithm(), 
                 simulation)
pop, ea = tuner.tune(num_candidates=10, max_iterations=100)
