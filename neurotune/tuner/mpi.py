from __future__ import absolute_import
import sys
import os
import time
from copy import deepcopy
import shutil
import subprocess
from copy import copy
from collections import deque
from mpi4py import MPI
from .__init__ import Tuner


class MPITuner(Tuner):
    """
    A tuner class that runs the optimisation algorithm on a master node and distributes candidates
    to slave nodes for simulation and evaluation of their fitness
    """ 
    
    MASTER = 0      # The processing node to be used for the master node
    COMMAND_MSG = 1 # A message tag signifying that the passed message is a command to a slave node
    DATA_MSG = 2    # A message tag signifying that the passed message is data returned by a slave node
    
    comm = MPI.COMM_WORLD           # The MPI communicator object
    rank = comm.Get_rank()          # The ID of the current process
    num_processes = comm.Get_size() # The number of processes available
       
    def set(self, *args, **kwargs):
        self.evaluate_on_master = kwargs.pop('evaluate_on_master', self.num_processes < 10)
        self.mpi_verbose = kwargs.pop('verbose', True)
        super(MPITuner, self).set(*args, **kwargs)         
    
    @classmethod
    def is_master(cls):
        """
        Checks if current processing node is the master
        """
        return cls.rank == cls.MASTER       
       
    def tune(self, **kwargs):
        """
        Runs the optimisation algorithm and returns the final population and algorithm state
        
        `num_candidates`  -- the number of candidates to use in the algorithm
        `max_iterations`  -- the maximum number of iterations to perform
        `random_seed`     -- the seed to initialize the candidates with
        `stats_filename`  -- the name of the file to save the generation-based statistics in
        `indiv_filename`  -- the name of the file to save the candidate parameters in
        `kwargs`          -- optional arguments to be passed to the optimisation algorithm
        """
        
        if self.is_master():
            result = self.algorithm.optimize(self._distribute_candidates_for_evaluation, **kwargs)
            self._release_slaves()
        else:
            self._listen_for_candidates_to_evaluate()
            result = (None, None)
        return result
    
    def __del__(self):
        """
        Calls MPI finalize, so care should be taken not to let MPITuner objects to go out of scope
        if you have multiple instantiations (why you would have multiple instantiations I am not 
        sure though as you can use 'set' to re-purpose an existing Tuner)
        """
        MPI.Finalize()

    def _distribute_candidates_for_evaluation(self, candidates, args=None): #@UnusedVariable args
        """
        Run on the master node, this method distributes candidates to to the slave nodes to be 
        evaluated then collates their results into a single numpy vector
        
        `candidates`  -- candidates to be evaluated
        `args`        -- unused but supplied for compatibility with inspyred library
        """
        assert self.is_master(), "Distribution of candidate jobs should only be performed by master node"
        candidate_jobs = deque(enumerate(candidates))
        free_processes = deque(xrange(1, self.num_processes))
        # Create a list of empty lists the same length as the candidate list
        evaluations = [[]] * len(candidates)
        remaining_evaluations = len(candidates)
        if self.evaluate_on_master:
            since_master_evaluation = 0
        while candidate_jobs:
            if free_processes:
                self.comm.send(candidate_jobs.pop(), dest=free_processes.pop(), 
                               tag=self.COMMAND_MSG)
                if self.evaluate_on_master:
                    since_master_evaluation += 1
                    if since_master_evaluation == self.num_processes - 1:
                        jobID, candidate = candidate_jobs.pop()
                        if self.mpi_verbose:
                            print ("Evaluating jobID: {}, candidate: {} on process {}"
                                   .format(jobID, candidate, self.rank))
                        evaluations[jobID] = self._evaluate_candidate(candidate)
                        remaining_evaluations -= 1
                        since_master_evaluation = 0
            else:    
                processID, jobID, result = self.comm.recv(source=MPI.ANY_SOURCE, tag=self.DATA_MSG)
                evaluations[jobID] = result
                remaining_evaluations -= 1
                free_processes.append(processID)
        while remaining_evaluations:
            processID, jobID, result = self.comm.recv(source=MPI.ANY_SOURCE, tag=self.DATA_MSG)
            evaluations[jobID] = result
            remaining_evaluations -= 1
        return evaluations

    def _listen_for_candidates_to_evaluate(self):
        """
        Run on the slave nodes, this method receives candidates to evaluate from the master node,
        evaluates them and sends back the master
        """
        assert not self.is_master(), "Evaluation of candidates should only be performed by slave nodes"
        command = self.comm.recv(source=self.MASTER, tag=self.COMMAND_MSG)
        while command != 'stop':
            jobID, candidate = command
            if self.mpi_verbose:
                print "Evaluating jobID: {}, candidate: {} on process {}".format(jobID, candidate, 
                                                                                 self.rank)
            evaluation = self._evaluate_candidate(candidate)
            self.comm.send((self.rank, jobID, evaluation), dest=self.MASTER, tag=self.DATA_MSG)
            command = self.comm.recv(source=self.MASTER, tag=self.COMMAND_MSG)
        if self.mpi_verbose:
            print "Stopping listening on process {}".format(self.rank)

        
    def _release_slaves(self):
        """
        Release slave nodes from listening to new candidates to evaluate
        """
        for processID in xrange(1, self.num_processes):
            self.comm.send('stop', dest=processID, tag=self.COMMAND_MSG)


class SGESubmitter(object):
    """
    This class automates the submission of MPI jobs on a SGE-powered high-performance computing 
    cluster
    """

#     PYTHON_INSTALL_DIR='/apps/python/272'
#     OPEN_MPI_INSTALL_DIR='/opt/mpi/gnu/openmpi-1.6.3'
#     NEURON_INSTALL_DIR='/apps/DeschutterU/NEURON-7.3'
#     NEST_INSTALL_DIR='/apps/DeschutterU/nest-2.2.1'
#     SUNDIALS_INSTALL_DIR='/apps/DeschutterU/sundials-2.5.0'
#     NEUROFITTER_INSTALL_DIR='/home/t/tclose/git/neurofitter/bin'
    
    def __init__(self, python_install_dir=None, open_mpi_install_dir=None, neuron_install_dir=None, 
                 nest_install_dir=None, sundials_install_dir=None):
        self.py_dir=python_install_dir if python_install_dir else os.environ.get('PYTHONHOME', None)
        self.mpi_dir=(open_mpi_install_dir 
                      if open_mpi_install_dir else os.environ.get('MPIHOME', None))
        self.nrn_dir=neuron_install_dir if neuron_install_dir else os.environ.get('NRNHOME', None)
        self.nest_dir=nest_install_dir if nest_install_dir else os.environ.get('NESTHOME', None)
        self.sdials_dir=(sundials_install_dir 
                         if sundials_install_dir else os.environ.get('SUNDIALSHOME', None))
    
    def add_sge_arguments(self, parser, np=256, que_name='short', max_memory='3g', 
                          virtual_memory='2g'):
        original_actions = copy(parser._actions)
        def remove_parser_arg(argname):
            try:
                parser._remove_action(next(a for a in parser._actions if a.dest == argname))
            except StopIteration:
                pass
        remove_parser_arg('plot')
        remove_parser_arg('output')
        remove_parser_arg('disable_mpi')
        parser.add_argument('--np', type=int, default=np, 
                        help="The the number of processes to use for the simulation "
                             "(default: %(default)s)")
        parser.add_argument('--que_name', type=str, default=que_name, 
                            help="The the que to submit the job to (default: '%(default)s')")
        parser.add_argument('--output_dir', default=None, type=str, 
                            help="The parent directory in which the output directory will be created "
                                 "(default: $HOME/Output)")
        parser.add_argument('--max_memory', type=str, default=max_memory, 
                            help="The maximum memory allocated to run the network "
                                 "(default: '%(default)s')")
        parser.add_argument('--virtual_memory', type=str, default=virtual_memory, 
                            help="The average memory usage required by the program, decides when "
                                  "the scheduler is able to run the job (default: '%(default)s')")
        parser.add_argument('--jobname', type=str, default=None, 
                            help="Saves a file within the output directory with the descriptive name"
                                 " for easy renaming of the output directory after it is copied to "
                                 "its final destination, via the command "
                                 "'mv <output_dir> `cat <output_dir>/name`'")
        parser.add_argument('--dry_run', action='store_true', 
                            help="Performs a dry run without trying to  submit the job to the "
                                 "cluster for testing")
        parser.add_argument('--work_dir', type=str, default=None,
                            help="The work directory in which to run the simulation")
        return parser, original_actions
        
    def create_env(self, work_dir):
        """
        Creates a dictionary containing the appropriate environment variables
        
        @param work_dir: The work directory to set the envinroment variables for
        """
        env = os.environ.copy()
        new_path = ''
        if self.py_dir:
            new_path += os.path.join(self.py_dir, 'bin') + os.pathsep
        if self.mpi_dir:
            new_path += os.path.join(self.mpi_dir, 'bin') + os.pathsep
        if self.nrn_dir:
            new_path += os.path.join(self.nrn_dir, 'x86_64', 'bin') + os.pathsep
        if self.nest_dir:
            new_path += os.path.join(self.nest_dir, 'bin') + os.pathsep
        if self.sdials_dir:
            os.path.join(self.sdials_dir, 'bin') + os.pathsep
        env['PATH'] = new_path + env['PATH']
        new_pythonpath = (os.path.join(work_dir, 'src') + os.pathsep +
                          os.path.join(work_dir, 'depend') + os.pathsep)
        if self.nest_dir:
            new_pythonpath += (os.path.join(self.nest_dir, 'lib', 'python2.7', 'dist-packages') +
                               os.pathsep)
        env['PYTHONPATH'] = new_pythonpath + os.pathsep.join(sys.path)
        new_library_path = ''
        if self.mpi_dir:
            new_library_path += os.path.join(self.mpi_dir, 'lib') + os.pathsep
        if self.nest_dir:
            new_library_path += os.path.join(self.nest_dir, 'lib', 'nest') + os.pathsep
        if self.sdials_dir:
            new_library_path += os.path.join(self.sdials_dir, 'lib')
        env['NINEML_SRC_PATH'] = os.path.join(work_dir, 'src')
        return env        

#     def get_project_dir(self):
#         """
#         Returns the root directory of the project
#         """
#         # Root directory of the project code
#         return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 
    
    def create_work_dir(self, script_name, output_dir_parent=None):
        """
        Generates unique paths for the work and output directories, creating the work directory in the 
        process.
        
        `script_name`       -- The name of the script, used to name the directories appropriately
        `work_dir`          -- The name of the 
        `output_dir_parent` -- The name of the parent directory in which the output directory """ \
                               "will be created (defaults to $HOME/Output)." \
        "`required_dirs`    -- The sub-directories that need to be copied into the work directory"
        if not output_dir_parent:
            output_dir_parent = os.path.join(os.environ['HOME'], 'output')
        work_dir_parent = os.path.realpath(os.path.join(os.environ['HOME'], 'work'))
        if not os.path.exists(work_dir_parent):
            raise Exception("Symbolic link to work directory is missing from your home directory " \
                            "(i.e. $HOME/work). A symbolic link should be created that points to " \
                            "an appropriate directory in your units sub-directory of '/work' " \
                            "(i.e. ln -s /work/<unit-name>/<user-name> $HOME/work)")
        if not work_dir_parent.startswith('/work'):
            raise Exception("$HOME/work be a symbolic link to a sub-directory of the " \
                            "high-performance filesystem mounted at '/work' (typically "\
                            "/work/<unit-name>/<user-name>).")
        # Automatically generate paths
        # Unique time for distinguishing runs    
        time_str = time.strftime('%Y-%m-%d-%A_%H-%M-%S', time.localtime()) 
        # Working directory path
        work_dir = os.path.join(work_dir_parent, script_name + "." + time_str + ".1") 
        #Ensure that working directory is unique
        created_work_dir=False
        count = 1
        while not created_work_dir:
            try:
                created_work_dir = not os.makedirs(work_dir) 
            except IOError as e:
                count += 1
                if count > 1000:
                    print "Something has gone wrong, can't create directory '{}' after 1000 " \
                          "attempts".format(work_dir)
                    raise e
                # Replace old count at the end of work directory with new count
                work_dir = '.'.join(work_dir.split('.')[:-1] + [str(count)])
        # Make output directory for the generated files
        os.mkdir(os.path.join(work_dir, 'output'))      
        # Write time string to file for future reference
        with open(os.path.join(work_dir, 'output', 'time_stamp'), 'w') as f:
            f.write(time_str + '\n')
        # Determine the path for the output directory when it is copied to the output directory destination
        output_dir = os.path.join(output_dir_parent, os.path.split(work_dir)[1])
        return work_dir, output_dir
    
    def work_dir_init(self, work_dir, required_dirs=[], dependencies=[]):
        """
        Copies directories from the project directory to the work directory
        
        @param work_dir: The destination work directory
        @param required_dirs: The required sub-directories to be copied to the work directory
        """

        # Copy snapshot of selected subdirectories to working directory
        for directory in required_dirs:
            print "Copying '{}' sub-directory to work directory".format(directory)
            shutil.copytree(directory, os.path.join(work_dir,directory), symlinks=True)
        if dependencies:
            dependency_dir = os.path.join(work_dir, 'depend') 
            os.mkdir(dependency_dir)
            for from_, to_ in dependencies:
#                 # If not an absolute path prepend the project directory as a relative path (useful for
#                 # getting dependencies from directories installed alongside the project directory
#                 if not from_.startswith('/'):
#                     from_ = get_project_dir() + os.path.sep + from_
                shutil.copytree(from_, os.path.join(dependency_dir, to_))
                
    def create_cmdline(self, script_name, script_args, work_dir, args):
        cmdline = 'time mpirun python {}.py'.format(script_name)
        options=' --output {}/output/'.format(work_dir)
        for arg in script_args:
            name = arg.dest
            if hasattr(args, name):
                val = getattr(args, name)
                if arg.required:
                        cmdline += ' {}'.format(val)
                else:
                    if val is not False:
                        options += ' --{}'.format(name)
                        if val is not True:
                            options += ' {}'.format(val)
        cmdline += options
        return cmdline
        
    def submit(self, script_name, cmds, work_dir, output_dir, args, que_name='longP', 
                   max_memory='4g', virtual_memory='3g', time_limit=None, env=None, 
                   copy_to_output=['xml'], strip_build_from_copy=True, name=None):
        """
        Create a jobscript in the work directory and then submit it to the tombo que
        
        @param script_name: The name of the script (used to give a meaningful name to the job)
        @param cmds: The commands to run on the cluster
        @param np: The number of processors to request for the job
        @param work_dir: The working directory to run the script from
        @param output_dir: The output directory to copy the results to
        @param time_limit [str]: The hard time limit for the job in "HH:MM:SS" format (increases the preference assigned by the scheduler)
        @param env: The required environment variables (defaults to those generated by 'create_env(work_dir)')    
        @param copy_to_output: Directories to copy into the output directory
        @param strip_build_from_copy: Removes all files and directories to be copied that have the name 'build'
        @param name: Records a name for the run (when generating multiple runs) so that the output directory can be easily renamed to a more informative name after it is copied to its destination via the command "mv <output_dir> `cat <output_dir>/name`"
        """
        if not env:
            env = self.create_env(work_dir)
        else:
            env = copy(env)
        copy_cmd = ''
        for to_copy in copy_to_output:
            origin = work_dir + os.path.sep + to_copy
            if strip_build_from_copy:
                copy_cmd+='find {origin} -name build -exec rm -r {{}} \; 2>/dev/null\n'.\
                          format(origin=origin)
            destination = output_dir + os.path.sep + to_copy
            base_dir = os.path.dirname(destination[:-1] if destination.endswith('/') else destination)
            copy_cmd += (
"""
mkdir -p {base_dir}
cp -r {origin} {destination}
"""
                         .format(base_dir=base_dir, origin=origin, destination=destination))
        #Create jobscript
        if time_limit:
            if type(time_limit) != str or len(time_limit.split(':')) != 3:
                raise Exception("Poorly formatted time limit string '{}' passed to submit job"
                                .format(time_limit))
            time_limit_option= "\n# Set the maximum run time\n#$ -l h_rt {}\n".format(time_limit)
        else:
            time_limit_option=''
        if name:
            name_cmd = "echo '{name}' > {output_dir}/name"
        else:
            name_cmd = ""
        jobscript_path = os.path.join(work_dir, script_name + '.job')
        f = open(jobscript_path, 'w')
        f.write(
"""#!/usr/bin/env sh
    
# Parse the job script using this shell
#$ -S /bin/bash

# Send stdout and stderr to the same file.
#$ -j y

# Standard output and standard error files
#$ -o {work_dir}/output_stream
#$ -e {work_dir}/output_stream

# Name of the queue
#$ -q {que_name}

# use OpenMPI parallel environment with {np} processes
#$ -pe openmpi {np}
{time_limit}
# Set the memory limits for the script
#$ -l h_vmem={max_mem}
#$ -l virtual_free={virt_mem}

# Export the following env variables:
#$ -v HOME
#$ -v PATH
#$ -v PYTHONPATH
#$ -v LD_LIBRARY_PATH
#$ -v NINEMLP_SRC_PATH
#$ -v NINEMLP_MPI
#$ -v BREP_DEVEL
#$ -v PARAMDIR
#$ -v VERBOSE

###################################################
### Copy the model to all machines we are using ###
###################################################

# Set up the correct paths 
export PATH={path}:$PATH
export PYTHONPATH={pythonpath}
export LD_LIBRARY_PATH={ld_library_path}
export NINEMLP_SRC_PATH={nine_src_path}
export NINEMLP_MPI=1

echo "============== Starting mpirun ===============" 

cd {work_dir}
{cmds}

echo "============== Mpirun has ended =============="

echo "Copying files to output directory '{output_dir}'"
cp -r {work_dir}/output {output_dir}
cp {jobscript_path} {output_dir}/job
cp {work_dir}/output_stream {output_dir}/output
{name_cmd}
{copy_cmd}

echo "============== Done ===============" 
"""
        .format(work_dir=work_dir, path=env['PATH'], pythonpath=env['PYTHONPATH'],
          ld_library_path=env['LD_LIBRARY_PATH'], nine_src_path=os.path.join(work_dir,'src'), np=args.np,
          que_name=que_name, max_mem=max_memory, virt_mem=virtual_memory, cmds=cmds, 
          output_dir=output_dir, name_cmd=name_cmd, copy_cmd=copy_cmd, jobscript_path=jobscript_path, 
          time_limit=time_limit_option))
        f.close()
        # Submit job
        print "Submitting job '%s' to que" % jobscript_path
        subprocess.check_call('qsub %s' % jobscript_path, shell=True)
        print "Your job '%s' has been submitted" % jobscript_path
        print "The output stream can be viewed by:"
        print "less " + os.path.join(work_dir, 'output_stream')
        print ("Once completed the output files (including the output stream and job script) of "
               "this job will be copied to:")
        print output_dir
        