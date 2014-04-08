#!/usr/bin/env python
"""
Wraps a executable to enable that script to be submitted to an SGE cluster engine
"""
import sys
import os.path
import argparse
from neurotune.tuner.mpi import SGESubmitter
script_name = sys.argv[1]
# Create submitter object
submitter = SGESubmitter()
# Import the script as a module
script = None # Actually set by the following 'exec' statement but initialised here to squash PyLint
if os.path.dirname(script_name):
    sys.path.append(os.path.dirname(script_name))
exec("import {} as script".format(os.path.splitext(os.path.basename(script_name))[0]))
# Place empty versions of parser and src_dir_init if they are not provided by script
if not hasattr(script, 'parser'):  
    script.parser = argparse.ArgumentParser() 
if not hasattr(script, 'src_dir_init'): 
    def dummy_func(src_dir, args):
        pass
    script.src_dir_init = dummy_func
parser, script_args = submitter.add_sge_arguments(script.parser)  # @UndefinedVariable: script
# Try to import 'src_dir_init' method from script otherwise fail gracefully
# Parse arguments that were supplied to script
args = parser.parse_args(sys.argv[2:])
# Create work dir on 
work_dir, output_dir = submitter.create_work_dir(script_name)
# Create command line to be run in job script from parsed arguments
cmdline = submitter.create_cmdline(script_name, script_args, work_dir, args)
# Initialise work directory
submitter.work_dir_init(work_dir)
# Copy and 
script.src_dir_init(os.path.join(work_dir, 'src'), args)
# Submit script to scheduler
submitter.submit(script_name, cmdline, work_dir, output_dir, args, dry_run=True)
