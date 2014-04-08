#!/usr/bin/env python
"""
Wraps a executable to enable that script to be submitted to an SGE cluster engine
"""
import sys
import argparse
from neurotune.tuner.mpi import SGESubmitter
script_name = sys.argv[1]
# Most definitely overkill but just check to see that no malicious strings are going to be passed to 
# exec function
if ';' in script_name:
    raise Exception("Malformed script name")
# Create submitter object
submitter = SGESubmitter()
try:
    exec("from {} import argparser as script_parser, compile_model".format(script_name))
    parser, script_args = submitter.add_sge_arguments(script_parser)  # @UndefinedVariable: script_parser
except ImportError:
    script_parser = compile_model = None
    parser, script_args = submitter.add_sge_arguments(argparse.ArgumentParser())
# Parse arguments that were supplied to script
args = parser.parse_args(sys.argv[2:])
# Create work dir on 
work_dir, output_dir = submitter.create_work_dir(script_name)
# Create command line to be run in job script from parsed arguments
cmdline = submitter.create_cmdline(script_name, script_args, work_dir, args)
# Initialise work directory
submitter.work_dir_init(work_dir)
if compile_model is not None:
    compile_model(args)
# Submit script to scheduler
submitter.submit(script_name, cmdline, work_dir, output_dir, args)
