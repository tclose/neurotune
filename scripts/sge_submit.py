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
# Try to import argument parser from script in order to add those arguments to the SGE-related
# arguments
try:
    exec("from {} import parser as script_parser".format(script_name))
    parser, script_args = submitter.add_sge_arguments(script_parser)  # @UndefinedVariable: script_parser
except ImportError:
    print "Warning: failed to import 'parser' from '{}'".format(script_name)
    script_parser = None
    parser, script_args = submitter.add_sge_arguments(argparse.ArgumentParser())
# Try to import 'src_dir_init' method from script otherwise fail gracefully
try:
    exec("from {} import src_dir_init".format(script_name))
except ImportError:
    print "Warning: failed to import 'src_dir_init' from '{}'".format(script_name)
    src_dir_init = None
# Parse arguments that were supplied to script
args = parser.parse_args(sys.argv[2:])
# Create work dir on 
work_dir, output_dir = submitter.create_work_dir(script_name)
# Create command line to be run in job script from parsed arguments
cmdline = submitter.create_cmdline(script_name, script_args, work_dir, args)
# Initialise work directory
submitter.work_dir_init(work_dir)
# Copy and 
if src_dir_init is not None:
    src_dir_init(os.path.join(work_dir, 'src'), args)
# Submit script to scheduler
submitter.submit(script_name, cmdline, work_dir, output_dir, args, dry_run=True)
