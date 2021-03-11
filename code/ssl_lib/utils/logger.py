# A simple torch style logger
# (C) Wei YANG 2017
from __future__ import absolute_import
import os
import sys
import numpy as np
import json

__all__ = ['Logger', 'LoggerMonitor', 'init_trial_path']

    

def init_trial_path(args):
	"""Initialize the path for a hyperparameter setting
	"""
	args.result_dir = os.path.join(args.out_dir, args.task_name)
	os.makedirs(args.result_dir, exist_ok=True)
	trial_id = 0
	path_exists = True
	while path_exists:
		trial_id += 1
		path_to_results = args.result_dir + '/{:d}'.format(trial_id)
		path_exists = os.path.exists(path_to_results)
	args.save_path = path_to_results
	os.makedirs(args.save_path, exist_ok=True)
	with open(os.path.join(args.save_path, 'args.json'), 'w') as f:
		json.dump(args.__dict__, f)
	return args

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers, verbose=True):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()
        if verbose:
            self.print()

    def print(self):
        log_str = ""
        for name, num in self.numbers.items():
            log_str += f"{name}: {num[-1]}, "
        print(log_str)


    def close(self):
        if self.file is not None:
            self.file.close()

class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)
