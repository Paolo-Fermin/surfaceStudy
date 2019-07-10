from os import path
from os import getcwd
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Error import error

from PyFoam.Execution.BasicRunner import BasicRunner

from PyFoam.Applications.Decomposer import Decomposer
from PyFoam.Applications.CaseReport import CaseReport
from PyFoam.Execution.ParallelExecution import LAMMachine
num_procs = 4

import csv

temps = [0.01, 0.005, 0.001]

depths = [-30.0, -60.0, -90.0]

base_case = 'basecase'

dire = SolutionDirectory(base_case, archive=None)
dire.addToClone('0') 	#make sure initial timestep directory is copied into clones

for temp in temps:
	for depth in depths:
		
		#clone base case
		clone_name = 'dTdz%0.3f_z%d' % (temp, depth)
		clone = dire.cloneCase(clone_name)
	
		#read parameter file and change parameter
		param_file = ParsedParameterFile(path.join(clone_name,'constant','parameterFile'))
		param_file['dTdz'] = 'dTdz [0 -1 0 0 0 0 0] %0.3f' % temp
		param_file['z0'] = 'z0 [0 1 0 0 0 0 0] %.1f' % depth
		param_file.writeFile()

		#set initial fields
		run_initFields = BasicRunner(argv=['setInitialFields', '-case', clone.name], logname='setInitialFields')
		run_initFields.start()
		print('initial fields set')

		#implement parallelization
		print('Decomposing...')
		Decomposer(args=['--progress', clone.name, num_procs])
		CaseReport(args=['--decomposition', clone.name])
		machine = LAMMachine(nr=num_procs)

		#run solver
		print('Running solver...')
		run_solver = BasicRunner(argv=['trainingSolver', '-case', clone.name], logname='trainingSolver', lam=machine)
		run_solver.start()
		if not run_solver.runOK():
			error('There was a problem with trainingSolver')
		print('Finished running solver')
		
		#run postprocessing
		run_postprocess = BasicRunner(argv=['postProcess', '-case', clone.name, '-func', 'sample'], logname='postProcessLog', lam=machine)
		run_postprocess.start()
		if not run_postprocess.runOK():
			errror('There was a problem running postprocessing')

		
		

