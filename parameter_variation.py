from datetime import datetime

start_time = datetime.now()

from os import path
from os import getcwd
from os import getpid
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.Error import error

from PyFoam.Execution.BasicRunner import BasicRunner

from PyFoam.Applications.Decomposer import Decomposer
from PyFoam.Applications.CaseReport import CaseReport
from PyFoam.Execution.ParallelExecution import LAMMachine
num_procs = 4

import csv

temps = [0.005, 0.001, 0.010]
depths = [-45, -75]
test_cases = []
for temp in temps:
	for depth in depths:
		test_cases.append([temp, depth])

#override previous instantiation 
test_cases = [
	[0.010, -45],
	[0.001, -75], 
	[0.010, -75], 
	[0.008, -30],
	[0.008, -60],
	[0.008, -90]
]
	

copy_dir = 'openfoamruns'
base_case = 'basecase'
if not path.exists(path.join(base_case,'0')):
	import shutil
	print('Copying 0 directory')
	shutil.copytree(path.join(base_case, '0.org'), path.join(base_case, '0'))
	print('Copied 0 directory')	

dire = SolutionDirectory(base_case, archive=None)
dire.addToClone('0') 	#make sure initial timestep directory is copied into clones
dire.addToClone('scripts')

for case in test_cases:

	temp = case[0]
	depth = case[1]

	#clone base case
	clone_name = '/%s/dTdz%0.3f_z%d' % (path.join(getcwd(), copy_dir), temp, depth)
	clone = dire.cloneCase(clone_name)

	#read parameter file and change parameter
	param_file = ParsedParameterFile(path.join(clone_name,'constant','parameterFile'))
	param_file['dTdz'] = 'dTdz [0 -1 0 0 0 0 0] %0.3f' % temp
	param_file['z0'] = 'z0 [0 1 0 0 0 0 0] %.1f' % depth
	param_file.writeFile()

	#set initial fields
	run_initFields = BasicRunner(argv=['setInitialFields', '-case', clone_name], logname='setInitialFields')
	run_initFields.start()
	print('initial fields set')

	#implement parallelization
	print('Decomposing...')
	Decomposer(args=['--progress', clone_name, num_procs])
	CaseReport(args=['--decomposition', clone_name])
	machine = LAMMachine(nr=num_procs)

	#run solver
	print('Running solver...')
	print('PID: ' + str(getpid()))
	run_solver = BasicRunner(argv=['trainingSolver', '-case', clone_name], logname='trainingSolver', lam=machine)
	run_solver.start()
	if not run_solver.runOK():
		error('There was a problem with trainingSolver')
	print('Finished running solver')
	
	#run postprocessing
	run_postprocess = BasicRunner(argv=['postProcess', '-case', clone_name, '-func', 'sample'], logname='postProcessLog', lam=machine)
	run_postprocess.start()
	if not run_postprocess.runOK():
		errror('There was a problem running postprocessing')
	

		
print('Execution time: ' + str(datetime.now() - start_time))
