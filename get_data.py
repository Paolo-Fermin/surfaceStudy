from shutil import copy, copytree, ignore_patterns
import os
from os import path
from subprocess import call

output_dir = 'data'

#get directory names
case_dirs = [f for f in os.listdir(os.getcwd()) if f.startswith('dTdz')]

print(case_dirs)

base_path = os.getcwd()


for case in case_dirs:
	os.chdir(path.join(case, 'scripts'))
	call(['octave', 'plotFarWake.m'])
	os.chdir(base_path)
	copytree(path.join(case, 'scripts'), path.join(output_dir, case), ignore=ignore_patterns('*.m'))

	

