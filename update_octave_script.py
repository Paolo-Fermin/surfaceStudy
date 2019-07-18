import os
import shutil

from subprocess import call

base_case_dir = os.path.join('basecase', 'scripts')

case_dirs = [f for f in os.listdir(os.path.join(os.getcwd(), 'openfoamruns')) if f.startswith('dTdz')]

for case in case_dirs:
	
	shutil.copyfile(os.path.join(base_case_dir, 'plotFarWake.m'), os.path.join('openfoamruns', case, 'scripts', 'plotFarWake.m'))	
