from shutil import copy, copytree, ignore_patterns
import os
import shutil
import pandas as pd

from subprocess import call


def run_octave(orig_case_dir, case_output_dir):
	#go into case directory and run octave script
	os.chdir(orig_case_dir)
	call(['octave', 'plotFarWake.m'])

	#return to home directory and copy produced csv files into data directory	
	os.chdir(base_path)
		
	#print(case_output_dir)
	if os.path.exists(case_output_dir):
		shutil.rmtree(case_output_dir)
	copytree(orig_case_dir, case_output_dir, ignore=ignore_patterns('*.m'))


#function to downsample data to be 128x256
def downsample(dire, component_file):
	
	#recombine file with directory path 
	component_filepath = os.path.join(dire, component_file)
	print(component_filepath)
	results = pd.read_csv(component_filepath, header=None)
	component_name = os.path.splitext(component_filepath)
	print(component_name)

	results_down = results.iloc[:, ::8] #get every nth value 
	results_down.drop(results_down.columns[-9:], axis=1, inplace=True) #drop last 9 columns to get even 256
	print(results_down)
	results_down.to_csv('%s_down.csv' % component_name[0])
	


output_dir = 'data'

#get directory names
case_dirs = [f for f in os.listdir(os.path.join(os.getcwd(),'openfoamruns')) if f.startswith('dTdz')]

print(case_dirs)

#same home directory path
base_path = os.getcwd()

for case in case_dirs:
	
	orig_case_dir = os.path.join('openfoamruns', case, 'scripts')
	case_output_dir = os.path.join(output_dir, case)	
	
	#comment this out to not run octave scripts	
	#run_octave(orig_case_dir, case_output_dir)	

	#iterate through each file in output dir and downsample
	for f in os.listdir(case_output_dir):
		if not f.endswith('_down.csv'):
			downsample(case_output_dir, f)

