import glob
import pandas as pd
import numpy as np
import io
import math
import matplotlib.pyplot as plt
import multiprocessing
import os
from datetime import datetime
import sys
import shutil


class HEAT_Analysis():
	def __init__(self):
		now = datetime.now()
		self.timestamp = now.strftime("_%Y_%m_%d_%H_%M")
		self.mini_timestamp = now.strftime("%Y_%m_%d")
		self.indicator = None
		self.meta_dict = None
		self.file_list = None
		self.title = None
		self.dirs = None
		self.start_line = None
		self.instrument = None
		self.meta_list = ['device_id','daq_limit_cycles','Length - Preloaded','Displacement per Cycle'] ## Might need to switch in 'Device ID'
		self.master_scatter = None 
		self.bend_list = ['05','06','07']   ### NEED
		self.instronita_list = ['01','02','03','04']
		self.master_path = None
		self.master_df = None
		self.limit_name = None
		self.limit_df = None
		self.huge_df = pd.DataFrame()

	def find_first_row(self,files_in):
		### This function finds the start of the data, you want to run this separate to not get an IOpub error
		search_word = 'Time ('
		if files_in is None:
			print('No csv files found')
		else:
			with open(files_in[0] , 'r') as f:
				lines = f.readlines()
				j = 0
				for line in lines:
					if search_word in line:
						self.start_line = lines.index(line)
						
					if self.start_line is not None:
						break
						
					j = j+1
					if j > 150:
						break

	def create_bigdf(self,files_in):
		# Initialize an empty dictionary to store the dataframes

		csv_files = files_in
		
		bigdf = pd.DataFrame()
		# Loop through each CSV file and convert it to a pandas dataframe
		for file in csv_files:
			
			 ### Open and delete first 4 rows
			with open(file , 'r') as f:
				data = f.readlines()[self.start_line:]
			
			# Then read from the next rows without joining the data
			df = pd.read_csv(io.StringIO('\n'.join(data)))


			bigdf = pd.concat([bigdf,df])
			### At this point you have the raw data and its named, time to move it

		### Keep these loops separate to not mess up work flow
		self.huge_df = pd.concat([self.huge_df,bigdf])

	def save_df_to_parquet(self,title):
		df_in = self.huge_df
		df_in.to_parquet(title + '_Reliability.parquet',engine='pyarrow')
	
		

def main():

	print('\n Move all the two different runs folder into the same directory of this script, should look like this: \n')
	print('Run1 \n Run2 \n Run_combiner.py')

	while True:
		prompt1 = input('Type a unique subset of the first folder name and this will go grab the files based on that \n Only needs to be a few characters')
		run1_in = '**' + prompt1 + '**\\**.csv'
		run1_files = glob.glob(glob_string,recursive = True)
		print('\n')
		print(run1_files)
		print('\n')
		prompt2 = input('Are these the files from the first run? (y) or (n)')
		if prompt2 == 'y':
			create_bigdf(run1_files)
			title = run1_files[0][0:13]
			break
		elif prompt2 == 'n':
			print('Try a different input stirng! \n')
		else:
			print('Type in an acceptable answer \n')


	while True:
		prompt1 = input('Type a unique subset of the next folder name and this will go grab the files based on that \n Only needs to be a few characters')
		runN_in = '**' + prompt1 + '**\\**.csv'
		runN_files = glob.glob(glob_string,recursive = True)
		print('\n')
		print(run1_files)
		print('\n')
		prompt2 = input('Are these the files from the next run? (y) or (n)')
		if prompt2 == 'y':
			prompt3 = input('Do you need to append any more files? (y) or (n)')
			if prompt3 == 'y':
				create_bigdf(runN_files)
				continue
			elif prompt3 == 'n':
				create_bigdf(runN_files)
				break
		elif prompt2 == 'n':
			print('Try a different input stirng! \n')
		else:
			print('Type in an acceptable answer \n')

	save_df_to_parquet()
	print('\n Move this parquet and one of the MetaData files then run the HEAT_Parquet_Analysis.py script')

