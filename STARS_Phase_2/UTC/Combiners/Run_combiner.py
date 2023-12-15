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
		self.start_line = None
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


						
	def create_bigdf_new(self,files_in):
		# Initialize an empty dictionary to store the dataframes

		csv_files = files_in
		
		self.bigdf = pd.DataFrame()
		# Loop through each CSV file and convert it to a pandas dataframe
		for file in csv_files:
			
			
			# Then read from the next rows without joining the data
			df = pd.read_csv(file,skiprows=1)


			self.bigdf = pd.concat([self.bigdf,df])
		#sort values incase something weird happens
		self.bigdf = self.bigdf.sort_values(by=['sec_incr']).reset_index(drop=True)
		print(self.bigdf.shape)

		if self.huge_df.empty == True:
			self.huge_df = pd.concat([self.huge_df,self.bigdf])
		else:
			### add on the cycles
			cycle_last = self.huge_df['cycle'].iloc[-1]
			self.bigdf['cycle'] = self.bigdf['cycle'] + cycle_last
			self.huge_df = pd.concat([self.huge_df,self.bigdf])
		


	def create_bigdf_old(self,files_in):
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

		bigdf = bigdf.sort_values(by=['Time (Seconds)'])
		### Keep these loops separate to not mess up work flow

		if self.huge_df.empty == True:
			self.huge_df = pd.concat([self.huge_df,bigdf])
		else:
			### add on the cycles
			cycle_last = self.huge_df['cycle'].iloc[-1]
			bigdf['cycle'] = bigdf['cycle'] + cycle_last
			self.huge_df = pd.concat([self.huge_df,bigdf])


	def save_df_to_parquet(self,title):
		df_in = self.huge_df
		df_in.to_parquet(title + '_Reliability.parquet',engine='pyarrow')
		print(df_in['cycle'].iloc[-10:-1])
		return
	
		

def main():

	print('\n Move all the two different runs folder into the same directory of this script, should look like this: \n')
	print(' Run1 \n Run2 \n Run_combiner.py \n \n')
	h = HEAT_Analysis()
	while True:
		prompt1 = input('Type a unique subset of the first folder name and this will go grab the files based on that \n Only needs to be a few characters \n')
		run1_in = '**' + prompt1 + '**\\**.csv'
		run1_files = glob.glob(run1_in,recursive = True)
		print('\n')
		print(run1_files)
		print('\n')
		prompt2 = input('Are these the files from the first run? (y) or (n)')
		if prompt2 == 'y':
			h.create_bigdf_new(run1_files)
			delim_name = run1_files[0]
			delim_file = delim_name.split('_')
			title = delim_file[-3]
			break
		elif prompt2 == 'n':
			print('Try a different input stirng! \n')
		else:
			print('Type in an acceptable answer \n')


	while True:
		prompt1 = input('Type a unique subset of the next folder name and this will go grab the files based on that \n Only needs to be a few characters \n')
		runN_in = '**' + prompt1 + '**\\**.csv'
		runN_files = glob.glob(runN_in,recursive = True)
		print('\n')
		print(runN_files)
		print('\n')
		prompt2 = input('Are these the files from the next run? (y) or (n)')
		if prompt2 == 'y':
			prompt3 = input('Do you need to append any more files? (y) or (n)')
			if prompt3 == 'y':
				h.create_bigdf_new(runN_files)
				continue
			elif prompt3 == 'n':
				h.create_bigdf_new(runN_files)
				break
		elif prompt2 == 'n':
			print('Try a different input stirng! \n')
		else:
			print('Type in an acceptable answer \n')

	h.save_df_to_parquet(title)
	print('\n Move this parquet and one of the MetaData files then run the HEAT_Parquet_Analysis.py script')
	sys.exit()


if __name__ == '__main__':
	main()