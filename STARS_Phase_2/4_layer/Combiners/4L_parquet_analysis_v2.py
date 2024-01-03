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
import re


def lim30for10(df):
	cycle_fail = None
	in_last = None
	for index,row in df.iterrows():
		if in_last == None: ## Establish the index and cycle last
			in_last = index
			cyc_start = row['cycle']

		track = index - in_last

		if track > 1.1:                    # See if we pull data below limit
			cyc_start = row['cycle']

		cyc_now = row['cycle']             #Grab Current cycle
		cyc_diff = cyc_now - cyc_start

		if cyc_diff >= 10:                 #Break if you the data has remained above for 10 cycles
			cycle_fail = cyc_now
			break

		in_last = index
	
	if cycle_fail == None:
		cycle_fail = 'Did not reach limit'

	return cycle_fail

def lim10for5samp(df):
	cycle_fail = None
	in_last = None
	for index,row in df.iterrows():
		if in_last == None: ## Establish the index and cycle last
			in_last = index
			in_start = index
	
		track = index - in_last
	
		if track > 1.1:                    # See if we pull data below limit
			in_start = index
	
		in_now = index            #Grab Current index
		in_diff = in_now - in_start
	
		if in_diff >= 5:                 #Break if you the data has remained above for 10 cycles
			cycle_fail = row['cycle']
			break
	
		in_last = index
	
	if cycle_fail == None:
		cycle_fail = 'Did not reach limit'
	
	return cycle_fail

class HEAT_Analysis():
	def __init__(self):
		now = datetime.now()
		self.timestamp = now.strftime("%Y_%m_%d_%H_%M")
		self.mini_timestamp = now.strftime("%Y_%m_%d")
		self.pretty_timestamp = now.strftime("%m/%d/%Y")
		self.indicator = None
		self.meta_dict = None
		self.file_list = None
		self.title = None
		self.dirs = None
		self.start_line = None
		self.instrument = None
		self.meta_list = ['device_id','daq_limit_cycles','Length - Preloaded','Displacement per Cycle'] ## Might need to switch in 'Device ID'
		self.master_scatter = None 
		self.bend_list = ['05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']   ### NEED
		self.instronita_list = ['01','02','03','04']
		self.master_path = None
		self.master_df = None
		self.limit_name = None
		self.limit_df = None
		self.channel_list = None
		self.dfs = None
		self.names = None
		self.daq_list = None


	def assign_channels(self,intake_channel_list):
		self.channel_list = intake_channel_list 


	def glob_search_csv(self):
		glob_string = '**.csv' # this generic one will just find all csv's 
		self.file_list = glob.glob(glob_string,recursive = True)
		
		delim_name = self.file_list[0]
		delim_file = delim_name.split('_')
		#self.title = delim_file[-3]
		self.title = delim_file[-6] + '-' +delim_file[-5] + '-' + delim_file[-4] +'-'+ delim_file[-3]

		self.dirs = self.title +'\\'
		if os.path.exists(self.dirs):
			pass
		else:
			os.makedirs(self.dirs)

		print('\n')
		print(self.title)
		print('\n')
		glob_prompt = input('\n (y) to continue, (n) to exit and go use a different script \n')
		if glob_prompt == 'y':
			pass
		else:
			print(' \n Goodbye! \n')
			sys.exit()

		return self.title

	def find_first_row(self):
		### This function finds the start of the data, you want to run this separate to not get an IOpub error
		search_word = 'Time ('
		if self.file_list is None:
			print('No csv files found')
		else:
			with open(self.file_list[0] , 'r') as f:
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

	def create_bigdf_old(self):
		# Initialize an empty dictionary to store the dataframes

		csv_files = self.file_list
		
		self.bigdf = pd.DataFrame()
		# Loop through each CSV file and convert it to a pandas dataframe
		for file in csv_files:
			
			 ### Open and delete first 4 rows
			with open(file , 'r') as f:
				data = f.readlines()[self.start_line:]
			
			# Then read from the next rows without joining the data
			df = pd.read_csv(io.StringIO('\n'.join(data)))


			self.bigdf = pd.concat([self.bigdf,df])
		#sort values incase something weird happens
		self.bigdf = self.bigdf.sort_values(by=['Time (Seconds)']).reset_index(drop=True)

			### At this point you have the raw data and its named, time to move it

		### Keep these loops separate to not mess up work flow
	
		raw_dirs = self.dirs + 'Raw\\'
		if os.path.exists(raw_dirs):
			pass
		else:
			os.makedirs(raw_dirs)


		# for files in csv_files:
		# 	shutil.move(files,raw_dirs + files)

	def create_bigdf_new(self):
		# Initialize an empty dictionary to store the dataframes

		csv_files = self.file_list
		
		self.bigdf = pd.DataFrame()
		# Loop through each CSV file and convert it to a pandas dataframe
		for file in csv_files:
			
			
			# Then read from the next rows without joining the data
			df = pd.read_csv(file,skiprows=1)


			self.bigdf = pd.concat([self.bigdf,df])
		#sort values incase something weird happens
		self.bigdf = self.bigdf.sort_values(by=['sec_incr']).reset_index(drop=True)
		print(self.bigdf.shape)
		

		### At this point you have the raw data and its named, time to move it

		### Keep these loops separate to not mess up work flow 

		### Wayyy too much raw data, just moving parquets now
	
		# raw_dirs = self.dirs + 'Raw\\'
		# if os.path.exists(raw_dirs):
		# 	pass
		# else:
		# 	os.makedirs(raw_dirs)


		# for files in csv_files:
		# 	shutil.move(files,raw_dirs + files)



	def create_limitdf(self,rod_diameter,maker,encapsulation,daq_style,back,shape):

		limit_columns = ['serial','date','manufacturer','encapsulation','backing','shape','trace','physical_position','strain_p','Start_ohms','10_p_increase_cycles','30_p_increase_for_10_cycles']
		limit_df=pd.DataFrame([],columns=limit_columns)


		title = self.title
		date = self.timestamp
		
		


		if daq_style =='8':
			daq_list = ['DAQ1','DAQ2','DAQ3','DAQ4','DAQ5','DAQ6','DAQ7','DAQ8']
			sample_list = ['hg_s_l3_l','sg_s_l3_l','hg_s_l2_l','sg_s_l2_l','hg_c_l3_l','sg_c_l3_l','hg_c_l2_l','sg_c_l2_l']
			cycle_list = ['cycle']
		elif daq_style == '16':
			daq_list = ['daq_ch1', 'daq_ch2', 'daq_ch3', 'daq_ch4', 'daq_ch5', 'daq_ch6', 'daq_ch7', 'daq_ch8', 'daq_ch9', 'daq_ch10', 'daq_ch11', 'daq_ch12', 'daq_ch13', 'daq_ch14', 'daq_ch15', 'daq_ch16']
			sample_list = ['hg_s_l3_l','hg_s_l3_r','sg_s_l3_l','sg_s_l3_r','hg_s_l2_l','hg_s_l2_r','sg_s_l2_l','sg_s_l2_r','hg_c_l3_l','hg_c_l3_r','sg_c_l3_l','sg_c_l3_r','hg_c_l2_l','hg_c_l2_r','sg_c_l2_l','sg_c_l2_r']
			cycle_list = ['cycle']
		else:
			print('daq number not reqcognized')
		

		bigdf = self.bigdf
		

		#shrink the df
		small_list = cycle_list + daq_list
		smoldf = bigdf[small_list].copy()
		bigdf = None


		j=0
		for i in daq_list:
			# Create a rolling average
			res_raw = smoldf[i]
			
			if res_raw.iloc[0] < 0.0 or res_raw.iloc[4]>110:
				j = j + 1
				continue

			if isinstance(rod_diameter,str):
				strain = rod_diameter + 'mm_bend'
			else:
				strain = str(rod_diameter) + 'mm_bend'
			
			res_start = smoldf[i].iloc[0]

			p30up = smoldf[i].iloc[0] * 1.3
			p30_df = smoldf[smoldf[i] > p30up]
			p30for10_lim = lim30for10(p30_df)	

			res10p_lim = res_start + res_start * 0.1
			
				
			compare10p = smoldf[smoldf[i] > res10p_lim].reset_index(drop=True)
			if len(compare10p) < 5:
				cycle_res10p = 'Did not reach limit'
			else:			
				cycle_res10p = compare10p['cycle'].iloc[4]
			compare10p = None
				
			
			row = pd.DataFrame([[title,date,maker,encapsulation,back,shape,sample_list[j],daq_list[j],strain,res_start,p10for5_lim,p30for10_lim]],columns=limit_columns )
			limit_df = pd.concat([limit_df,row]) #limit_df.append(row)
			j = j + 1

		limit_df = limit_df.reset_index(drop=True)


		trace_df = limit_df['trace'].to_frame().reset_index(drop=True)
		expanded_df = trace_df['trace'].str.split('_',expand=True)
		expanded_df = expanded_df.rename(columns={0:'ground',1:'co-planar',2:'layer',3:'loop_postion'})


		def ground_tranform(row):
			if row['ground'] == 'hg':
				return 'hatched ground'
			else:
				return 'solid ground'

		expanded_df['ground'] = expanded_df.apply(ground_tranform, axis=1)

		def planar_tranform(row):
			if row['co-planar'] == 's':
				return 'stripline'
			else:
				return 'co-planar'

		expanded_df['co-planar'] = expanded_df.apply(planar_tranform, axis=1)

		def layer_tranform(row):
			if row['layer'] == 'l3':
				return 'layer 3'
			else:
				return 'layer 2'

		expanded_df['layer'] = expanded_df.apply(layer_tranform, axis=1)

		def loop_tranform(row):
			if row['loop_postion'] == 'l':
				return 'left'
			else:
				return 'right'

		expanded_df['loop_position'] = expanded_df.apply(loop_tranform, axis=1)

		
		limit_df = pd.concat([limit_df,expanded_df],axis=1)



		# save Limit df
		limit_name =self.title + '_limits.csv'
		limit_df.to_csv(self.dirs + limit_name,index=False)
		
		self.limit_name = limit_name
		self.daq_list = daq_list
		
		self.limit_df = limit_df

	def append_limit_df_to_master(self):
		### Find master csv
		npre = 'N:\\4L_test_data\\master\\'
		masname = '4L_Bend_Master.csv'
		oldname = npre + 'old\\' +  self.mini_timestamp +'_' +  masname

		self.master_path = npre


		current = self.master_path + masname
		mas_to_folder = self.dirs + masname

		master_df_old = pd.read_csv(current)

		master_df_old.to_csv(oldname,index=False)

		self.master_df = pd.concat([master_df_old,self.limit_df]).reset_index(drop=True)

		self.master_df.to_csv(current,index=False)
		#Save to current dirs
		self.master_df.to_csv(mas_to_folder,index=False)

		print('\n Appending new data to \n' + current + '\n')

	def plot_bigdf_moving_average(self):
		bigdf = self.bigdf
		mov_avg = int(np.round(len(bigdf)/1000,0))
		if mov_avg < 3:
			mov_avg = 3
		graph_title = self.title + ': '
		daq_list = self.daq_list
			
	 


		### Moving Average Plot
		plt.rcParams['agg.path.chunksize'] = 10000
		fig, (ax1,ax2) = plt.subplots(2,figsize=(30,20))
		j = 0
		print(self.channel_list)
		for i in daq_list:


			# Create a rolling average
			res_raw = bigdf[i]
			res_avg = res_raw.rolling(window=mov_avg).mean()
			cycle_count = bigdf['cycle']
			if res_raw.iloc[0] < 0.0 or res_raw.iloc[4]>110:
				j=j+1 # Open
				continue
			
			ax1.plot(cycle_count,res_avg,label=self.channel_list[j])
			ax2.plot(cycle_count,res_raw,label=self.channel_list[j])
			j=j+1
		
		axy = 'Resistance (Ohms)'
		ax1.set_xlabel('Cycle Count',fontsize=20)
		ax1.set_ylabel(axy,fontsize=20)
		ax1.tick_params(axis='both', which='major', labelsize=20)
		ax1.tick_params(axis='both', which='minor', labelsize=20)
		ax1.set_title(graph_title + 'Moving Average of ' + str(mov_avg),fontsize=24)
		ax1.legend(fancybox=True,framealpha=1,fontsize=20)
		
		ax2.set_xlabel('Cycle Count',fontsize=20)
		ax2.set_ylabel(axy,fontsize=20)
		ax2.tick_params(axis='both', which='major', labelsize=20)
		ax2.tick_params(axis='both', which='minor', labelsize=20)
		ax2.set_title(graph_title + 'Raw',fontsize=24)
		ax2.legend(fancybox=True,framealpha=1,fontsize=20)

		
		fig.savefig(self.dirs + self.title + ' Resistance_cycle_plot_' + self.timestamp + '.jpg')
		print('\n Raw cycle plot created')


	def read_parquet_file(self,parquet_file):
		dfp=pd.read_parquet(parquet_file)
		self.bigdf = dfp 
		self.title = parquet_file[0:18] # Morteza wants 14
		print(self.title)
		self.dirs = self.title +'\\'
		

		self.dirs = self.title +'\\'
		if os.path.exists(self.dirs):
			pass
		else:
			os.makedirs(self.dirs)
		# 	try:
		# 		shutil.move(parquet_file,self.dirs + parquet_file)
		# 	except:
		# 		print('Could not move parquet file!!!')


		return self.title,self.indicator



	def save_df_to_parquet(self):
		df_in = self.bigdf
		df_in.to_parquet(self.dirs + self.title + '_Reliability_'+ self.timestamp +  '.parquet',engine='pyarrow')


	def move_pngs(self):
		png_files = glob.glob('**.png')
		txt_files = glob.glob('**.txt')
		parquet_files = glob.glob('**.parquet')


		if os.path.exists(self.dirs):
			pass
		else:
			os.makedirs(self.dirs)

		try: 
			for files in png_files:
				shutil.move(files,self.dirs + files)
		except:
			print('\n No .pngs to move!!! \n ')
			pass

		try:
			for files in txt_files:
				shutil.move(files,self.dirs + files)
		except:
			print('\n No .txts to move!!! \n ')
			pass

		try:
			for files in parquet_files:
				shutil.move(files,self.dirs + files)
		except:
			print('\n No parquets to move!!! \n ')
			pass

	def move_to_Ndrive(self):
		## Check if they have N drive mapped 
		Ndrive_prefix = 'N:\\4L_test_data\\'
		if os.path.exists(Ndrive_prefix):
			pass
		else:
			print('\n \n Ndrive is not mapped!!!! Data is not backed up')
		
		print('\n Starting the Ndrive copy now, grab a coffee cause this will take a minute \n')

		n_path = Ndrive_prefix + self.title
	
		if os.path.exists(n_path):
			print('\n Serial Already exists on Ndrive, replace by hand IF truly necessary \n Try to avoid this!!!')
		else:
			shutil.copytree(self.dirs, n_path)
			print('\n Backed up to N drive \n')
		print('HEAT STARS analysis complete. \n Files can be found in the folder you ran this and they are backed up on the Ndrive')


	def end(self):
		print('Finished!!')
		
		sys.exit()
###################################################################


def main():

	h = HEAT_Analysis()
	print('\n Input the answer in the parenthesis \n')
	#h.glob_search_csv()
	glob_par = glob.glob('**.parquet')
	#h.create_bigdf_new()
	#h.save_df_to_parquet()
	h.read_parquet_file(glob_par[0])


	while True:
		prompt1 = input('\n What size bend rod was used? \n')
		if prompt1 == '7':
			rod_d = prompt1
			break
		elif prompt1 == '5':
			rod_d = prompt1
			break
		elif prompt1 == '4':
			rod_d = prompt1
			break
		elif prompt1 == '3':
			rod_d = prompt1
			break
		else:
			print('Only 3, 4, 5, and 7 have been tested')

	while True:
		prompt2 = input('\n Manufacturer? \n (i) In-House (RDL) \n (a) Altaflex \n')
		if prompt2 == 'i':
			manufacturer = 'In-house'
			break
		elif prompt2 == 'a':
			manufacturer = 'Altaflex'
			break
		else:
			print('not an option')

	while True:
		prompt22 = input('\n Cut-out? \n (c) Cut-out \n (p) Polyimide-Backing \n')
		if prompt22 == 'c':
			backplane = 'cut-out'
			break
		elif prompt22 == 'p':
			backplane = 'polyimide'
			break
		else:
			print('not an option')


	while True:
		prompt3 = input('\n Shape? \n (t) Straight \n (p) Serp \n')
		if prompt3 == 't':
			shape = 'straight'
			break
		elif prompt3 == 'p':
			shape = 'serpentine' #comment
			break
		else:
			print('Try again there bud')

	while True:
		prompt4 =input('\n Encapsulation? \n (y) yes or (n) no \n')
		if prompt4 =='y':
			encap = 'Y'
			break
		elif prompt4 =='n':
			encap = 'N'
			break
		else:
			print('Try again there bud')

	while True:
		prompt5 = input('\n 8 or 16 channel? \n (8) or (16) \n ')
		if prompt5 == '8' or prompt5 =='16':
			daq_number = prompt5
			break
		else:
			print('Try again there bud')



	### Run standard functions
	#h.find_first_row()
	#h.create_bigdf_new()
	#h.save_df_to_parquet()

	# Create and append limit	
	h.create_limitdf(rod_d,manufacturer,encap,daq_number,backplane,shape)
	
	
	h.append_limit_df_to_master()

	#h.plot_bigdf_moving_average()
	#h.master_to_percentage_plt()
	#h.master_v_trace_width()

	h.move_pngs()
	h.move_to_Ndrive()
	h.end()
					




if __name__ == '__main__':
	main()


	### J1 to J1