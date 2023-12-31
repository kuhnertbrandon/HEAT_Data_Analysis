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
		self.bend_list = ['05','06','07','08','09','10','11']   ### NEED
		self.instronita_list = ['01','02','03','04']
		self.master_path = None
		self.master_df = None
		self.limit_name = None
		self.limit_df = None



	#### Finds all csvs, creates file names, new directory
	def glob_search_csv(self):
		glob_string = '**.csv' # this generic one will just find all csv's 
		self.file_list = glob.glob(glob_string,recursive = True)
		self.title = self.file_list[0][0:13]
		if 'Hack' in self.title or 'AC' in self.title:
			self.indicator = 0
		elif 'CuS' in self.title and 'LM' not in self.title:
			self.indicator = 1
		elif 'CUS' in self.title and 'LM' not in self.title:
			self.indicator =1 
		elif 'LM' in self.title:
			self.indicator = 2
		elif 'ADEBS35' in self.title:
			self.indicator = 3
		else:
			print('\n UNRECOGNIZED SERIAL \n PLS FIX')
			sys.exit()

		self.dirs = self.title +'\\'
		if os.path.exists(self.dirs):
			pass
		else:
			os.makedirs(self.dirs)

		print('\n \n')
		print(self.file_list)

		print('\n \n')

		return self.title, self.indicator
		


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
			  

	def create_bigdf(self):
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
			### At this point you have the raw data and its named, time to move it

		### Keep these loops separate to not mess up work flow

		self.bigdf = self.bigdf.sort_values(by=['Time (Seconds)'])
	
		raw_dirs = self.dirs + 'Raw\\'
		if os.path.exists(raw_dirs):
			pass
		else:
			os.makedirs(raw_dirs)


		for files in csv_files:
			shutil.move(files,raw_dirs + files)
		#

	def move_pngs(self):
		png_files = glob.glob('**.png')


		if os.path.exists(self.dirs):
			pass
		else:
			os.makedirs(self.dirs)


		for files in png_files:
			shutil.move(files,self.dirs + files)


	def create_limitdf(self):

		limit_columns = ['Title','Date','Sample','Physical Positon','Strain (%)','Start (ohms)','10 % increase (ohms)','> 6 ohms','> 10 ohms','> 50 ohms','> 100 ohms','Omars 10 %','Omars 50 %']
		limit_df=pd.DataFrame([],columns=limit_columns)


		title = self.title
		date = self.mini_timestamp
		daq_list = ['DAQ1','DAQ2','DAQ3','DAQ4','DAQ5','DAQ6','DAQ7','DAQ8']
		if self.indicator == 0:
			hack_labels = ['MSW','100D_1','100D_2','MSWS','CPWS','85D_1','85D_2','CPW']
		else:
			hack_labels = ['Sample 1','Sample 2','Sample 3','Sample 4','Sample 5','Sample 6','Sample 7','Sample 8']
		

		bigdf = self.bigdf
		#### Omars Limit work
		f_100_cycles = bigdf[bigdf['Cycle'] <= 100]
		last = f_100_cycles['Cycle'].iloc[0]
		tops = f_100_cycles.iloc[0]
		for index,row in f_100_cycles.iterrows():
			if row['Cycle'] > last and (row['Cycle'] % 1) == 0.5:
				tops = pd.concat([tops,row])
				last = row['Cycle']

		#shrink the df
		small_list = ['Cycle'] + daq_list
		smoldf = bigdf[small_list].copy()
		bigdf = None


		j=0
		for i in daq_list:
			# Create a rolling average
			res_raw = smoldf[i]
			
			if res_raw.iloc[0] < 0.0 or res_raw.iloc[4]>110:
				j = j + 1
				continue

			if self.instrument == 'Benderita':
				strain = '7mm Bend'
			else:
				raw_strain = float(self.meta_dict['Displacement per Cycle']) / float(self.meta_dict['Length - Preloaded'])
				strain = np.round(raw_strain,2)*100

			### Omar's limit
			omars_start = tops[i].mean()
			omars_10p = omars_start * 1.1
			omars_50p = omars_start * 1.5	

			
			
			
			
			
			res_start = smoldf[i].iloc[0]

			res10p_lim = res_start + res_start * 0.1
			
			
			compare = smoldf[smoldf[i] > 6].reset_index()
			if len(compare) < 5:
				cycle_6 = 'Did not reach limit'
			else:
				cycle_6 = compare['Cycle'].iloc[4]
			compare	= None

			compare2 = smoldf[smoldf[i] > 10].reset_index()
			if len(compare2) < 5:
				cycle_10 = 'Did not reach limit'
			else:
				cycle_10 = compare2['Cycle'].iloc[4]
			compare2 = None

			compare3 = smoldf[smoldf[i] > 50].reset_index()
			if len(compare3) < 5:
				cycle_50 = 'Did not reach limit'
			else:
				cycle_50 = compare3['Cycle'].iloc[4]
			compare3 = None

			compare4 = smoldf[smoldf[i] > 100].reset_index()
			if len(compare4) < 5:
				cycle_100 = 'Did not reach limit'
			else:			
				cycle_100 = compare4['Cycle'].iloc[4]
			compare4 = None
				
			compare10p = smoldf[smoldf[i] > res10p_lim].reset_index()
			if len(compare10p) < 5:
				cycle_res10p = 'Did not reach limit'
			else:			
				cycle_res10p = compare10p['Cycle'].iloc[4]
			compare10p = None

			omars_df_10p = smoldf[smoldf[i] > omars_10p].reset_index(drop=True)
			if len(omars_df_10p) < 5:
				omars_10p_limit = 'Did not reach limit'
			else:			
				omars_10p_limit = omars_df_10p['Cycle'].iloc[4]
			omars_df_10p = None

			omars_df_50p = smoldf[smoldf[i] > omars_50p].reset_index(drop=True)
			if len(omars_df_50p) < 5:
				omars_50p_limit = 'Did not reach limit'
			else:			
				omars_50p_limit = omars_df_50p['Cycle'].iloc[4]
			omars_df_50p = None
				
			
			row = pd.DataFrame([[title,date,hack_labels[j],daq_list[j][-1],strain,res_start,cycle_res10p,cycle_6,cycle_10,cycle_50,cycle_100,omars_10p_limit,omars_50p_limit]],
							   columns=limit_columns )
			limit_df = pd.concat([limit_df,row]) #limit_df.append(row)
			j = j + 1

		# save Limit df
		limit_name =self.title + '_limits.csv'
		limit_df.to_csv(self.dirs + limit_name,index=False)
		
		self.limit_name = limit_name
		
		self.limit_df = limit_df

	def append_limit_df_to_master(self):
		### Find master csv
		npre = 'N:\\test_data\\'
		if self.indicator == 0:
			path = npre + 'Alloys\\'
			if self.instrument == 'Instronita':
				print('\n \n no master for instronita Alloys \n \n ')
				return
			else:
				masname = 'Hack_bend_master.csv'
			old_name = 'old\\Hack_master_'
		elif self.indicator == 1:
			path = npre + 'CuS\\'
			if self.instrument == 'Instronita':
				masname = 'CuS_master_strain_cycle.csv'
			else:
				masname = 'CuS_bend_master.csv'
			old_name = 'old\\CuS_master_'
		elif self.indicator == 2:
			path = npre + 'Liquid_Metal\\'
			if self.instrument == 'Instronita':
				masname = 'LM_master_strain_cycle.csv'
			else:
				masname = 'LM_bend_master.csv'
			old_name = 'old\\LM_master_'
		else:
			print('No indicator assigned at append to master function')

		self.master_path = path + self.instrument + '\\master\\' 


		current = self.master_path + masname
		mas_to_folder = self.dirs + masname

		master_df_old = pd.read_csv(current)

		master_df_old.to_csv(self.master_path + old_name + self.mini_timestamp + '.csv',index=False)

		self.master_df = pd.concat([master_df_old,self.limit_df])

		self.master_df.to_csv(current,index=False)
		#Save to current dirs
		self.master_df.to_csv(mas_to_folder,index=False)

		print('\n Appending new data to \n' + current + '\n')



	def Meta_data_reader(self):
		### Create a metadata dictionary
		meta_dict = {}

		try:
			file_list = glob.glob('**Metadata**')
			if len(file_list) >1:
				print('Too many metadata files')
			elif len(file_list) == 0 :
				print('No Metadata file found!!!')
				sys.exit()
			else:
				file = file_list[0]
		except:
			file_list = None

		if file_list == None:
			print('No Metadata file available, skipping this part')
		else:
			search_words = self.meta_list
			with open(file , 'r') as f:
					data = f.readlines()
					for line in data:
						for word in search_words:
							if line.find(word) != -1:
								line = line.strip('\n')
								stuff = line.split(':')
								if 'system_settings.json' in stuff:
									for i, s in enumerate(stuff):
										if word in s:
											mini_stuff = stuff[i+1].split(',')
											meta_dict[word] = mini_stuff[0]
								else:
									meta_dict[stuff[0]] = stuff[1]

		#### Assign device ID based on the serial number
		
		instrument = None
		
		devid = 'device_id'

		if devid in meta_dict:
			value = meta_dict[devid]
			if any(bend_serial in value for bend_serial in self.bend_list ) == True:
				instrument = 'Benderita'
				
				
		if devid in meta_dict:
			value = meta_dict[devid]
			if (any(instronita_serial in value for instronita_serial in self.instronita_list )) == True:
				if instrument is None:
					instrument = 'Instronita'
				else:
					print('Instrument already assigned to Benderita!!!')

		if instrument == None:
			print('Rogue instrument serial number!!!!')
		
		# Move to new dirs
		shutil.move(file,self.dirs + file)

		print(instrument)
		## Return important variables to the class
		self.meta_dict = meta_dict
		self.instrument = instrument

	#def plot_mini_plots(self):
	### Commented below is to make separate plots
		# if self.limit_df = None:
		# 	print('Limit df is not established yet')
		# else:
		# 	failure_vals = self.limit_df['> 100 ohms'].values.to_list

		# ## Plot mini_plots as well
			# figz,axz = plt.figure(figure=(20,15))
			# mini_plots_path = self.dirs + 'mini_plots\\'
			# if os.path.exists(mini_plots_path):
			# 	pass
			# else:
			# 	os.makedirs(mini_plots_path)

			# last = failure_vals[j]
			# axz.plot(cycle_count[:last],res_norm[:last])
			# axz.xlabel('Cycle Count',fontsize=20)
			# axz.ylabel('Unit Resistance (Ohms/cm)',fontsize=20)
			# axz.tick_params(axis='both', which='major', labelsize=20)
			# axz.tick_params(axis='both', which='minor', labelsize=20)
			# axz.title(hack_labels[j] + ': Cycle Count vs Unit Resistance',fontsize=24)
			# axz.ylim((0,100))
			# figz.savefig(mini_plots_path + '_' + hack_labels[j] + '_' + self.mini_timestamp + '.jpg')


	def plot_bigdf_moving_average(self):
		bigdf = self.bigdf
		mov_avg = int(np.round(len(bigdf)/1000,0))
		if mov_avg < 3:
			mov_avg = 3
		graph_title = self.title + ': '
		daq_list = ['DAQ1','DAQ2','DAQ3','DAQ4','DAQ5','DAQ6','DAQ7','DAQ8']

		if self.indicator ==0:
			hack_labels = ['MSW','100D_1','100D_2','MSWS','CPWS','85D_1','85D_2','CPW']
		else:
			hack_labels = ['Sample 1','Sample 2','Sample 3','Sample 4','Sample 5','Sample 6','Sample 7','Sample 8']
			
	 


		### Moving Average Plot
		plt.rcParams['agg.path.chunksize'] = 10000
		fig, (ax1,ax2) = plt.subplots(2,figsize=(30,20))
		j = 0
		for i in daq_list:


			# Create a rolling average
			res_raw = bigdf[i]
			res_avg = res_raw.rolling(window=mov_avg).mean()
			cycle_count = bigdf['Cycle']
			if res_raw.iloc[0] < 0.0 or res_raw.iloc[4]>110:
				j=j+1 # Open
				continue
			

						
			elif self.indicator == 0 or self.indicator == 1 or self.indicator == 3: 
				ax1.plot(cycle_count,res_avg,label=hack_labels[j])
				ax2.plot(cycle_count,res_raw,label=hack_labels[j])
			elif self.indicator == 2 and self.instrument == 'Instronita': 
				try:
					sample_length = float(self.meta_dict['Length - Preloaded'])
				except:
					sample_length = 15
					print('Cannot find sample length \n Assuming sample length is 15 \n')

				res_norm = res_raw/((sample_length + bigdf['Displacement (mm)'])/10) ### Need to change
				ax1.plot(cycle_count,res_norm,label=hack_labels[j])
				ax2.plot(cycle_count,res_raw,label=hack_labels[j])
			elif self.indicator == 2 and self.instrument == 'Benderita':
				ax1.plot(cycle_count,res_avg,label=hack_labels[j])
				ax2.plot(cycle_count,res_raw,label=hack_labels[j])



				
			j=j+1

		if self.indicator == 0 or self.indicator ==1:
			axy = 'Resistance (Ohms)'
		elif self.instrument =='Benderita':
			axy = 'Resistance (Ohms)'
		else:
			axy = 'Unit Resistance (Ohms/cm)'
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

		
		fig.savefig(self.dirs + self.title + ' Resistance_cycle_plot' + self.timestamp + '.jpg')
		print('\n Raw cycle plot created')


	def save_df_to_parquet(self):
		df_in = self.bigdf
		df_in.to_parquet(self.dirs + self.title + '_Reliability_'+ self.timestamp +  '.parquet',engine='pyarrow')

	def read_parquet_file(self,parquet_file):
		dfp=pd.read_parquet(parquet_file)
		self.bigdf = dfp 
		self.title = parquet_file[0:13] # Morteza wants 14
		self.dirs = self.title +'\\'
		if 'Hack' in self.title or 'AC' in self.title:
			self.indicator = 0
		elif 'CuS' in self.title and 'LM' not in self.title:
			self.indicator = 1
		elif 'CUS' in self.title and 'LM' not in self.title:
			self.indicator =1 
		elif 'LM' in self.title:
			self.indicator = 2
		print('Read the df from a parquet')

		self.dirs = self.title +'\\'
		if os.path.exists(self.dirs):
			pass
		else:
			os.makedirs(self.dirs)
			try:
				shutil.move(parquet_file,self.dirs + parquet_file)
			except:
				print('Could not move parquet file!!!')


		return self.title,self.indicator


	def master_scatter_plot(self): 
		## Grab from N drive
		if self.master_df is None:
			print('No Master available')
			return
		else:
			df = self.master_df
		#Skip

		if self.instrument == 'Instronita':
			x_ax = '> 100 ohms'
			x_ax_label = 'Cycle Count'
			y_ax = 'Strain (%)'
			y_ax_label = 'Strain % during Cycling'
		else:
			print('Strain values for Benderita Master coming soon')
			return

		column_of_interest = '> 100 ohms'
		if pd.api.types.is_numeric_dtype(df[column_of_interest]) == True:
			pass
		else:
			if df[column_of_interest].str.contains('not').any() == False:
				print('Unexpected strings in the Master plot, skipping')
				return
				pass
			else:
				df = df.drop(df[df[column_of_interest].str.contains('not',na=False)].index)
				df[column_of_interest] = pd.to_numeric(df[column_of_interest])
				print(df.dtypes)
		
		
		# ### Move old plots away
		# old_files = glob.glob(self.master_path + '**.csv')
		# print(old_files)
		# try
		# for i in old_files:
		# 	print(i)
		# 	shutil.move(i,self.master_path + 'old\\master_from_' +  self.mini_timestamp + '.csv')
		
		
	
		title_last = df['Title'].iloc[-1]

		lastdf = df[df['Title'] == title_last]
		

		olddf = df[df['Title'] != title_last]
		
		column_list = df.head()
		if x_ax in column_list:
			pass
		else:
			print('Error! cannot find x axis')

		if y_ax in column_list:
			pass
		else:
			print('Error! cannot find x axis')

		plt.figure(figsize=(20,15))
		plt.scatter(olddf[x_ax],olddf[y_ax],color='grey')
		plt.scatter(lastdf[x_ax],lastdf[y_ax],color='red')
		plt.xlabel(x_ax_label,fontsize=20)
		plt.ylabel(y_ax_label,fontsize=20)
		plt.tick_params(axis='both', which='major', labelsize=20)
		plt.tick_params(axis='both', which='minor', labelsize=20)
		plt.title('MOA w/ Most Recent '+ title_last +' in red',fontsize=24)
		plt.ylim((0,100))
		plt.savefig(self.dirs + 'Master_plot_' + self.mini_timestamp + '.jpg')

		plt.figure(figsize=(20,15))
		plt.scatter(olddf[y_ax],olddf[x_ax],color='grey')
		plt.scatter(lastdf[y_ax],lastdf[x_ax],color='red')
		plt.xlabel(y_ax_label,fontsize=20)
		plt.ylabel(x_ax_label,fontsize=20)
		plt.tick_params(axis='both', which='major', labelsize=20)
		plt.tick_params(axis='both', which='minor', labelsize=20)
		plt.title('MOA w/ Most Recent '+ title_last +' in red',fontsize=24)
		plt.savefig(self.dirs + 'Master_plot_' + self.mini_timestamp + '_transposed.jpg')

	def mini_barplot(self):

		y_bars = ['> 100 ohms','10 % increase (ohms)']
		### skips string columns
		for a in y_bars:
			df = self.limit_df
			column_of_interest = a
			if pd.api.types.is_numeric_dtype(df[column_of_interest]) == True:
				pass
			else:
				if df[column_of_interest].str.contains('not').any() == False:
					print('Unexpected strings in the Master plot, skipping')
					return
					pass
				else:
					df = df.drop(df[df[column_of_interest].str.contains('Did not',na=False)].index)
					df[column_of_interest] = pd.to_numeric(df[column_of_interest])
				

			## Find the max value
			labels = df['Sample']
			values = df[column_of_interest]
			try:
				vals_max = max(values)
			except:
				return
			
			round_up = 10000
			window_max = np.round((vals_max/round_up),0)*round_up+round_up
			
			fig, ax = plt.subplots(figsize=(20,10))
			num_labels = len(labels)
			colors = plt.cm.viridis(np.linspace(0,1,num_labels))
			
			plt.bar(labels,values,color=colors)
			ax.set_ylabel('Cycles at Failure',fontsize=20)
			ax.set_title(self.title + ' Failure Plot',fontsize=24)
			ax.set_ylim(0,window_max)
			ax.yaxis.grid(which='major',linestyle='--')
			for i,v in enumerate(values):
				ax.text(i,v,v,ha = 'center')
			ax.set_yticks(np.arange(0,window_max,step=round_up))

			if a == '> 100 ohms':
				a_name = '100ohms_'
			elif a == '10 % increase (ohms)':
				a_name = '10_per_increase_'
			
			fig.savefig(self.dirs +  self.title + '_bar_plot_for_'+ a_name + self.mini_timestamp +'.jpg')


	def comparison_bar_plot_Hack(self,call_for_comparison):
		comparison_dictionary = {
		'1' : 'HackFPC_w_islands_limits.csv'
		}

		## master prefix and appending
		master_prefix = 'N:\\test_data\\Alloys\\Benderita\\master\\'
		#append file you are looking for 
		master_title = comparison_dictionary[call_for_comparison]
		master_path = master_prefix + master_title

		## Call N-drive here 
		old_df = pd.read_csv(master_path)
		new_df = self.limit_df
		
		## Find the max value
		labels_1 = old_df['Sample']
		values_1 = old_df['> 100 ohms']
		
		labels_2 = new_df['Sample']
		values_2 = new_df['> 100 ohms']
		try:
			max_1 = max(values_1)
			max_2 = max(values_2)
		except:
			return
		
		if max_1 > max_2:
			vals_max = max_1
		else:
			vals_max = max_2    
			
		round_up = 10000
		window_max = np.round((vals_max/round_up),0)*round_up+round_up
		
		
		fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
		num_labels = len(labels_1)
		colors = plt.cm.viridis(np.linspace(0,1,num_labels))
		
		ax1.bar(labels_1,values_1,color=colors)
		ax1.set_ylabel('Cycles at Failure',fontsize=20)
		ax1.set_title(master_title[:-4],fontsize=24)
		ax1.set_ylim(0,window_max)
		ax1.yaxis.grid(which='major',linestyle='--')
		for i,v in enumerate(values_1):
			ax1.text(i,v,v,ha = 'center')
		ax1.set_yticks(np.arange(0,window_max,step=round_up))
		
		
		ax2.bar(labels_2,values_2,color=colors)
		ax2.set_ylabel('Cycles at Failure',fontsize=20)
		ax2.set_title(self.title,fontsize=24)
		ax2.set_ylim(0,window_max)
		ax2.yaxis.grid(which='major',linestyle='--')
		for i,v in enumerate(values_2):
			ax2.text(i,v,v,ha = 'center')
		ax2.set_yticks(np.arange(0,window_max,step=round_up))
		
		fig.savefig(self.dirs +  self.title + '_compare_w_average.jpg')

	def move_to_Ndrive(self):
		## Check if they have N drive mapped 
		Ndrive_prefix = 'N:\\test_data\\'
		if os.path.exists(Ndrive_prefix):
			pass
		else:
			print('\n \n Ndrive is not mapped!!!! Data is not backed up')
		
		print('\n Starting the Ndrive copy now, grab a coffee cause this will take a minute \n')

		n_suffix = self.instrument +'\\'+ self.title
				

		if self.indicator == 1:
			n_path = Ndrive_prefix + 'CuS\\' + n_suffix
		elif self.indicator == 2:
			n_path = Ndrive_prefix + 'Liquid_Metal\\' + n_suffix
		elif self.indicator == 0:
			n_path = Ndrive_prefix + 'Alloys\\' + n_suffix

	
		if os.path.exists(n_path):
			print('\n Serial Already exists on Ndrive, replace by hand IF truly necessary \n Try to avoid this!!!')
		else:
			shutil.copytree(self.dirs, n_path)
			print('\n Backed up to N drive \n')
		

	def end(self):
		print('Finished!!')
		print('HEAT STARS analysis complete. \n Files can be found in the folder you ran this and they are backed up on the Ndrive')
		sys.exit()


######################################################################
def main():
	### Intialize Ndrive folder to save
	files_for_Ndrive = []
	current_direct = os.getcwd() + '\\'

	h = HEAT_Analysis()

	title,indicator = h.glob_search_csv()
	## build title
	fresh_directory = current_direct + title

	h.find_first_row()
	h.Meta_data_reader()
	h.move_pngs()

	### These four things get done in this order no matter what
	h.create_bigdf()
	h.save_df_to_parquet()
	h.plot_bigdf_moving_average()
	h.create_limitdf()
	h.mini_barplot()


	### Here on uses logic to identify file type
	if indicator == 0:
		h.append_limit_df_to_master()
		h.master_scatter_plot()
		# while True:
		# 	prompt = input("\n Which master plot do you want to compare with? (Input the number, s to skip) \n '1' : 'HackFPC_w_islands_limits.csv' \n")
		# 	if prompt == 's':
		# 		break
		# 	elif prompt == '1':
		# 		h.comparison_bar_plot_Hack(prompt)
		# 		break
		# 	else:
		# 		print('\n Not an option! \n ') 
	elif indicator == 1 :
		h.append_limit_df_to_master()
		h.master_scatter_plot()
	elif indicator == 2:
		h.append_limit_df_to_master()
		h.master_scatter_plot()
	else:
		print('UNRECOGNIZED Sample')

	
	h.move_to_Ndrive()
	h.end()
	

if __name__ == '__main__':
	main()