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


def lim30for10(df):
	cycle_fail = None
	in_last = None
	for index,row in df.iterrows():
		if in_last == None: ## Establish the index and cycle last
			in_last = index
			cyc_start = row['Cycle']

		track = index - in_last

		if track > 1.1:                    # See if we pull data below limit
			cyc_start = row['Cycle']

		cyc_now = row['Cycle']             #Grab Current cycle
		cyc_diff = cyc_now - cyc_start

		if cyc_diff >= 10:                 #Break if you the data has remained above for 10 cycles
			cycle_fail = cyc_now
			break

		in_last = index
	
	if cycle_fail == None:
		cycle_fail = 'Did not reach limit'

	return cycle_fail   

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
		self.limit_df_ave = None
		self.bigdf_list = []
		#



	#### Finds all csvs, creates file names, new directory
	def glob_search_csv(self):
		while True:
			glob_string = '**\\**.csv' # this generic one will just find all csv's
			self.file_list = glob.glob(glob_string,recursive = True) 
			print(self.file_list)
			prompt = input('Are these the right files?')
			if prompt == 'y':
				break
			else:
				continue

		
		self.title = self.file_list[0][0:13]
		if 'Hack' in self.title or 'AC' in self.title:
			self.indicator = 0
		elif 'CuS' in self.title and 'LM' not in self.title:
			self.indicator = 1
		elif 'CUS' in self.title and 'LM' not in self.title:
			self.indicator =1 
		elif 'LM' in self.title:
			self.indicator = 2
		elif 'INT' in self.title:
			self.indicator = 7
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
		search_word = 'Time (Seconds'
		if self.file_list is None:
			print('No csv files found')
		else:
			with open(self.file_list[0] , 'r') as f:
				lines = f.readlines()
				j = 0
				for line in lines:
					if search_word in line:
						self.start_line = lines.index(line)
						print(self.start_line)
						
					if self.start_line is not None:
						break
						
					j = j+1
					if j > 150:
						break
			  

	def create_bigdf(self):
		# Initialize an empty dictionary to store the dataframes

		csv_files = self.file_list
		
		self.bigdf_top = pd.DataFrame()
		self.bigdf_bottom = pd.DataFrame()
		# Loop through each CSV file and convert it to a pandas dataframe
		for file in csv_files:
			
			 ### Open and delete first 4 rows
			with open(file , 'r') as f:
				data = f.readlines()[self.start_line:]
			
			# Then read from the next rows without joining the data
			df = pd.read_csv(io.StringIO('\n'.join(data)))

			if '_T_' in file: 
				self.bigdf_top = pd.concat([self.bigdf_top,df])
			else:
				self.bigdf_bottom = pd.concat([self.bigdf_bottom,df])
			### At this point you have the raw data and its named, time to move it

		### Keep these loops separate to not mess up work flow
	
		# raw_dirs = self.dirs + 'Raw\\'
		# if os.path.exists(raw_dirs):
		# 	pass
		# else:
		# 	os.makedirs(raw_dirs)
		self.bigdf_list = [self.bigdf_top,self.bigdf_bottom]

	def move_pngs(self):
		png_files = glob.glob('**\\**.png')


		if os.path.exists(self.dirs):
			pass
		else:
			os.makedirs(self.dirs)


		for files in png_files:
			shutil.move(files,self.dirs + files)



	def create_limitdf(self,sipsize,underfill,encapsulation):

		limit_columns = ['Title','Date','Sample','Sample Type','Physical Positon','Underfill','Encapsulation','Strain (%)','Start (ohms)','10 % increase','30 % limit','> 6 ohms','> 10 ohms']
		limit_df=pd.DataFrame([],columns=limit_columns)


		title = self.title
		date = self.mini_timestamp
		if sipsize == '3':
			s_type = '3x3'
			daq_list = ['DAQ1','DAQ2','DAQ3']
		else:
			s_type = '9x9'
			daq_list = ['DAQ1']

		
		d = 0
		for z in self.bigdf_list:
			bigdf = z
			if d == 0:
				if sipsize == '3':
					sip_list = ['SIP3','SIP1','SIP2']
				else:
					sip_list = ['SIP1']
			else:
				if sipsize == '3':
					sip_list = ['SIP4','SIP6','SIP5']
				else:
					sip_list = ['SIP2']
			
			j=0
			for i in daq_list:
				if len(bigdf) < 4:
					continue

				# Create a rolling average
				res_raw = bigdf[i]
				if res_raw.iloc[0] < 0.0 or res_raw.iloc[4]>110:
					j = j + 1
					continue
				

				if self.instrument == 'Benderita':
					strain = '7mm Bend'
				else:
					raw_strain = float(self.meta_dict['Displacement per Cycle']) / float(self.meta_dict['Length - Preloaded'])
					strain = np.round(raw_strain,2)*100

				small_df = bigdf[['Cycle',i]].copy()

				p30up = small_df[i].iloc[0] * 1.3
				p30_df = small_df[small_df[i] > p30up]
				p30for10_lim = lim30for10(p30_df)				


				compare = bigdf[bigdf[i] > 6].reset_index()
				compare2 = bigdf[bigdf[i] > 10].reset_index()

				
				res_start = bigdf[i].iloc[0]

				res10p_lim = res_start + res_start * 0.1
				compare10p = bigdf[bigdf[i] > res10p_lim].reset_index()

				if len(compare) < 5:
					cycle_6 = 'Did not reach limit'
				else:
					cycle_6 = compare['Cycle'].iloc[4]

				if len(compare2) < 5:
					cycle_10 = 'Did not reach limit'
				else:
					cycle_10 = compare2['Cycle'].iloc[4]
				
					
				if len(compare10p) < 5:
					cycle_res10p = 'Did not reach limit'
				else:			
					cycle_res10p = compare10p['Cycle'].iloc[4]

					
				
				row = pd.DataFrame([[title,date,sip_list[j],s_type,daq_list[j][-1],underfill,encapsulation,strain,res_start,cycle_res10p,p30for10_lim,cycle_6,cycle_10]],
								   columns=limit_columns )
				limit_df = pd.concat([limit_df,row])
				j = j + 1
			d = d+1


		#limit_w_av = pd.concat([limit_df, limit_df.mean().to_frame('Average', ).T.astype(object), limit_df.std().to_frame('Standard Devication', ).T.astype(object)])
		# save Limit df
		limit_name =self.title + '_limits.csv'
		limit_df.to_csv(self.dirs + limit_name, index=False)
		
		self.limit = limit_name
		self.limit_df = limit_df

	def append_limit_df_to_master(self):
		### Find master csv
		npre = 'N:\\test_data\\'
		if self.indicator == 0:
			path = npre + 'Alloys\\'
			if self.instrument == 'Instronita':
				print('no master for instronita Alloys')
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
			path = npre + 'Liquid_Metal'
			if self.instrument == 'Instronita':
				masname = 'LM_master_strain_cycle.csv'
			else:
				print('No bend master yet')
				return
				masname = 'name soon'
			old_name = 'old\\LM_master_'
		elif self.indicator == 7:
			#path = 'N:\\Interconnects_test_data'
			path = 'N:\\Interconnects_test_data'
			if self.instrument == 'Instronita':
				print('No instronita master for Interconects')
				return
				
			else:
				masname = 'Interconnect_bend_master.csv'
			old_name = 'old\\Interconnect_bend_master_'
		else:
			print('No indicator assigned at append to master function')

		self.master_path = path + '\\master\\' 


		current = self.master_path + masname
		mas_to_folder = self.dirs + masname

		master_df_old = pd.read_csv(current)

		master_df_old.to_csv(self.master_path + old_name + self.mini_timestamp + '.csv',index=False)

		self.master_df = pd.concat([master_df_old,self.limit_df])

		try:
			self.master_df.to_csv(current,index=False)
		except:
			print('\n \n Someone locked the N drive master!!!! \n \n')
		#Save to current dirs
		self.master_df.to_csv(mas_to_folder,index=False)

		print('\n Appending new data to \n' + current + '\n')



	def Meta_data_reader(self):
		### Create a metadata dictionary
		meta_dict = {}

		try:
			file_list = glob.glob('**\\**Metadata**')
			# if len(file_list) >1:
			# 	print('Too many metadata files')
			if len(file_list) == 0 :
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
		for i in self.bigdf_list:
			bigdf = i
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
				

							
				elif self.indicator == 2: 
					try:
						sample_length = float(self.meta_dict['Length - Preloaded'])
					except:
						sample_length = 15
						print('Cannot find sample length \n Assuming sample length is 15 \n')

					res_norm = res_raw/((sample_length + bigdf['Displacement (mm)'])/10) ### Need to change
					ax1.plot(cycle_count,res_norm,label=hack_labels[j])
					ax2.plot(cycle_count,res_raw,label=hack_labels[j])

				else: 
					ax1.plot(cycle_count,res_avg,label=hack_labels[j])
					ax2.plot(cycle_count,res_raw,label=hack_labels[j])
					



					
				j=j+1

			if self.indicator == 2:
				axy = 'Unit Resistance (Ohms/cm)'
				
			else:
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

			
			fig.savefig(self.dirs + self.title + ' Resistance_cycle_plot' + self.timestamp + '.jpg')
			print('\n Raw cycle plot created')


	def save_df_to_parquet(self):
		for i in self.bigdf_list:
			df_in = i
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
		df = self.master_df
		#Skip
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
		
		
		if self.instrument == 'Instronita':
			x_ax = '> 100 ohms'
			x_ax_label = 'Cycle Count'
			y_ax = 'Strain (%)'
			y_ax_label = 'Strain % during Cycling'
		else:
			print('Strain values for Benderita Master coming soon')
			return

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
		df = self.limit_df
		

		y_bars = ['10 % increase (ohms)']
		### skips string columns
		for a in y_bars:
			column_of_interest = a
			if pd.api.types.is_numeric_dtype(df[column_of_interest]) == True:
				pass
			else:
				if df[column_of_interest].str.contains('Did not').any() == False:
					print('Unexpected strings in the Master plot, skipping')
					return
					pass
				else:
					df = df.drop(df[df[column_of_interest].str.contains('Did not',na=False)].index)
					df[column_of_interest] = pd.to_numeric(df[column_of_interest])
				

			## Find the max value
			labels = df['Sample']
			values = df[column_of_interest]
			vals_max = max(values)
			
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
		
		max_1 = max(values_1)
		max_2 = max(values_2)
		
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
		elif self.indicator ==7:
			n_path = 'N:\\Interconnects_test_data\\' + self.title
	
		if os.path.exists(n_path):
			print('\n Serial Already exists on Ndrive, replace by hand IF truly necessary \n Try to avoid this!!!')
		else:
			shutil.copytree(self.dirs, n_path)
			print('\n Backed up to N drive \n')
		

	def end(self):
		print('Finished!!')
		print('HEAT STARS analysis complete. \n Files can be found in the folder you ran this and they are backed up on the Ndrive')
		sys.exit()


def main():
	### Intialize Ndrive folder to save
	files_for_Ndrive = []
	current_direct = os.getcwd() + '\\'

	h = HEAT_Analysis()

	while True:
		prompt = input('\n Is it a 3x3 or a 9x9?   ')
		if prompt == '3' or prompt == '9':
			break
		else:
			print('enter 3 or 9')

	while True:
		prompt2 = input('\n Encapsulatation Material? (n) to skip   ')
		if prompt2 == 'n':
			encap = 'N/A'
			break
		else:
			encap = prompt2
			break

	while True:
		prompt3 = input('\n Underfill Material? (n) to skip   ')
		if prompt3 == 'n':
			uf = 'N/A'
			break
		else:
			uf = prompt3
			break


	title,indicator = h.glob_search_csv()
	## build title
	fresh_directory = current_direct + title

	h.find_first_row()
	h.Meta_data_reader()
	#h.move_pngs()

	### These four things get done in this order no matter what
	h.create_bigdf()
	#h.save_df_to_parquet()
	h.plot_bigdf_moving_average()
	h.create_limitdf(prompt,uf,encap)
	#h.mini_barplot()
	h.append_limit_df_to_master()
	h.move_to_Ndrive()

	h.end()

if __name__ == '__main__':
	main()