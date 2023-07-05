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
		self.meta_list = ['device_id','daq_limit_cycles','Length - Preloaded','Displacement per Cycle']
		self.master_scatter = None
		self.bend_list = ['05','06','07']   ### NEED
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
		elif 'LM' in self.title:
			self.indicator = 2
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
	
		raw_dirs = self.dirs + 'Raw\\'
		if os.path.exists(raw_dirs):
			pass
		else:
			os.makedirs(raw_dirs)


		for files in csv_files:
			shutil.move(files,raw_dirs + files)
		#



	def create_limitdf(self):

		limit_df=pd.DataFrame([],columns=['Title','Date','Sample','Physical Positon','Strain (%)','Start (ohms)','> 6 ohms','> 10 ohms','> 50 ohms','> 100 ohms'])


		title = self.title
		date = self.mini_timestamp
		daq_list = ['DAQ1','DAQ2','DAQ3','DAQ4','DAQ5','DAQ6','DAQ7','DAQ8']
		if self.indicator == 0:
			hack_labels = ['MSW','100D_1','100D_2','MSWS','CPWS','85D_1','85D_2','CPW']
		else:
			hack_labels = ['Sample 1','Sample 2','Sample 3','Sample 4','Sample 5','Sample 6','Sample 7','Sample 8']
		

		bigdf = self.bigdf
		j=0
		for i in daq_list:
			# Create a rolling average
			res_raw = bigdf[i]
			
			if res_raw.iloc[0] < 0.0 or res_raw.iloc[5]>2000:
				continue
			
			if self.instrument == 'Benderita':
				strain = '7mm Bend'
			else:
				raw_strain = self.meta_dict['Displacement per Cycle'] / self.meta_dict['Length - Preloaded']
				strain = np.round(raw_strain,2)*100


			compare = bigdf[bigdf[i] > 6].reset_index()
			compare2 = bigdf[bigdf[i] > 10].reset_index()
			compare3 = bigdf[bigdf[i] > 50].reset_index()
			compare4 = bigdf[bigdf[i] > 100].reset_index()
			
			res_start = bigdf[i].iloc[0]					### Need to fix so it finds 5 in a row not just first 5
			cycle_6 = compare['Cycle'].iloc[4]
			cycle_10 = compare2['Cycle'].iloc[4]
			cycle_50 = compare3['Cycle'].iloc[4]
			cycle_100 = compare4['Cycle'].iloc[4]

			
			row = pd.DataFrame([[title,date,hack_labels[j],daq_list[j][-1],res_start,cycle_6,cycle_10,cycle_50,cycle_100]],
							   columns=['Title','Date','Sample','Physical Positon','Strain (%)','Start (ohms)','> 6 ohms','> 10 ohms','> 50 ohms','> 100 ohms'])
			limit_df = limit_df.append(row)
			j = j + 1

		# save Limit df
		limit_name =self.title + '_limits.csv'
		limit_df.to_csv(self.dirs + limit_name,index=False)
		
		self.limit_name
		self.limit_df = limit_df

	def append_limit_df_to_master(self):
		### Find master csv
		npre = 'N:\\test_data\\'
		if self.indicator == 2:
			path = npre + 'Liquid_Metal\\'
			masname = 'LM_master_strain_cycle.csv'
			old_name = 'old\\LM_master_'
		else:
			print('Only have liquid metal master for now')

		self.master_path = path + self.instrument + '\\master\\' 

		current = self.master_path + mas_name
		mas_to_folder = self.dirs + mas_name

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
											print(i)
											mini_stuff = stuff[i+1].split(',')
											meta_dict[word] = mini_stuff[0]
								else:
									meta_dict[stuff[0]] = stuff[1]

		#### Assign device ID based on the serial number
		instrument = None
		print(meta_dict)
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
		fig, (ax1,ax2) = plt.subplots(2,figsize=(30,20))
		j = 0
		for i in daq_list:


			# Create a rolling average
			res_raw = bigdf[i]
			res_avg = res_raw.rolling(window=mov_avg).mean()
			cycle_count = bigdf['Cycle']
			if res_raw.iloc[0] < 0.0 or res_raw.iloc[2]>2000:
				j=j+1 # Open
				continue
			

						
			elif self.indicator == 0 or self.indicator == 1: 
				ax1.plot(cycle_count,res_avg,label=hack_labels[j])
				ax2.plot(cycle_count,res_raw,label=hack_labels[j])
			else: 
				try:
					sample_length = meta_dict['Length - Preloaded']
				except:
					sample_length = 15
					print('Cannot find sample length \n Assuming sample length is 15 \n')

				res_norm = res_raw/((sample_length + bigdf['Displacement (mm)'])/10) ### Need to change
				ax1.plot(cycle_count,res_avg,label=hack_labels[j])
				ax2.plot(cycle_count,res_raw,label=hack_labels[j])



				
			j=j+1

		if self.indicator == 0:
			axy = 'Resistance (Ohms)'
		else:
			axy = 'Unit Resistance (Ohms/cm)'
		ax1.set_xlabel('Cycle Count',fontsize=20)
		ax1.set_ylabel(axy,fontsize=20)
		ax1.tick_params(axis='both', which='major', labelsize=20)
		ax1.tick_params(axis='both', which='minor', labelsize=20)
		ax1.set_title(graph_title + 'Moving Average of ' + str(mov_avg),fontsize=20)
		ax1.legend(fancybox=True,framealpha=1,fontsize=20)
		
		ax2.set_xlabel('Cycle Count',fontsize=20)
		ax2.set_ylabel(axy,fontsize=20)
		ax2.tick_params(axis='both', which='major', labelsize=20)
		ax2.tick_params(axis='both', which='minor', labelsize=20)
		ax2.set_title(graph_title + 'Raw',fontsize=20)
		ax2.legend(fancybox=True,framealpha=1,fontsize=20)
		
		fig.savefig(self.dirs + self.title + ' Resistance_cycle_plot' + self.timestamp + '.jpg')
		print('\n Raw cycle plot created')


	def save_df_to_parquet(self):
		df_in = self.bigdf
		df_in.to_parquet(self.dirs + self.title + '_Reliability_'+ self.timestamp +  '.parquet',engine='pyarrow')

	def read_parquet_file(self,parquet_file):
		dfp=pd.read_parquet(parquet_file)
		self.bigdf = dfp 
		self.title = parquet_file[0:13]
		self.dirs = self.title +'\\'
		if 'Hack' in self.title or 'AC' in self.title:
			self.indicator = 0
		elif 'CuS' in self.title and 'LM' not in self.title:
			self.indicator = 1
		elif 'LM' in self.title:
			self.indicator = 2
		print('Read the df from a parquet')

		return self.title,self.dirs



	def master_scatter_plot(self): 
		## Grab from N drive
		df = self.master_df
		
		
		### Move old plots away
		pictures = glob.glob(self.master_path + '**.csv')
		for i in pictures:
			shutil.move(i,self.master_path + 'old\\LM_' + self.mini_timestamp + '.csv')
		
		x_ax = '> 100 ohms'
		y_ax = 'Strain (%)'

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
		plt.scatter(df[x_ax][:-4],df[y_ax][:-4],color='grey')
		plt.scatter(lastdf[x_ax],lastdf[y_ax],color='red')
		plt.xlabel('Cycle Count',fontsize=20)
		plt.ylabel('Strain % during Cycling',fontsize=20)
		plt.tick_params(axis='both', which='major', labelsize=20)
		plt.tick_params(axis='both', which='minor', labelsize=20)
		plt.title('Master Graph w/ Most Recent '+ title_last +' in red',fontsize=24)
		plt.ylim((0,100))
		plt.savefig(self.dirs + 'Master_plot_' + self.mini_timestamp + '.jpg')

	def mini_barplot(self):
		df = self.limit_df
		
		## Find the max value
		labels = df['Sample']
		values = df['> 100 ohms']
		
		vals_max = max(values)
		
		
		round_up = 10000
		window_max = np.round((vals_max/round_up),0)*round_up+round_up
		
		
		fig, ax = plt.figure(figsize=(20,10))
		num_labels = len(labels)
		colors = plt.cm.viridis(np.linspace(0,1,num_labels))
		
		ax.bar(labels,values,color=colors)
		ax.set_ylabel('Cycles at Failure',fontsize=20)
		ax.set_title(master_title[:-4],fontsize=24)
		ax.set_ylim(0,window_max)
		ax.yaxis.grid(which='major',linestyle='--')
		for i in range(num_labels):
			ax.text(i,values[i],values[i],ha = 'center')
		ax.set_yticks(np.arange(0,window_max,step=round_up))
		
		
		fig.savefig(self.dirs +  self.title + '_bar_plot_' + self.mini_timestamp +'.jpg')


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
		ax1.set_yticks(np.arange(0,window_max,step=round_up))
		
		
		ax2.bar(labels_2,values_2,color=colors)
		ax2.set_ylabel('Cycles at Failure',fontsize=20)
		ax2.set_title(self.title,fontsize=24)
		ax2.set_ylim(0,window_max)
		ax2.yaxis.grid(which='major',linestyle='--')
		ax2.set_yticks(np.arange(0,window_max,step=round_up))
		
		fig.savefig(self.dirs +  self.title + '_compare_w_average.jpg')

	def move_to_Ndrive(self):
		## Check if they have N drive mapped 
		Ndrive_prefix = 'N:\\test_data\\'
		if os.path.exists(Ndrive_prefix):
			pass
		else:
			print('\n \n Ndrive is not mapped!!!! Data is not backed up')
		
		print('\n Starting the Ndrive copy now, grab a coffee or some beer cause this will take a minute \n')

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

	### These four things get done in this order no matter what
	h.create_bigdf()
	h.save_df_to_parquet()
	h.Meta_data_reader()
	h.plot_bigdf_moving_average()
	h.create_limitdf()
	h.mini_barplot()
	

	### Here on uses logic to identify file type
	if indicator == 0:
		while True:
			prompt = input("\n Which master plot do you want to compare with? (Input the number, s to skip) \n '1' : 'HackFPC_w_islands_limits.csv' \n")
			if prompt == 's':
				break
			elif prompt == '1':
				h.comparison_bar_plot_Hack(prompt)
				break
			else:
				print('\n Not an option! \n ') 
	elif indicator == 1 :
		print('\n CuS needs more data to have a master plot :( \n')
	elif indicator == 2:
		h.append_limit_df_to_master()
		h.master_scatter()
	else:
		print('UNRECOGNIZED Sample')

	
	h.move_to_Ndrive()
	h.close()
	print('HEAT STARS analysis complete. \n Files can be found in the folder you ran this and they are backed up on the Ndrive')
	sys.exit()

if __name__ == '__main__':
	main()