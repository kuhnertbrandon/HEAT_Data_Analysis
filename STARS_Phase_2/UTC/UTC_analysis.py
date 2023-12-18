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



#### Limit functions
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

		### Find serial in sea of lies
		delim_name = self.file_list[0]
		delim_file = delim_name.split('_')
		self.title = delim_file[-3]
		

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

		return self.title

	def pull_daq_channels(self):
		df = self.bigdf
		# get a list of all column headers
		headers = df.columns.tolist()
		# create a sublist of headers containing the substring 'daq'
		daq_headers = [header for header in headers if 'daq' in header]
		print(daq_headers)


		if len(daq_headers) == 48:
			self.daq_list = daq_headers
			self.channel_list = ['200D', '150D', '100D', '50D', '25D', '10D', '200C1', '150C1', '100C1', '50C1', '25C1', '10C1', '200B1', '150B1', '100B1', '50B1', '25B1', '10B1', '200A1', '150A1', '100A1', '50A1', '25A1', '10A1', '10A2', '25A2', '50A2', '100A2', '150A2', '200A2', '10B2', '25B2', '50B1', '100B2', '150B2', '200B2', '10C2', '25C2', '50C2', '100C2', '150C2', '200C2', '200E', '150E', '100E', '50E', '25E', '10E']
		else:
			print('\n See the channel list below \n ')
			print(daq_headers)

			print('\n You will now assign each channel by hand \n')

			user_chan_list = []
			while True:
				for i in range(1,(len(daq_headers) + 1)):
					prompt_repeating = input('\n What was on channel DAQ' + str(i) + '? \n Formats: 150A2  25B1  200C2  50D  150E  (n if channel was unused) \n')
					user_chan_list.append(prompt_repeating)
					print(user_chan_list)

				prompt2 = input('\n Does this channel list match your sample? (y) or (n) \n')
				if prompt2 == 'y':
					print('Moving On')
					break
				elif prompt2 == 'n':
					user_chan_list = []
					continue
				else:
					user_chan_list = []
					print('Invalid Input. Restarting ')


			#print('Lengths of lists')
			#print(len(daq_headers) == len(user_chan_list))
			if len(daq_headers) == len(user_chan_list):
				self.daq_list = daq_headers
				self.channel_list = user_chan_list
			else:
				print('DAQ channels do not align!!!! This is a code error')


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
		self.bigdf = self.bigdf.sort_values(by=['Time (Seconds)'])

			### At this point you have the raw data and its named, time to move it

		### Keep these loops separate to not mess up work flow
	
		raw_dirs = self.dirs + 'Raw\\'
		if os.path.exists(raw_dirs):
			pass
		else:
			os.makedirs(raw_dirs)


		for files in csv_files:
			shutil.move(files,raw_dirs + files)

	def move_pngs(self):
		png_files = glob.glob('**.png')


		if os.path.exists(self.dirs):
			pass
		else:
			os.makedirs(self.dirs)


		for files in png_files:
			shutil.move(files,self.dirs + files)


	def create_limitdf(self,coupon_type,rod_diameter,maker,material,coverlay,moduli):

		limit_columns = ['serial','coupon','date','manufacturer','coverlay','modulus_gpa','alloy','trace','physical_position','strain_p','Start_ohms','10_p_increase_cycles','30_p_increase_for_10_cycles','max_cycle_for_test'] #,'opens_prior_to_lowest','shorts_prior_to_lowest']
		limit_df=pd.DataFrame([],columns=limit_columns)


		print(self.daq_list)
		print(self.channel_list)

		title = self.title
		date = self.timestamp
		#daq_list = ['DAQ1','DAQ2','DAQ3','DAQ4','DAQ5','DAQ6','DAQ7','DAQ8']

		bend_max_cyc = self.bigdf['cycle'].iloc[-1]

		j=0
		for i in self.daq_list:
			smol_list = ['cycle',i]
			print(i)
			#print(smol_list)
			#print(self.bigdf.shape)

			smoldf = self.bigdf[smol_list].copy()

			### All skipping function
			if smoldf[i].iloc[0:10].isnull().values.all():
				print(str(i) + ' is empty! Skipping')
				j = j + 1
				continue


			### Check if open on start
			if smoldf.iloc[0] < 0.0 or smoldf.iloc[0]>110:
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

			#### Code for the improved 10 percent limit
			# p10up = res_start * 1.1
		 #    p10_df = smoldf[smoldf[i] > p10up]
		 #    p10for5_lim = lim10for5samp(p10_df)

		 #    if p10for5_lim == 'Did not reach limit':
		 #        df_opens1 = smoldf[smoldf['cycle'] <= bend_max]
		 #    else:
		 #        df_opens1 = smoldf[smoldf['cycle'] <= p10for5_lim]  
			# df_opens = df_opens1[df_opens1[i] >= 100]
			# opens_b4 = df_opens.shape[0]
			
			# df_shorts = df_opens1[df_opens1[i] <= res_start * 0.01]
			# shorts_b4 = df_shorts.shape[0]

			res10p_lim = res_start + res_start * 0.1
			
				
			compare10p = smoldf[smoldf[i] > res10p_lim].reset_index(drop=True)
			if len(compare10p) < 5:
				cycle_res10p = 'Did not reach limit'
			else:			
				cycle_res10p = compare10p['cycle'].iloc[4]
			compare10p = None
				
			
			row = pd.DataFrame([[title,coupon_type,date,maker,coverlay,moduli,material,self.channel_list[j],self.daq_list[j],strain,res_start,cycle_res10p,p30for10_lim,bend_max_cyc]],
							   columns=limit_columns )
			limit_df = pd.concat([limit_df,row]) #limit_df.append(row)
			j = j + 1


		### Add more columns to df
		limit_df['shape'] = limit_df['trace'].str.extract(r'([A-Z]+)')
		limit_df['width'] = limit_df['trace'].str.extract(r'(\d+)')

		limit_df['array'] = limit_df['trace'].str[-2:]

		def capitalize_letters(row):
			if row['shape'] == 'D' or row['shape'] == 'E':
				return row['shape']
			else:
				return row['array']

		limit_df['array'] = limit_df.apply(capitalize_letters, axis=1)



		limit_df.loc[limit_df['shape'] == 'A','shape'] = 'straight'
		limit_df.loc[limit_df['shape'] == 'B','shape'] = 'straight'
		limit_df.loc[limit_df['shape'] == 'C','shape'] = 'serpentine'
		limit_df.loc[limit_df['shape'] == 'E','shape'] = 'straight'
		limit_df.loc[limit_df['shape'] == 'D','shape'] = 'serpentine'


		limit_df['width'] = pd.to_numeric(limit_df['width'])
		limit_df['radius'] = limit_df['strain_p'].str[0:1]
		limit_df['radius'] = pd.to_numeric(limit_df['radius'])
		limit_df['channel'] = limit_df['physical_position'].str.extract(r'(\d+)')
		limit_df['channel'] = pd.to_numeric(limit_df['channel'])

		# save Limit df
		limit_name = self.title + '_limits.csv'

		limit_df.to_csv(self.dirs + limit_name,index=False)
		
		self.limit_name = limit_name
		
		self.limit_df = limit_df


		

	def append_limit_df_to_master(self):
		### Find master csv
		#npre = 'N:\\STARS2_test_data\\master\\'
		npre = 'N:\\users\\Brandon\\STARS2\\master\\'
		masname = 'STARS2_Unified_FPC_Bend_Master.csv'
		oldname = npre + 'old\\' +  self.mini_timestamp + masname

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



		### Moving Average Plot
		plt.rcParams['agg.path.chunksize'] = 10000
		fig, (ax1,ax2) = plt.subplots(2,figsize=(30,20))
		j = 0
		print(self.channel_list)
		for i in self.daq_list:

			if bigdf[i].iloc[0:10].isnull().values.all():
				print(str(i) + ' is empty! Skipping')
				j = j + 1
				continue


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


	def master_to_percentage_plt(self):
		df = self.master_df
		
		names = ['10 %','30 %']
		trace_widths = [10,25,50,100,150,200]
		
		df['shape'] = df['trace'].str.extract(r'([A-Z]+)')
		df['width'] = df['trace'].str.extract(r'(\d+)')
		df.loc[df['shape'] == 'A','shape'] = 'straight'
		df.loc[df['shape'] == 'C','shape'] = 'serpentine'
		df['width'] = pd.to_numeric(df['width'])
		
		for j in trace_widths:
			
			tdf = df[df['width'] == j]
			if tdf.size <2:
				continue
			
			dfst = tdf[tdf['shape'] == 'straight'].reset_index(drop=True)
			dfse =tdf[tdf['shape'] == 'serpentine'].reset_index(drop=True)
			dfs = [dfst,dfse]



			for name in names:
				fig, ax = plt.subplots(figsize=(20,15))
				if name == '10 %':
					x_col = '10_p_increase_cycles'
					save = '10p'
				elif name == '30 %':
					x_col = '30_p_increase_for_10_cycles'
					save = '30p_for_10'
				for i in dfs:
					dft = i.sort_values(by=[x_col]).reset_index(drop=True)
					dft_top = dft.index[-1]
					dft['percentage'] = np.round(dft.index/dft_top*100,2)
					ax.plot(dft[x_col],dft['percentage'],'--o',markersize=15)

				ax.legend(['Straight','Serpentine'],fancybox=True,framealpha=1,fontsize=22)

				ax.set_ylabel('Percentage Failed',fontsize=22)
				ax.set_title('Cycles vs Percent Population Failed '+ name + ': Trace Width: '+ str(j) +' (um) '+ self.pretty_timestamp,fontsize=28)
				ax.set_xlabel('Cycles',fontsize=22)
				ax.set_yticks(range(0,110,10))
				ax.yaxis.grid(which='major',linestyle='--')
				ax.tick_params(axis='both', which='major', labelsize=18)
				ax.set_xlim(left=0)
				fig.savefig(self.dirs + save + '_tw' +str(j) +'_increase_percentPop' + self.timestamp+'.jpg')

	def master_v_trace_width(self):
			df = self.master_df
		
			names = ['10 %','30 %']
			
			
			df['shape'] = df['trace'].str.extract(r'([A-Z]+)')
			df['width'] = df['trace'].str.extract(r'(\d+)')
			df.loc[df['shape'] == 'A','shape'] = 'straight'
			df.loc[df['shape'] == 'C','shape'] = 'serpentine'
			df['width'] = pd.to_numeric(df['width'])

			dfst = df[df['shape'] == 'straight'].reset_index(drop=True)
			dfse =df[df['shape'] == 'serpentine'].reset_index(drop=True)
			dfs = [dfst,dfse]

			for name in names:
				fig, ax = plt.subplots(figsize=(20,15))
				if name == '10 %':
					y_col = '10_p_increase_cycles'
					save = '10p'
				elif name == '30 %':
					y_col = '30_p_increase_for_10_cycles'
					save = '30p_for_10'
				for i in dfs:
					dft = i.sort_values(by=[y_col]).reset_index(drop=True)
					dft_top = dft.index[-1]
					dft['percentage'] = np.round(dft.index/dft_top*100,2)
					ax.scatter(pd.to_numeric(dft['width']),dft[y_col],s=150)


				ax.legend(['Straight','Serpentine'],fancybox=True,framealpha=1,fontsize=22)

				ax.set_ylabel('Cycles',fontsize=22)
				ax.set_title('Cycles vs Trace Width ' + name+  ' '+ self.pretty_timestamp,fontsize=28) #+ name+ ': '+ pretty_timestamp
				ax.set_xlabel('Trace Width (microns)',fontsize=22)
				ax.set_xticks(range(0,275,25))
				ax.xaxis.grid(which='major',linestyle='--')
				ax.tick_params(axis='both', which='major', labelsize=22)
				ax.set_xlim(0,250)
				ax.set_xlim(left=0)
				fig.savefig(self.dirs + save +'_cycles_v_tw' + self.timestamp+'.jpg')


	def read_parquet_file(self,parquet_file):
		dfp=pd.read_parquet(parquet_file)
		self.bigdf = dfp 
		self.title = parquet_file[0:12] # Morteza wants 14
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


		if os.path.exists(self.dirs):
			pass
		else:
			os.makedirs(self.dirs)

		try: 
			for files in png_files:
				shutil.move(files,self.dirs + files)
		except:
			pass

		try:
			for files in txt_files:
				shutil.move(files,self.dirs + files)
		except:
			pass

	def move_to_Ndrive(self):
		## Check if they have N drive mapped 
		Ndrive_prefix = 'N:\\STARS2_test_data\\'
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
	h.glob_search_csv()
	#glob_par = glob.glob('**.parquet')
	h.create_bigdf_new()
	h.save_df_to_parquet()
	#h.read_parquet_file(glob_par[0])
	h.pull_daq_channels()
	print('\n Input the answer in the parenthesis \n  \n YOU PROBABLY NEED TO CLOSE THE CHROME MONSTER \n \n')

	while True:
		prompt1 = input('\n What type of Sample is this? (Just select u, I do not want to refactor) \n (l) for 1L Unified Copper \n (u) Unified Coupon 2.0 \n (4) for 4L Copper Coupon \n')
		if prompt1 == 'l':
			s_type = '1L_copper_coupon'
			alloy = 'Cu_RA'
			manufacturer = 'Altaflex'
			break
		elif prompt1 == 'u':
			s_type = 'unified_coupon'
			prompt17 = input('\n Manufacturer? \n (c) Carlisle \n (a) Altaflex \n')
			if prompt17 == 'c':
				manufacturer = 'Carlisle'
			elif prompt17 == 'a':
				manufacturer = 'Altaflex'
			else:
				print('not an option')
				continue

			
			prompt18 = input('\n Alloy? \n (c) Cu_ED \n (o) Cu_RA_O2F \n (cn) CuNi3Si \n (cz) CuZn30 \n (cc) CuCrZr \n (cm) CuMgAgP \n (cs) CuSn6 \n')
			if prompt18 == 'c':
				alloy = 'Cu_ED'
				break
			elif prompt18 == 'o':
				alloy = 'Cu_RA_O2F'
				break
			elif prompt18 == 'cn':
				alloy = 'CuNi3Si'
				break
			elif prompt18 == 'cz':
				alloy = 'CuZn30'
				break
			elif prompt18 == 'cc':
				alloy = 'CuCrZr'
				break
			elif prompt18 == 'cm':
				alloy = 'CuMgAgP'
				break
			elif prompt18 == 'cs':
				alloy = 'CuSn6'
				break
			else:
				print('Try again there bud')
				continue


	while True:
		prompt11 = input('\n What size bend rod was used? \n')
		if prompt11 == '7':
			rod_d = prompt11
			break
		elif prompt11 == '3':
			rod_d = prompt11
			break
		elif prompt11 == '5':
			rod_d = prompt11
			break
		else:
			print('Only 3, 5, and 7 have been tested')

	while True:
		prompt21 = input('\n Coverlay? (y) or (n)\n')
		if prompt21 == 'y':
			c_lay = 'Y'
			break
		elif prompt21 == 'n':
			c_lay = 'N'
			break
		else:
			print('Try again there buddy')

	while True:
		prompt22= input('\n Dupont or Panasonic Polyimide? (d) or (p)\n')
		if prompt22 == 'd':
			modulus = 4.8
			break
		elif prompt22 == 'p':
			modulus = 7.1
			break
		else:
			print('Try again there buddy')



	# h.assign_channels(user_chan_list)
	
	#h.find_first_row()
	
	



	# Create and append limit	
	h.create_limitdf(s_type,rod_d,manufacturer,alloy,c_lay,modulus)
	h.append_limit_df_to_master()

	#h.plot_bigdf_moving_average() ### Can't do this, 
	#h.master_to_percentage_plt()
	#h.master_v_trace_width()

	h.move_pngs()
	#h.move_to_Ndrive()
	h.end()
					




if __name__ == '__main__':
	main()
