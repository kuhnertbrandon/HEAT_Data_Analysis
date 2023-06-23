import glob
import pandas as pd
import numpy as np
import io
import math
import matplotlib.pyplot as plt
import multiprocessing
import os
from datetime import datetime


# class HEAT_Analysis():
# 	### Create a global timestamp
global timestamp
global indicator
now = datetime.now()
timestamp = now.strftime("_%Y_%m_%d_%H_%M")

def glob_search_csv():
	glob_string = '**.csv' # this generic one will just find all csv's 
	file_list = glob.glob(glob_string,recursive = True)
	title = file_list[0][0:13]
	if 'Hack' in title:
		indicator = 0
	else:
		indicator = 1
	dirs = title +'\\'
	if os.path.exists(dirs):
		pass
	else:
		os.makedirs(dirs)

	print('\n \n')
	print(file_list)
	return file_list,title,dirs


def find_first_row(file_list):
	### This function finds the start of the data, you want to run this separate to not get an IOpub error
	search_word = 'Time ('
	start_line = None
	with open(file_list[0] , 'r') as f:
		lines = f.readlines()
		j = 0
		for line in lines:
			if search_word in line:
				start_line = lines.index(line)
				
			if start_line is not None:
				break
				
			j = j+1
			if j > 150:
				break
		   
	return start_line


def create_bigdf(file_list,start_line):
	# Initialize an empty dictionary to store the dataframes

	csv_files = file_list
	
	bigdf = pd.DataFrame()
	# Loop through each CSV file and convert it to a pandas dataframe
	for file in csv_files:
		
		 ### Open and delete first 4 rows
		with open(file , 'r') as f:
			data = f.readlines()[start_line:]
		
		#short_name = file[-25:-4] ### Standardize on name
		#filename = 'Cleaned_' + short_name + '.csv'
		# Then read from the next rows without joining the data
		df = pd.read_csv(io.StringIO('\n'.join(data)))
		#df.columns=df.columns.str.split(' ').str[0]

		bigdf = pd.concat([bigdf,df])
		### At this point you have the raw data and its named

	return bigdf

def create_limitdf(bigdf,title,dirs):

	limit_df=pd.DataFrame([],columns=['Sample','Physical Positon','Start (ohms)','> 6 ohms','> 10 ohms','> 50 ohms','> 100 ohms'])


	title = title
	daq_list = ['DAQ1','DAQ2','DAQ3','DAQ4','DAQ5','DAQ6','DAQ7','DAQ8']
	if 'Hack' in title:
		hack_labels = ['MSW','100D_1','100D_2','MSWS','CPWS','85D_1','85D_2','CPW']
		indicator = 0
	else:
		hack_labels = ['Sample 1','Sample 2','Sample 3','Sample 4','Sample 5','Sample 6','Sample 7','Sample 8']
		indicator = 1 


	j=0
	for i in daq_list:
		# Create a rolling average
		res_raw = bigdf[i]
		
		if res_raw.iloc[0] < 0.0 or res_raw.iloc[5]>2000:
			continue
		

		compare = bigdf[bigdf[i] > 6].reset_index()
		compare2 = bigdf[bigdf[i] > 10].reset_index()
		compare3 = bigdf[bigdf[i] > 50].reset_index()
		compare4 = bigdf[bigdf[i] > 100].reset_index()
		
		res_start = bigdf[i].iloc[0]					### Need to fix so it finds 5 in a row not just first 5
		cycle_6 = compare['Cycle'].iloc[4]
		cycle_10 = compare2['Cycle'].iloc[4]
		cycle_50 = compare3['Cycle'].iloc[4]
		cycle_100 = compare4['Cycle'].iloc[4]

		
		row = pd.DataFrame([[hack_labels[j],daq_list[j][-1],res_start,cycle_6,cycle_10,cycle_50,cycle_100]],
						   columns=['Sample','Physical Positon','Start (ohms)','> 6 ohms','> 10 ohms','> 50 ohms','> 100 ohms'])
		limit_df = limit_df.append(row)
		j = j + 1

	# save Limit df
	limit_name = title[0:14] + '_limits.csv'
	limit_df.to_csv(dirs + limit_name,index=False)
	return limit_name, limit_df

def append_limit_df_to_master(title,limit_df):
	#

	if 'Hack' in title:
		path_to_master = N_prefix 
	elif 'LM' in title:
		path_to_master = N_prefix


def plot_bigdf_moving_average(bigdf,title,dirs):
	mov_avg = int(np.round(len(bigdf)/1000,0))
	if mov_avg < 3:
		mov_avg = 3
	graph_title = title + ': '
	daq_list = ['DAQ1','DAQ2','DAQ3','DAQ4','DAQ5','DAQ6','DAQ7','DAQ8']

	if 'Hack' in title:
		hack_labels = ['MSW','100D_1','100D_2','MSWS','CPWS','85D_1','85D_2','CPW']
		indicator = 0
	else:
		hack_labels = ['Sample 1','Sample 2','Sample 3','Sample 4','Sample 5','Sample 6','Sample 7','Sample 8']
		indicator = 1 


	### Moving Average Plot
	fig, (ax1,ax2) = plt.subplots(2,figsize=(30,20))
	j = 0
	for i in daq_list:


		# Create a rolling average
		res_raw = bigdf[i]
		res_avg = res_raw.rolling(window=mov_avg).mean()
		if res_raw.iloc[0] < 0.0 or res_raw.iloc[5]>2000:
			continue
		elif indicator == 0:
			ax1.plot(bigdf['Cycle'],res_avg,label=hack_labels[j])
			ax2.plot(bigdf['Cycle'],res_raw,label=hack_labels[j])
		else: 
			sample_length = 50
			res_norm = res_raw/((sample_length + sample_length*bigdf['% Displacement'])/10)
			ax1.plot(bigdf['Cycle'],res_avg,label=hack_labels[j])
			ax2.plot(bigdf['Cycle'],res_raw,label=hack_labels[j])
		j=j+1

	if indicator ==0:
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
	
	fig.savefig(dirs + title + ' Resistance_cycle_plot' + timestamp + '.jpg')
	print('\n Raw cycle plot created')

	

def save_df_to_parquet(df_in,title,dirs):
	df_in.to_parquet(dirs + title + '_Reliability_'+ timestamp +  '.parquet',engine='pyarrow')

def master_scatter(limit_df,new_df,x_ax,y_ax): ## On hold
 
	column_list = df.columns()
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
	plt.scatter(df[x_ax][-4:-2],df[y_ax][-4:-2],color='blue')
	plt.scatter(df[x_ax][-2:],df[y_ax][-2:],color='red')

def comparison_bar_plot_Hack(old_path,new_df,title,dirs):
	## Call N-drive
	old_df = pd.read_csv(old_path)

	
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
	print(vals_max)
		
	round_up = 10000
	window_max = np.round((vals_max/round_up),0)*round_up+round_up
	print(window_max)
	
	fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
	num_labels = len(labels_1)
	colors = plt.cm.viridis(np.linspace(0,1,num_labels))
	
	ax1.bar(labels_1,values_1,color=colors)
	ax1.set_ylabel('Cycles at Failure',fontsize=20)
	ax1.set_title(old_path[-24:-4],fontsize=24)
	ax1.set_ylim(0,window_max)
	ax1.yaxis.grid(which='major',linestyle='--')
	ax1.set_yticks(np.arange(0,window_max,step=round_up))
	
	
	ax2.bar(labels_2,values_2,color=colors)
	ax2.set_ylabel('Cycles at Failure',fontsize=20)
	ax2.set_title(title,fontsize=24)
	ax2.set_ylim(0,window_max)
	ax2.yaxis.grid(which='major',linestyle='--')
	ax2.set_yticks(np.arange(0,window_max,step=round_up))
	
	fig.savefig(dirs +  title + '_compare_w_average.jpg')

def move_to_Ndrive(title,instrument,files_to_move):
	## Check if they have N drive mapped 
	Ndrive_prefix = 'N:\\test_data\\'
	if os.path_exist(Ndrive_prefix):
		pass
	else:
		print('\n \n Ndrive is not mapped!!!! Data is not backed up')
	
	

	if 'Cu' in title:
		n_path = Ndrive_prefix + 'CuS\\' + title
	elif 'LM' in title:
		n_path = Ndrive_prefix + 'Liquid_Metal\\' + title
	else:
		n_path = Ndrive_prefix + 'Alloys\\' + title





######################################################################
def main():
	### Intialize Ndrive folder to save
	files_for_Ndrive = []
	current_direct = os.getcwd()

	h = HEAT_Analysis()

	file_list,title,dirs = glob_search_csv()
	start_line = find_first_row(file_list)
	

	bigdf = create_bigdf(file_list,start_line)

	
	#plot_bigdf_moving_average(bigdf,title,dirs)
	save_df_to_parquet(bigdf,title,dirs)
	limit_name,limit_df = create_limitdf(bigdf,title,dirs)
	old_path = current_direct + '\\master\\HackFPC_f5avg_limits.csv'
	comparison_bar_plot_Hack(old_path,limit_df,title,dirs)



if __name__ == '__main__':
	main()