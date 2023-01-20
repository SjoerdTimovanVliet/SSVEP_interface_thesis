import numpy as np
import pandas as pd
import glob
import sys

def filter_dataframe_exp(df_pp: pd.DataFrame, df_snr: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the dataframe per participant and per video and put it in another format for the analysis
    :param df_pp: dataframe with the snr data per participant
    :param df_snr: dataframe with the snr data per video
    :return: dataframe with the filtered data
    """

    pp_numbers = df_snr['pp'].unique()
    pp_numbers.sort()
    # loop over the files
    for folder in df_snr['Folder'].unique():
        # extract unique video names of the folder
        unique_video_names = df_snr[df_snr['Folder'] == folder]['Video'].unique()
        # loop over the unique video names and extract the data
        for video in unique_video_names:
            # extract the data for the video
            video_data = df_snr[(df_snr['Folder'] == folder) & (df_snr['Video'] == video)]
            # extract the settings of the video such as pixel surface, color, shape, frequency
            pixel_surface = video_data['pixel_surface'].unique()[0]
            color = video_data['color'].unique()[0]
            shape = video_data['shape'].unique()[0]
            frequency = video_data['frequency'].unique()[0]
            row = [folder, video, pixel_surface, color, shape, frequency]
            # loop over the pp numbers
            for pp_number in pp_numbers:
                # extract the data for the pp
                pp_data = video_data[video_data['pp'] == pp_number]
                # extract the snr data for the pp
                snr_data = np.array(pp_data['MAX_SNR'])[0]
                # append the snr data to the row
                row.append(snr_data)

            # append the row to the dataframe
            df_pp.loc[len(df_pp)] = row
            
    return df_pp

# main folder path
print("main folder path")
path = r"/media/sjoerd/BackUp Drive/Thesis_project/Data_SNR"
# find child folders
folders = glob.glob(path + "/*")
print(f" folders: {folders}")
# find all files in child folders
files = {}
# loop over the folders
for folder in folders:
    # remove the path from the folder name
    folder_name = folder.replace(path + "/", "")
    files[folder_name] = glob.glob(folder + "/*")

# create a dictionary to store the data
files_data = {}
# create a dataframe to store the data
headers_dataframe = ['Folder', 'File', 'pp', 'Video', 'pixel_surface', 'color', 'shape', 'frequency', 'MAX_SNR']
# create empty dataframe
df_snr = pd.DataFrame(columns=headers_dataframe)

for folder in files:
    print(f"folder: {folder}")
    print(f"files: {files[folder]}")
    # create a dictionary for the folder
    files_data[folder] = {}
    for file in files[folder]:
        # remove the path from the file name
        file_name = file.replace(path + "/" + folder + "/", "")
        # read the data
        data = pd.read_csv(file, header=None)
        # get headers from the first row of the data
        headers = data.iloc[0]
        # set the headers
        data.columns = headers
        # make tthe first row the headers
        data = data[1:]
        # remove the first 4 trials
        data = data[4:]
        if 'Experiment_1' == folder:
            assert(len(data['TRIAL_INDEX']) == 324)
        elif 'Experiment_2' == folder:
            assert(len(data['TRIAL_INDEX']) == 432)
        else:
            print(f"Error: folder {folder} not recognized")
            sys.exit()

        # add the data to the dictionary
        files_data[folder][file_name] = {}
        files_data[folder][file_name]['data'] = data
        # get  the VIDEO_NAME column
        video_name = data['VIDEO_NAME']
        # find the unique video names
        unique_video_names = video_name.unique()

        if 'Experiment_1' == folder:
            unique_video_names_experiment_1 = unique_video_names
        elif 'Experiment_2' == folder:
            unique_video_names_experiment_2 = unique_video_names
            
        #pdb.set_trace()
        files_data[folder][file_name]['unique_video_names'] = np.array(unique_video_names)
        # extract for each video_name the data
        for video in unique_video_names:
            # get the data for the video
            video_data = data[data['VIDEO_NAME'] == video]
            # extract the SNR data
            video_data_snr = video_data['MAX_SNR']
            # add the data to the dictionary
            files_data[folder][file_name][video] = video_data
            files_data[folder][file_name]['SNR'] = video_data_snr
            # extract the settings of each video
            pixel_surface = int(video_data['PIXEL_SURFACE'].unique()[0])
            color = video_data['COLOR'].unique()[0]
            shape = video_data['SHAPE'].unique()[0]
            frequency = int(video_data['FREQUENCY'].unique()[0])         
            # extract the pp number from the file name
            pp_number = int(file_name.split("_")[-1].split(".")[0].replace("pp", ""))
            # create the row for the dataframe
            row_dataframe = [folder, file_name, pp_number, video, pixel_surface, color, shape, frequency, np.array(video_data_snr).astype(float)]

            # add the row to the dataframe
            df_snr.loc[len(df_snr)] = row_dataframe
      
# save the dataframe
df_snr.to_csv(path + "/SNR.csv", index=False)

# create empty dataframe
# generate the dataframe in the format 'Folder', 'Video'. 'pixel_surface', 'color', 'shape', 'frequency' followed by the SNR data for each pp
headers_dataframe = ['Folder', 'Video', 'pixel_surface', 'color', 'shape', 'frequency']
# add the pp numbers to the headers
pp_number = df_snr['pp'].unique()

for pp in pp_number:
    headers_dataframe.append(f'pp{pp}')
# create empty dataframe
df_sorted_by_participant = pd.DataFrame(columns=headers_dataframe)

df_pp = filter_dataframe_exp(df_sorted_by_participant, df_snr)
# save the dataframe to a csv file named 'SNR_sorted_by_participant.csv'
df_pp.to_csv(path + "/SNR_sorted_by_participant.csv", index=False)


