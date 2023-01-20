# Introduction
This research investigates how stimuli characteristics in SSVEP-based interfaces influence the signal-to-noise ratio (SNR) (dB). It used Experiment Builder to record the eye-tracking and eeg data. 2 Experiments are executed. Experiment 1 shows one stimulus across all the trials. This stimulus varies in shape, frequency, color, and size. Experiment 2 simulates the same stimulus in an 2X2 SSVEP-speller across the same frequencies, colors, and sizes. A comparison can be made between the two experiments to investigate the effect a SSVEP-speller environment has on the measured SNR.

One important aspect when executing this code is that there were a lot of conflicts between dependencies. This caused a lot of seperate anaconda environments.
# Experiments - Experimental Builder
The experiments itself are located in the Experimental Builder folder. The version of the application is v2.3.38.

# Output files Experimental Builder
One of the files that Experiment Builder outputs after the project are run in test modes to record the data are files like EXPERIMENT_1.txt and EXPERIMENT_2.txt. If the name part 'experiment' is not in capital letters change it so that it can be recognized in the python files. 

To create files like EXPERIMENT_1.csv, first a dupicate is made and renamed and then the file extension is changed from .txt of EXPERIMENT_1.txt to ".csv".

The .edf files can be modified with Data Viewer which can cut the trials.
To do this the trials can be cut between messages: "Frame to be displayed" to "TIMEOUT_TRIAL_1X1" or "TIMEOUT_TRIAL_2X2" meaning for EXPERIMENT_1X1 and "EXPERIMENT_2X2". After that the data can be converted to a sample report. In this research all the variables possible are saved in the output '.csv' files.




# The Code explained

Executed on Ubuntu 20.04.4 LTS

## x264 library to create videos in .avci format

    sudo apt-get update -y
    sudo apt-get apt-get remove -y x264 ffmpeg libx264-dev
    sudo apt-get install -y x264 libx264-dev 
    sudo apt-get install -y ffmpeg 

I read this tutorial to find the correct commands: https://www.swiftlane.com/blog/generating-mp4s-using-opencv-python-with-the-avc1-codec/

 ## Setting up the environment 
 
    conda create -n video_creater python==3.9.0

    conda activate video_creater

    conda install -c conda-forge opencv==4.6.0
  
  ## How the videos and photos of the trials are created 
  
  Now you can run ssvep_interface_video_creater_v1.py
  
  ## Setting up environment for the first 2 analysis scripts
  Unfortunately the packages used for creating the videos collide with the packages required for the following statistical analyses
  Let's create a new environment for the next script
  
    conda create -n  analysis python==3.9.0
  
    conda activate analysis
  
    pip install mne==1.2.3
    
    conda install -c conda-forge pandas==1.5.1
    
    conda install -c conda-forge opencv==4.6.0
    
    conda install -c conda-forge gcc=12.1.0
    
   ## Step 1: Processing with process_eeg_and_eye_tracking_v1.py
   This has to be executed for each experiment for each participant seperately.
   ![instruction_2](https://user-images.githubusercontent.com/27996213/213590999-fce106d1-fe70-4bf0-97df-f9254cb803c3.png)

   1. In line 1216 the variable folder_path has to be changed to the path containing the eye-tracking and EEG data in combination with the EXPERIMENT_X.csv. The X is here either 1 or 2.  An example of the folder structure can be seen in the picture above. The highlight folder/files are generated. The other files need to be present in a folder to generate these folders/files. Example for line 1216 is: path = /media/sjoerd/BackUp Drive/Thesis_project/participant data/raw data/pp1/Experiment 1
 
   Important: There is a difference between Linux and Windows in path which means one uses '/' and the other '\'. Thus, you might need to change that to get the code running in places where paths are seperated and are created.
   
   2. Then run the file with python

   3. This outputs Experiment_XXX_eye_tracking_processed_fixations.csv which contains the fixations. XXX is either 1X1 or 2X2 depending on the experiment. It also output the figures folder with visualizations (power-spectral density plots) of the raw eeg data of each trial. In the Experiment_XXX_eye_tracking folder there is an images folder that contains visualizations of the raw eye tracking data. It also contains a dropped_frames.png which shows the dropped frames per trial, and a heatmap that shows all the data across the trials as a heatmap. Lastly, it contains the processed_data.csv. This file contains all the metrics per trial such as max_snr, radius to center of stimulus, eye-tracking samples on target, etc. 
    
   ## Step 2: Prepare data for SNR extraction by renaming the processed_data.csv in 
   ![Instruction_data_snr](https://user-images.githubusercontent.com/27996213/213533984-aa621efe-9ee0-4c3c-b5fb-022bc985a41f.png)

   1. Rename the processed_data.csv to processed_data_exp_X_ppX.csv, such as processed_data_exp_1_pp2.csv.  This means experiment 1 participant 2.
   2. Place it in the folder within Data_SNR that is either called Experiment_1 or Experiment_2.
   3. Do this for all the participants for both experiments. See the photo as example.
   
    
   ## Step 3: Run process_SNR.py to combine all the data
   Make sure that the Data_SNR folder is empty except for the folders Experiment_1 and Experiment_2.
   1. On line 46, change the path to the Data_SNR path containg all the '.csv' files.
   2. Run the script with: python process_SNR.py
   
   It outputs a file named SNR_sorted_by_participant.csv located in the Data_SNR folder
   
   
   
   ## Step 4: Create environment for the rest of the analyses
   One of the necessary python libraries: pinguoin conflicts with the previous envirnoment and makes the previous scripts not executable anymore due to dependencies conflicts. Additionally, it uses a different version of matplotlib.
   
    conda create -n  analysis_snr python==3.9.0
  
    conda activate analysis_snr
  
    pip install mne==1.2.3
    
    conda install -c conda-forge pandas==1.5.1
    
    conda install -c conda-forge opencv==4.6.0
    
    conda install -c conda-forge pingouin==0.5.2
    
    conda install matplotlib
    
   ## Step 5: Run analyses scripts for the SNR
   The scripts here save everything in the SNR results
   This performs statistical analysis of the SNR and plot some boxplots and means andstandard deviations:
   1. In SNR_statistic_analyses_v1.py change line 384 by updating its path to the file: SNR_sorted_by_participant.csv.
   2. Run the file with python.
   
   It outputs the p values of the analyses and creates figures and boxplots.
    
   
   This creates extensive visualizations of the data:
   1. In SNR_statistic_4d_plot_v1.py change line 940 by updating its path to the file: SNR_sorted_by_participant.csv.
   2. Run the python file
   
   It creates boxplots and shows the interaction between multiple variables with respect to the SNR
   
   ## Step 6: Questionnaire analysis
   1. In analysis_of_questionnaire_v1.py update the path on line 136
   2. Run the script
   
   Outputs results_logger_questionnaire.csv in the same folder. This shows if the differences between the means were significant.
    
  
  
