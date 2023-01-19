import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def process_eyes(pXX):
    #%%Handy function to group all consecutive numbers in an array. Handy to identift NaN's, fixations and saccades.
    def group_consecutives(vals, step=1):
        """Return list of consecutive lists of numbers from vals (number list)."""
        run = []
        result = [run]
        expect = None
        for v in vals:
            if (v == expect) or (expect is None):
                run.append(v)
            else:
                run = [v]
                result.append(run)
            expect = v + step
        return result
    #%% Interpolate, clean and filter XYP data. filter_data(pXX). Parses pXX to xf,yf,and pf format.
    def filter_data(pXX):
        extra=100*2
        mfi=100*2 #time interval of the median filter, samples 
        t={}#initialize x filtered_data
        xf={}#initialize x filtered_data
        yf={}#initialize y filtered_data
        pf={}#initialize y filtered_data
        
        for i, trial in enumerate(pXX):
            
            t[i]=pXX[i]['time']#identify time vector
            x = pXX[i]['AVERAGE_GAZE_X']#identify x vector
            # remove any '.' entries to np.NaN
            x = x.replace('.', np.nan)
            # convert to float
            x = x.astype(float)

            y = pXX[i]['AVERAGE_GAZE_Y']#identify y vector
            # remove any '.' entries to np.NaN
            y = y.replace('.', np.nan)
            # convert to float
            y = y.astype(float)
            
            # check if the pupil data is available by checking if entries any entries are not '.'
            left_p_array = np.array(pXX[i]['left_p'])
            right_p_array = np.array(pXX[i]['right_p'])
            # create empty array for the average pupil size
            average_pupil_size = np.empty(len(left_p_array))
            for j in range(0, len(left_p_array)):
               
                if left_p_array[j] == '.':
                    left_p_array[j] = right_p_array[j]
                if right_p_array[j] == '.':
                    right_p_array[j] = left_p_array[j]
            
                # check if the pupil data is available by checking if entries any entries are not '.'
                if left_p_array[j] != '.' and right_p_array[j] != '.':
                    average_pupil_size[j] = float(float(left_p_array[j]) + float(right_p_array[j]))/2
                elif left_p_array[j] == np.nan:
                    average_pupil_size[j] = right_p_array[j]
                elif right_p_array[j] == np.nan:
                    average_pupil_size[j]= left_p_array[j]
                else:
                    average_pupil_size[j] = np.nan
 
            # overwrite the column average_p with the new average pupil size
            pXX[i]['average_p'] = average_pupil_size

            # write average pupil pupil size to average_p
            p = pXX[i]['average_p']
            # convert to float
            p = p.astype(float)
   
            #reset indexes:
            x.reset_index(drop=True,inplace=True)   
            y.reset_index(drop=True,inplace=True)
            p.reset_index(drop=True,inplace=True)
            #filter x and y for extreme values, remove extreme values.
            x[x<-50]=np.NaN
            x[x>1920+50]=np.NaN
            y[y>1080+50]=np.NaN
            y[y<-50]=np.NaN
            #identify where vectors of x and y contain NaNs
            nan_x=x[x.isna()].index.values #get all indexvalues of X NaNs.
            # nan_y=y[y.isna()].index.values #get all indexvalues of Y NaNs.
            nan_groups=group_consecutives(nan_x) #get all the individual groups of NaNs
            if len(nan_groups)> 1: #if there are indeed NaNS in the array
                nan_range=[]#initialize vectors with all the nan_groups+extras
                for nan in range(0,len(nan_groups)): 
                    #append NaN from nan_groups with +/- extra NaNs to remove edges of artefacts and blinks
                    nan_range.append([*range(nan_groups[nan][0]-extra,nan_groups[nan][-1]+extra)]) 
                    if nan_range[nan][0] <= 0: #if nan_range on index nan goes below 0:
                        nan_range[nan]=[*range(0,nan_groups[nan][-1]+extra)] #then start from 0
                    if nan_range[-1][-1] >= len(x): #also if the +extra goes beyond length of the vector:
                        nan_range[nan]=[*range(nan_groups[nan][0]-extra, len(x))] #create range that goes to len(x)
                    x[nan_range[nan]]=np.nan #remove edges of NaN with +/- extra NaNs. (to remove artefacts)
                    y[nan_range[nan]]=np.nan #remove edges of NaN with +/- extra NaNs. (to remove artefacts)
                    p[nan_range[nan]]=np.nan #remove edges of NaN with +/- extra NaNs. (to remove artefacts)
            #if the data starts with NaNs, then remove first part of vector:
            nanextra=group_consecutives(x[x.isna()].index.values) #calculate new vectors with the added nans around gaps/blinks
            if len(nanextra)>1: #if there are nans in the vector
                if nanextra[0][0]==0: #if the first index of the new nan vector is 0
                    # t[trial].drop(nan_range[0],inplace=True)
                    x.drop(nanextra[0],inplace=True)
                    y.drop(nanextra[0],inplace=True)
                    p.drop(nanextra[0],inplace=True)
            #interpolate data linearly if there are gaps/nans in the data
            x.interpolate(method='linear',inplace=True)
            y.interpolate(method='linear',inplace=True)
            p.interpolate(method='linear',inplace=True)
            #filter data with median filter
            x=x.rolling(mfi,min_periods=1).median() 
            y=y.rolling(mfi,min_periods=1).median() 
            p=p.rolling(mfi,min_periods=1).median() 
            #fill filtered data vectors
            xf[i]=x#fill x filtered_data
            yf[i]=y#fill y filtered_data
            pf[i]=p#fill p filtered_data
        return t,xf,yf,pf


    #%%Calculate gazeSpeeds and determine fixations and saccades
    def eye_data(xf,yf,pf,pXX):
        # calculate gazeSpeeds and extract fixations and saccades
        fixations={}
        saccades={}
        for i, trial in enumerate(pXX):
      
            xdiff=xf[i].diff(periods=1)#calulcate diff values for x and y
            ydiff=yf[i].diff(periods=1)#calulcate diff values for x and y
            gazeSpeed = 2000*np.sqrt(xdiff**2+ydiff**2)#data shot at 2000fps, 2
            try:
                gazeSpeed = pd.Series(savgol_filter(gazeSpeed.loc[1::],41,2,mode='nearest'),index=xdiff.index.values[1::]) #= order of the Savitzky-Golay filter, 41 = frame lenght of the % order of the Savitzky-Golay filter (in samples)
            except:
                pass
            sac=gazeSpeed>2000#identify saccades
            fix=gazeSpeed<2000#identify fixations
            fixations[i]=group_consecutives(fix[fix].index.values) #get indexes of all the fixations
            saccades[i]=group_consecutives(sac[sac].index.values) #get indexes of all the saccades

        return fixations, saccades
    #%% extract fixation duration, fixation frequency, saccade length, saccade frequency, saccade_amplitude, fixation amplitude (pursuit movement)
    def eye_stats(xf,yf,fixations, saccades,minfd,minsd,maxsd,pXX):
        fd={}#fixation duration
        sl={}#Saccade length
        ff={}#fixation frequency
        sf={}#saccade frequency
        sa={}#saccade amplitude = length of saccade 
        fa={}#fixation amplitude = length of pursuit movements
        for i, trial in enumerate(pXX):
            fd[i]=[]#fixation duration
            sl[i]=[]#Saccade length
            sa[i]=[]#saccade amplitude
            fa[i]=[]#fixation amplitude = length of pursuit movements
            for fixation in range(0,len(fixations[i])): #look at each fixation:
                if len(fixations[i][fixation])>minfd: #fixation minimum duration longer then 40ms (80/2)
                    fd[i].append(len(fixations[i][fixation]))#fixation duration
                    fa[i].append(np.sqrt((xf[i][fixations[i][fixation][0]]-xf[i][fixations[i][fixation][-1]])**2+(yf[i][fixations[i][fixation][0]]-yf[i][fixations[i][fixation][-1]])**2))#fixation amplitude
            for saccade in range(0,len(saccades[i])):
                if len(saccades[i][saccade])>minsd & len(saccades[i][saccade])<maxsd: #saccade maximum duration shorter then 300ms (80/2)
                    sl[i].append(len(saccades[i][saccade]))#Saccade length
                    sa[i].append(np.sqrt((xf[i][saccades[i][saccade][0]]-xf[i][saccades[i][saccade][-1]])**2+(yf[i][saccades[i][saccade][0]]-yf[i][saccades[i][saccade][-1]])**2))#saccade amplitude
            ff[i]=len(fd[i])#fixation frequency
            sf[i]=len(sl[i])##saccade frequency
        return fd,sl,ff,sf,sa,fa
    #execute above functions
    minfd=40*2 #minimum fixation duration (ms)
    minsd=10*2; #minimum allowable saccade duration (ms)
    maxsd=150*2; #maximum allowable saccade duration (ms)
    t,xf,yf,pf=filter_data(pXX)
    fixations, saccades=eye_data(xf,yf,pf,pXX)
    fd,sl,ff,sf,sa,fa= eye_stats(xf,yf,fixations, saccades,minfd,minsd,maxsd,pXX)
    
    return t,xf,yf,pf,fixations, saccades,fd,sl,ff,sf,sa,fa