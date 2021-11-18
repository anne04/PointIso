from __future__ import division
from __future__ import print_function
import math
import csv
#from time import time
import pickle
import numpy as np
#from sklearn import metrics
#from collections import deque
from collections import defaultdict
#import sys
#import copy
#import scipy.misc
import scipy.stats
#import bisect
import gzip
import matplotlib.pyplot as plt

#import seaborn as sns

isotope_gap=np.zeros((10))
isotope_gap[0]=0.01
isotope_gap[1]=1.00
isotope_gap[2]=0.50
isotope_gap[3]=0.33
isotope_gap[4]=0.25
isotope_gap[5]=0.20
isotope_gap[6]=0.17
isotope_gap[7]=0.14
isotope_gap[8]=0.13
isotope_gap[9]=0.11


RT_window=15
mz_window=211
frame_width=11
mz_resolution=2
total_class=10 # charge
RT_unit=0.01
mz_unit=0.01
fc_size= 4
total_frames_hor=6
num_class=total_frames_hor # number of isotopes to report
state_size = fc_size
num_neurons= num_class #mz_window*RT_window
truncated_backprop_length = 6
mappath='/data/fzohora/dilution_series_syn_pep/LC_MS/'
datapath='/data/fzohora/dilution_series_syn_pep/'      #'/data/fzohora/water/' #'/media/anne/Study/study/PhD/bsi/update/data/water/'  #
dataname=['130124_dilA_1_01','130124_dilA_1_02', '130124_dilA_1_03', '130124_dilA_1_04',
'130124_dilA_2_01','130124_dilA_2_02', '130124_dilA_2_03', '130124_dilA_2_04','130124_dilA_2_05', '130124_dilA_2_06', '130124_dilA_2_07', 
'130124_dilA_3_01','130124_dilA_3_02','130124_dilA_3_03','130124_dilA_3_04','130124_dilA_3_05','130124_dilA_3_06','130124_dilA_3_07',
'130124_dilA_4_01','130124_dilA_4_02','130124_dilA_4_03','130124_dilA_4_04','130124_dilA_4_05','130124_dilA_4_06','130124_dilA_4_07',
'130124_dilA_5_01','130124_dilA_5_02', '130124_dilA_5_03', '130124_dilA_5_04', 
'130124_dilA_6_01','130124_dilA_6_02', '130124_dilA_6_03', '130124_dilA_6_04',
'130124_dilA_7_01','130124_dilA_7_02', '130124_dilA_7_03', '130124_dilA_7_04',
'130124_dilA_8_01','130124_dilA_8_02', '130124_dilA_8_03', '130124_dilA_8_04',
'130124_dilA_9_01','130124_dilA_9_02', '130124_dilA_9_03', '130124_dilA_9_04',
'130124_dilA_10_01','130124_dilA_10_02', '130124_dilA_10_03', '130124_dilA_10_04', 
'130124_dilA_11_01', '130124_dilA_11_02', '130124_dilA_11_03', '130124_dilA_11_04', 
'130124_dilA_12_01', '130124_dilA_12_02', '130124_dilA_12_03', '130124_dilA_12_04'] 
#for test_index in range (int(sys.argv[1]), int(sys.argv[2])):
result_all=np.zeros((len(dataname), 12))
threshold_score=10.0
percent_feature=20
min_RT=10.00
avg_sen=np.zeros((10))
avg_set_count=0

filename ='/data/fzohora/dilution_series_syn_pep/feature_list/mascot/peptides_in_the_dilution_series.csv'
peptide_list = [] 
csvfile=open(filename, 'r')
csvreader = csv.reader(csvfile)     
for row in csvreader:
    peptide_list.append(row)
csvfile.close()

spiked_peptide_dict=dict()#defaultdict(list)
for i in range (1, 116):
   spiked_peptide_dict[peptide_list[i][0]]='potato' #.append('potato') #108
for i in range (1, 159):
   spiked_peptide_dict[peptide_list[i][1]]='human' #.append('human') #142
#for i in range (1, 30):
#   spiked_peptide_dict[peptide_list[i][4]].append('background') #2425

#2584 proteins in total
 
pointIso_human_peptide_quantity=defaultdict(list)
pointIso_potato_peptide_quantity=defaultdict(list)
pointIso_background_peptide_quantity=defaultdict(list)
#potato_detected_dict=dict()
#human_detected_dict=dict()
#background_detected_dict=dict()
for test_index in range (0, 57):
#    print(dataname[test_index])
    ###########################################################################            
    for runtime in range (0, 1): #
        threshold_score=25
        f=open(datapath+'feature_list/mascot/'+dataname[test_index]+'_mascot_db_search', 'rb')
        peptide_mascot=pickle.load(f)
        f.close()    
        temp_peptide_mascot=[]
        temp_peptide_mascot.append(peptide_mascot[0])
        for i in range (1, len(peptide_mascot)):
            if float(peptide_mascot[i][4])<=threshold_score:
                continue
            if round(float(peptide_mascot[i][7]), 2) < min_RT:#  or round(float(peptide_mascot[i][2]), mz_resolution)<min_mz: #or round(float(peptide_mascot[i][2]), mz_resolution)>800 or 
                continue
            temp_peptide_mascot.append(peptide_mascot[i])
        peptide_mascot=temp_peptide_mascot
        temp_peptide_mascot=0


        peptide_mascot[0].append('type')
        count=np.zeros((6))
        potato_detected_dict=dict()
        human_detected_dict=dict()
        background_detected_dict=dict()
        for i in range (1, len(peptide_mascot)):
            if peptide_mascot[i][5] in spiked_peptide_dict:
                peptide_mascot[i].append(spiked_peptide_dict[peptide_mascot[i][5]])
                if spiked_peptide_dict[peptide_mascot[i][5]]=='potato':
                    count[0]=count[0]+1
                    potato_detected_dict[peptide_mascot[i][5]]='found'
                elif spiked_peptide_dict[peptide_mascot[i][5]]=='human' :
                    count[1]=count[1]+1
                    human_detected_dict[peptide_mascot[i][5]]='found'
            else:
                peptide_mascot[i].append('background')
                background_detected_dict[peptide_mascot[i][5]]='found'
                count[5]=count[5]+1  
                
        total_potato=len(potato_detected_dict.keys())
        total_human=len(human_detected_dict.keys()) 
        total_background=len(background_detected_dict.keys()) 
        
        total_report=np.zeros((1, 5)) # 0 = openMS, 1=maxQuant, 2=dino, 3=DeepIso, 4= peaks
        ftr_matched_auc=np.zeros((len(peptide_mascot), 5)) # 0 = openMS, 1=maxQuant, 2=dino, 3=DeepIso, 4= peaks
        ########################################################################
        detected_peptide=np.zeros((len(peptide_mascot), 7)) # 0 = our, 1 = peaks, 2 = maxquant, 3= charge by Peaks, 4=peaks id, 5=dino, 6=openMS

#        f=gzip.open(datapath+'/feature_list/deepIsoV2_'+dataname[test_index]+'_featureTable_v6r1_cv5_ev2r6b_merged_auc','rb') #[-2.4,1] - potato, [-1,2.5, bad, peak is moved right to 1] -human
#        f=gzip.open(datapath+'/feature_list/deepIsoV2_'+dataname[test_index]+'_featureTable_v6r1_cv5_ev2r6b_merged_auc_NO_mztolerance_fullwidth_fullRT','rb') #[-2.4,1]-potato; [-1.5,3] human   
#        f=gzip.open(datapath+'/feature_list/deepIsoV2_'+dataname[test_index]+'_featureTable_v6r1_cv5_ev2r6b_merged_auc_fullwidth_fullRT','rb') #[-1.5,3.0]human, [-2.6,1.0] potato
#        f=gzip.open(datapath+'/feature_list/deepIsoV2_'+dataname[test_index]+'_featureTable_v6r1_cv5_ev2r6b_merged_auc_NO_mztolerance_fullwidth','rb') # [-1.15, 2.44,peak displaced] human-SRM (2)     
#        f=gzip.open(datapath+'/feature_list/deepIsoV2_'+dataname[test_index]+'_featureTable_v6r1_cv5_ev2r6b_merged_auc_exact_mz','rb') #[-1.5, 0]-potato-SRM; [0,1.7] human-SRM  (3) 
        f=gzip.open(datapath+'/feature_list/deepIsoV2_'+dataname[test_index]+'_featureTable_v6r1_cv5_ev2r6b_merged_auc_exact_mz_fullRT','rb') #[-1.43,2.44] human-SRM 
        feature_table,auc_list=pickle.load(f)
        f.close()     
        count=0
        mz_list=list(feature_table.keys())
        for i in range (0, len(mz_list)):
#            print(i)
            ftr_list=feature_table[mz_list[i]]
            for f in range (0, len(ftr_list)):
                ftr=ftr_list[f]
                score_ftr=max(ftr[len(ftr)-1][1])
                if (np.argmax(ftr[len(ftr)-1][1])==1 and score_ftr<.80) or (len(ftr)-1==1 and score_ftr<.50) or score_ftr<.30: # or # : #  (len(ftr)-1)==1 and 
                    continue
                count=count+1
        
#        print(count)
        total_report[0, 3]=count

        

        conf=100
        ####################################################
        potato_detected_dict=dict()
        human_detected_dict=dict()
        background_detected_dict=dict()
        RT_tolerance=0.2 # dino is same
        found_ftr=0
        detected_peptide[:, 0]=0
        ftr_matched_auc[:, 3]=0
        total_feature=0
        found_score=[]
        detected_ftr_list=[]
        detected_ftr_list.append([])
        for i in range (1, len(peptide_mascot)):
            detected_ftr_list.append([])
            found=0
            mz_exact=round(float(peptide_mascot[i][2]), mz_resolution)
            total_feature=total_feature+1           
            mz_range=[]
            mz_range.append(mz_exact)    
            tolerance_mz=0.01 #(mz_exact*10.0)/10**6  # dinosaur = 0.005 
            mz_range.append(round(mz_exact-tolerance_mz, mz_resolution))
            mz_range.append(round(mz_exact+tolerance_mz, mz_resolution))
            pep_seq_key=str(test_index)+'-'+str(mz_exact)+str(round(float(peptide_mascot[i][7]), 2))+'|'+str(int(peptide_mascot[i][3]))+peptide_mascot[i][5]
            for j in range (0, len(mz_range)):
                mz=mz_range[j]
                if mz in feature_table:
                    ftr_list=feature_table[mz]
                    for k in range (0, len(ftr_list)):
                        ftr=ftr_list[k]
#                        if ftr[len(ftr)-1][1]<auc_list[(conf*(len(auc_list)-1))//100]:
#                            continue
                        ftr_z=int(ftr[len(ftr)-1][0])    
                        peak_RT=ftr[0][1][0] 
                        score_ftr=max(ftr[len(ftr)-1][1])
                        if (np.argmax(ftr[len(ftr)-1][1])==1 and score_ftr<.80) or (len(ftr)-1==1 and score_ftr<.50) or score_ftr<.30: # or (len(ftr)-1==1 and score_ftr<.50): #  (len(ftr)-1)==1 and 
                            continue
                        if (round(float(peptide_mascot[i][7])-RT_tolerance, 2) <= peak_RT) and (peak_RT<=round(float(peptide_mascot[i][7])+RT_tolerance, 2)) and ftr_z==int(peptide_mascot[i][3]):
                            found=1
                            found_ftr=found_ftr+1
                            detected_peptide[i, 0]=1
                            if peptide_mascot[i][8]=='potato':
                                potato_detected_dict[peptide_mascot[i][5]]='found'
                                pointIso_potato_peptide_quantity[pep_seq_key].append(ftr[len(ftr)-1][2])
                            elif peptide_mascot[i][8]=='human':
                                human_detected_dict[peptide_mascot[i][5]]='found'
                                pointIso_human_peptide_quantity[pep_seq_key].append(ftr[len(ftr)-1][2])
                            else:
                                background_detected_dict[peptide_mascot[i][5]]='found'
                                pointIso_background_peptide_quantity[pep_seq_key].append(ftr[len(ftr)-1][2])
                                
#                            break

#                    if found==1:
#                        break
#                        

#    print('%s, %d, %g, %g, %d, %g, %g, %d'%(dataname[test_index], len(list(human_detected_dict.keys())),  (len(list(human_detected_dict.keys()))/158)*100, 
#    (total_human/158)*100, len(list(potato_detected_dict.keys())),  (len(list(potato_detected_dict.keys()))/115)*100, (total_potato/115)*100, total_report[0, 3]))
#    print('%g'%((len(list(background_detected_dict.keys()))/total_background)*100))
    
#    print('%s, %g, %d'%(dataname[test_index], (len(list(background_detected_dict.keys()))/total_background)*100, total_report[0, 3]))
    
#candidate_peptide_quantity=pointIso_human_peptide_quantity
#candidate_peptide_quantity=pointIso_potato_peptide_quantity
candidate_peptide_quantity=pointIso_background_peptide_quantity

candidate_peptide_quantity_sample=[]
for i in range (0, 12):
    candidate_peptide_quantity_sample.append(defaultdict(list)) 
    
for pep_seq_key in candidate_peptide_quantity.keys():
#    print(candidate_peptide_quantity[pep_seq_key])
    intensity=np.mean(candidate_peptide_quantity[pep_seq_key]) #sum,avg,max? # multiple hit to the same psm
    candidate_peptide_quantity[pep_seq_key]=intensity
    ms_file=int((pep_seq_key.split('-'))[0])
    mz_rt_z_seq=(pep_seq_key.split('-'))[1]
    if ms_file>=53:
        candidate_peptide_quantity_sample[11][mz_rt_z_seq].append(intensity)
    elif ms_file>=49:
        candidate_peptide_quantity_sample[10][mz_rt_z_seq].append(intensity)
    elif ms_file>=45:
        candidate_peptide_quantity_sample[9][mz_rt_z_seq].append(intensity)
    elif ms_file>=41:
        candidate_peptide_quantity_sample[8][mz_rt_z_seq].append(intensity)
    elif ms_file>=37:
        candidate_peptide_quantity_sample[7][mz_rt_z_seq].append(intensity)
    elif ms_file>=33:
        candidate_peptide_quantity_sample[6][mz_rt_z_seq].append(intensity)
    elif ms_file>=29:
        candidate_peptide_quantity_sample[5][mz_rt_z_seq].append(intensity)
    elif ms_file>=25:
        candidate_peptide_quantity_sample[4][mz_rt_z_seq].append(intensity)
    elif ms_file>=18:
        candidate_peptide_quantity_sample[3][mz_rt_z_seq].append(intensity)
    elif ms_file>=11:
        candidate_peptide_quantity_sample[2][mz_rt_z_seq].append(intensity)
    elif ms_file>=4:
        candidate_peptide_quantity_sample[1][mz_rt_z_seq].append(intensity)
    elif ms_file>=0:
        candidate_peptide_quantity_sample[0][mz_rt_z_seq].append(intensity)
    
    

temp_hold=candidate_peptide_quantity_sample
candidate_peptide_quantity_sample=[]
for i in range (0, 12):
    candidate_peptide_quantity_sample.append(defaultdict(list))   
for sample_id in range (0, 12):
    feature_list=temp_hold[sample_id].keys()
    for feature in feature_list:
        considered_intensity = np.max(temp_hold[sample_id][feature]) # alignment over runs
        seq=(feature.split('|'))[1]        
        if considered_intensity>0:
            candidate_peptide_quantity_sample[sample_id][seq].append(considered_intensity)

temp_hold=candidate_peptide_quantity_sample
candidate_peptide_quantity_sample=[]
for i in range (0, 12):
    candidate_peptide_quantity_sample.append(defaultdict(list))   
for sample_id in range (0, 12):
    feature_list=temp_hold[sample_id].keys()
    for feature in feature_list:
        considered_intensity = np.max(temp_hold[sample_id][feature]) # z,seq
        seq=feature[1:len(feature)] 
        z=int(feature[0])
        if considered_intensity>0:
            candidate_peptide_quantity_sample[sample_id][seq].append(considered_intensity)

feature_auc_per_sample=defaultdict(list) 
total_intensity=0
for sample_id in range (0, 12):
    feature_list=candidate_peptide_quantity_sample[sample_id].keys()
    for feature in feature_list:
        considered_intensity = np.sum(candidate_peptide_quantity_sample[sample_id][feature]) # seq
        if considered_intensity>0:
            feature_auc_per_sample[feature].append([sample_id, considered_intensity])
        total_intensity=total_intensity+considered_intensity
            
peptide_slopes=[]
feature_list=list(feature_auc_per_sample.keys())
for j in range(0, len(feature_list)):
    feature=feature_list[j]
    sample_list=feature_auc_per_sample[feature]
    if len(sample_list)<2:
        continue
    #calculate slope
    x_series=[]
    y_series=[]
    for i in range (0, len(sample_list)):
        x_series.append(sample_list[i][0])
        y_series.append(sample_list[i][1])

    for i in range (0, len(y_series)):
        y_series[i]=math.log(y_series[i]) #(y_series[i]-min_intensity)/(max_intensity-min_intensity)
    
#    print(j)
#    print(y_series)
    slope, intercept, r, p, se = scipy.stats.linregress(x_series, y_series) #scipy.stats.linregress(x_series[start_file:end_file], y_series[start_file:end_file])
    peptide_slopes.append([round(slope,2), feature])

slope_dist_pointIso=[]
peptide_slopes_pointIso=sorted(peptide_slopes)
for i in range (0, len(peptide_slopes)):
    slope_dist_pointIso.append(peptide_slopes_pointIso[i][0])
    
plt.hist(slope_dist_pointIso, bins=10, alpha=0.9, density=True, label='pointIso', histtype='step') #
plt.legend(loc='upper right')
plt.show(block=False)
######################################################################################################
peaks_background_peptide_quantity=defaultdict(list)
peaks_human_peptide_quantity=defaultdict(list)
peaks_potato_peptide_quantity=defaultdict(list)
for test_index in range (0, 57):
    for runtime in range (0, 1): #
        filename ='/data/fzohora/dilution_series_syn_pep/feature_list/mascot/peptides_in_the_dilution_series.csv'
        peptide_list = [] 
        csvfile=open(filename, 'r')
        csvreader = csv.reader(csvfile)     
        for row in csvreader:
            peptide_list.append(row)
        csvfile.close()
        

        threshold_score=25
        f=open(datapath+'feature_list/mascot/'+dataname[test_index]+'_mascot_db_search', 'rb')
        peptide_mascot=pickle.load(f)
        f.close()    
#        
        temp_peptide_mascot=[]
        temp_peptide_mascot.append(peptide_mascot[0])
        for i in range (1, len(peptide_mascot)):
            if float(peptide_mascot[i][4])<=threshold_score:
                continue
            if round(float(peptide_mascot[i][7]), 2) < min_RT:#  or round(float(peptide_mascot[i][2]), mz_resolution)<min_mz: #or round(float(peptide_mascot[i][2]), mz_resolution)>800 or 
                continue
            temp_peptide_mascot.append(peptide_mascot[i])
        peptide_mascot=temp_peptide_mascot
        temp_peptide_mascot=0

        total_report=np.zeros((1, 5)) # 0 = openMS, 1=maxQuant, 2=dino, 3=DeepIso, 4= peaks
        ftr_matched_auc=np.zeros((len(peptide_mascot), 5)) # 0 = openMS, 1=maxQuant, 2=dino, 3=DeepIso, 4= peaks
        detected_peptide=np.zeros((len(peptide_mascot), 7)) # 0 = our, 1 = peaks, 2 = maxquant, 3= charge by Peaks, 4=peaks id, 5=dino, 6=openMS


        peptide_mascot[0].append('type')
        count=np.zeros((6))
        potato_detected_dict=dict()
        human_detected_dict=dict()
        background_detected_dict=dict()
        for i in range (1, len(peptide_mascot)):
            if peptide_mascot[i][5] in spiked_peptide_dict:
                peptide_mascot[i].append(spiked_peptide_dict[peptide_mascot[i][5]])
                if spiked_peptide_dict[peptide_mascot[i][5]]=='potato':
                    count[0]=count[0]+1
                    potato_detected_dict[peptide_mascot[i][5]]='found'
                elif spiked_peptide_dict[peptide_mascot[i][5]]=='human' :
                    count[1]=count[1]+1
                    human_detected_dict[peptide_mascot[i][5]]='found'
            else:
                peptide_mascot[i].append('background')
                background_detected_dict[peptide_mascot[i][5]]='found'
                count[5]=count[5]+1  
                
        total_potato=len(potato_detected_dict.keys())
        total_human=len(human_detected_dict.keys()) 
        total_background=len(background_detected_dict.keys()) 
                        
        ##########################PEAKS###################################
        #logfile=open(datapath+'feature_list/'+dataname[test_index]+'_combineIsotopes_featureList.csv', 'rb')
        #peptide_feature=np.loadtxt(logfile, delimiter=',')
        #logfile.close()


        filename ='/data/fzohora/dilution_series_syn_pep/feature_list/PEAKs/PEAKSX/'+dataname[test_index]+'_peptide_feature.csv'
         
        # initializing the titles and peptide_mascot list
        peptide_feature = [] 
        # reading csv file
        csvfile=open(filename, 'r')
        # creating a csv reader object
        csvreader = csv.reader(csvfile)  
        
        # extracting each data row one by one
        for row in csvreader:
            peptide_feature.append(row)
        csvfile.close()

        feature_table_peaks=defaultdict(list)
    #    auc_list_peaks=[]
        auc_dict_peaks=defaultdict(list)
        for i in range (1, len(peptide_feature)):
            if round(float(peptide_feature[i][4]), 2)<min_RT: # or  round(peptide_feature[i, 0], mz_resolution) < min_mz: # or round(peptide_feature[i, 0], mz_resolution)>800:
                continue
            
            new_ftr=[]
            new_ftr.append(round(float(peptide_feature[i][2]), mz_resolution))
            new_ftr.append( round(float(peptide_feature[i][4]), 2)) #RT
            new_ftr.append( int(float(peptide_feature[i][3]))) #charge
            new_ftr.append(float(peptide_feature[i][7]))
            new_ftr.append(i)
            auc_dict_peaks[round(float(peptide_feature[i][7]))].append(i) # auc
            feature_table_peaks[round(float(peptide_feature[i][2]), mz_resolution)].append(new_ftr)
            total_report[0, 4]=total_report[0, 4]+1   
        auc_list_peaks=sorted(list(auc_dict_peaks.keys()), reverse=True)

        RT_tolerance=0.2 # dino is same
        found_ftr=0
        total_feature=0
        detected_peptide[:, 1]=0
        potato_detected_dict=dict()
        human_detected_dict=dict()
        background_detected_dict=dict()
        for i in range (1, len(peptide_mascot)):
            total_feature=total_feature+1    
            mz_exact=round(float(peptide_mascot[i][2]), mz_resolution)
            mz_range=[]
            mz_range.append(mz_exact)            
            tolerance_mz=0.01 #(mz_exact*10.0)/10**6 # dinosaur = 0.005 
            mz_range.append(round(mz_exact-tolerance_mz, mz_resolution))
            mz_range.append(round(mz_exact+tolerance_mz, mz_resolution))
            found=0 
            pep_seq_key=str(test_index)+'-'+str(mz_exact)+str(round(float(peptide_mascot[i][7]), 2))+'|'+str(int(peptide_mascot[i][3]))+peptide_mascot[i][5]
            for j in range (0, len(mz_range)):
                mz=mz_range[j]
                if mz in feature_table_peaks:
                    ftr_list=feature_table_peaks[mz]
                    for k in range (0, len(ftr_list)):
                        ftr=ftr_list[k]
#                        if ftr[3]<auc_list_peaks[high_conf_limit]:
#                            continue
                        peak_RT=ftr[1]
                        if (round(float(peptide_mascot[i][7])-RT_tolerance, 2) <= peak_RT) and (peak_RT<=round(float(peptide_mascot[i][7])+RT_tolerance, 2)) and int(ftr[2])==int(peptide_mascot[i][3]):
                            found=1
                            detected_peptide[i, 1]=1
                            detected_peptide[i, 3]=ftr[2]
                            detected_peptide[i, 4]=ftr[4]
                            found_ftr=found_ftr+1
                            ftr_matched_auc[i, 4]=ftr[3]
                            
                            if peptide_mascot[i][8]=='potato': #'potato-SRM':
                                potato_detected_dict[peptide_mascot[i][5]]='found'
                                peaks_potato_peptide_quantity[pep_seq_key].append(ftr[3])
                            elif peptide_mascot[i][8]=='human': #-SRM
                                human_detected_dict[peptide_mascot[i][5]]='found'
                                peaks_human_peptide_quantity[pep_seq_key].append(ftr[3])
                            else:
                                background_detected_dict[peptide_mascot[i][5]]='found'
                                peaks_background_peptide_quantity[pep_seq_key].append(ftr[3])
                                
#                            break

#                    if found==1:
#                        break
                        
#        print('%d'%len(list(human_detected_dict.keys())))
#        print('%d'%len(list(potato_detected_dict.keys())))
        #human peptides are grouped by sample 1 to 12
#        print(' %g, %g'%((len(list(human_detected_dict.keys()))/158)*100, (len(list(potato_detected_dict.keys()))/115)*100))
        print('%g'%((len(list(background_detected_dict.keys()))/total_background)*100))

        
#candidate_peptide_quantity=peaks_human_peptide_quantity
#candidate_peptide_quantity=peaks_potato_peptide_quantity
candidate_peptide_quantity=peaks_background_peptide_quantity


candidate_peptide_quantity_sample=[]
for i in range (0, 12):
    candidate_peptide_quantity_sample.append(defaultdict(list)) 
    
for pep_seq_key in candidate_peptide_quantity.keys():
#    print(candidate_peptide_quantity[pep_seq_key])
    intensity=np.mean(candidate_peptide_quantity[pep_seq_key]) #sum,avg,max? # multiple hit to the same psm
    candidate_peptide_quantity[pep_seq_key]=intensity
    ms_file=int((pep_seq_key.split('-'))[0])
    mz_rt_z_seq=(pep_seq_key.split('-'))[1]
    if ms_file>=53:
        candidate_peptide_quantity_sample[11][mz_rt_z_seq].append(intensity)
    elif ms_file>=49:
        candidate_peptide_quantity_sample[10][mz_rt_z_seq].append(intensity)
    elif ms_file>=45:
        candidate_peptide_quantity_sample[9][mz_rt_z_seq].append(intensity)
    elif ms_file>=41:
        candidate_peptide_quantity_sample[8][mz_rt_z_seq].append(intensity)
    elif ms_file>=37:
        candidate_peptide_quantity_sample[7][mz_rt_z_seq].append(intensity)
    elif ms_file>=33:
        candidate_peptide_quantity_sample[6][mz_rt_z_seq].append(intensity)
    elif ms_file>=29:
        candidate_peptide_quantity_sample[5][mz_rt_z_seq].append(intensity)
    elif ms_file>=25:
        candidate_peptide_quantity_sample[4][mz_rt_z_seq].append(intensity)
    elif ms_file>=18:
        candidate_peptide_quantity_sample[3][mz_rt_z_seq].append(intensity)
    elif ms_file>=11:
        candidate_peptide_quantity_sample[2][mz_rt_z_seq].append(intensity)
    elif ms_file>=4:
        candidate_peptide_quantity_sample[1][mz_rt_z_seq].append(intensity)
    elif ms_file>=0:
        candidate_peptide_quantity_sample[0][mz_rt_z_seq].append(intensity)
    
    

temp_hold=candidate_peptide_quantity_sample
candidate_peptide_quantity_sample=[]
for i in range (0, 12):
    candidate_peptide_quantity_sample.append(defaultdict(list))   
for sample_id in range (0, 12):
    feature_list=temp_hold[sample_id].keys()
    for feature in feature_list:
        considered_intensity = np.max(temp_hold[sample_id][feature]) # alignment over runs
        seq=(feature.split('|'))[1]        
        if considered_intensity>0:
            candidate_peptide_quantity_sample[sample_id][seq].append(considered_intensity)

temp_hold=candidate_peptide_quantity_sample
candidate_peptide_quantity_sample=[]
for i in range (0, 12):
    candidate_peptide_quantity_sample.append(defaultdict(list))   
for sample_id in range (0, 12):
    feature_list=temp_hold[sample_id].keys()
    for feature in feature_list:
        considered_intensity = np.max(temp_hold[sample_id][feature]) # z,seq
        seq=feature[1:len(feature)] 
        z=int(feature[0])
        if considered_intensity>0:
            candidate_peptide_quantity_sample[sample_id][seq].append(considered_intensity)

feature_auc_per_sample=defaultdict(list) 
total_intensity=0
for sample_id in range (0, 12):
    feature_list=candidate_peptide_quantity_sample[sample_id].keys()
    for feature in feature_list:
        considered_intensity = np.sum(candidate_peptide_quantity_sample[sample_id][feature]) # seq
        if considered_intensity>0:
            feature_auc_per_sample[feature].append([sample_id, considered_intensity])
        total_intensity=total_intensity+considered_intensity
            
peptide_slopes=[]
feature_list=list(feature_auc_per_sample.keys())
for j in range(0, len(feature_list)):
    feature=feature_list[j]
    sample_list=feature_auc_per_sample[feature]
    if len(sample_list)<2:
        continue
    #calculate slope
    x_series=[]
    y_series=[]
    for i in range (0, len(sample_list)):
        x_series.append(sample_list[i][0])
        y_series.append(sample_list[i][1])

    for i in range (0, len(y_series)):
        y_series[i]=math.log(y_series[i]) #(y_series[i]-min_intensity)/(max_intensity-min_intensity)
    
#    print(j)
#    print(y_series)
    slope, intercept, r, p, se = scipy.stats.linregress(x_series, y_series) #scipy.stats.linregress(x_series[start_file:end_file], y_series[start_file:end_file])
    peptide_slopes.append([round(slope,2), feature])

slope_dist_peaks=[]
peptide_slopes_peaks=sorted(peptide_slopes)
for i in range (0, len(peptide_slopes)):
    slope_dist_peaks.append(peptide_slopes_peaks[i][0])
    
plt.hist(slope_dist_peaks, bins=10, alpha=0.9, density=True, label='peaks', histtype='step') #
plt.legend(loc='upper right')
plt.show(block=False)

#############################################################################################
dino_human_peptide_quantity=defaultdict(list)
dino_potato_peptide_quantity=defaultdict(list)
dino_background_peptide_quantity=defaultdict(list)
for test_index in range (0, 57):
    for runtime in range (0, 1): #
        filename ='/data/fzohora/dilution_series_syn_pep/feature_list/mascot/peptides_in_the_dilution_series.csv'
        peptide_list = [] 
        csvfile=open(filename, 'r')
        csvreader = csv.reader(csvfile)     
        for row in csvreader:
            peptide_list.append(row)
        csvfile.close()
           
        threshold_score=25
        f=open(datapath+'feature_list/mascot/'+dataname[test_index]+'_mascot_db_search', 'rb')
        peptide_mascot=pickle.load(f)
        f.close()    
#        
        temp_peptide_mascot=[]
        temp_peptide_mascot.append(peptide_mascot[0])
        for i in range (1, len(peptide_mascot)):
            if float(peptide_mascot[i][4])<=threshold_score:
                continue
            if round(float(peptide_mascot[i][7]), 2) < min_RT:#  or round(float(peptide_mascot[i][2]), mz_resolution)<min_mz: #or round(float(peptide_mascot[i][2]), mz_resolution)>800 or 
                continue
            temp_peptide_mascot.append(peptide_mascot[i])
        peptide_mascot=temp_peptide_mascot
        temp_peptide_mascot=0

        total_report=np.zeros((1, 5)) # 0 = openMS, 1=maxQuant, 2=dino, 3=DeepIso, 4= peaks
        ftr_matched_auc=np.zeros((len(peptide_mascot), 5)) # 0 = openMS, 1=maxQuant, 2=dino, 3=DeepIso, 4= peaks
        detected_peptide=np.zeros((len(peptide_mascot), 7)) # 0 = our, 1 = peaks, 2 = maxquant, 3= charge by Peaks, 4=peaks id, 5=dino, 6=openMS

        peptide_mascot[0].append('type')
        count=np.zeros((6))
        potato_detected_dict=dict()
        human_detected_dict=dict()
        background_detected_dict=dict()
        for i in range (1, len(peptide_mascot)):
            if peptide_mascot[i][5] in spiked_peptide_dict:
                peptide_mascot[i].append(spiked_peptide_dict[peptide_mascot[i][5]])
                if spiked_peptide_dict[peptide_mascot[i][5]]=='potato':
                    count[0]=count[0]+1
                    potato_detected_dict[peptide_mascot[i][5]]='found'
                elif spiked_peptide_dict[peptide_mascot[i][5]]=='human' :
                    count[1]=count[1]+1
                    human_detected_dict[peptide_mascot[i][5]]='found'
            else:
                peptide_mascot[i].append('background')
                background_detected_dict[peptide_mascot[i][5]]='found'
                count[5]=count[5]+1  
                
        total_potato=len(potato_detected_dict.keys())
        total_human=len(human_detected_dict.keys()) 
        total_background=len(background_detected_dict.keys()) 
        ########### Dinosaurs ################################################
#        filename ='/data/fzohora/dilution_series_syn_pep/feature_list/dino/'+dataname[test_index]+'_dino.csv'
        dino_peptide_mascot = []
        try:
            filename ='/data/fzohora/dilution_series_syn_pep/feature_list/dino/'+dataname[test_index]+'.features.tsv'
            # reading csv file
            csvfile=open(filename, 'r')
            # creating a csv reader object
            csvreader = csv.reader(csvfile, delimiter='\t')     
            # extracting each data row one by one
            for row in csvreader:
                dino_peptide_mascot.append(row)
            csvfile.close() 

        except:
#        if test_index in (36, 43, 44):
            filename ='/data/fzohora/dilution_series_syn_pep/feature_list/dino/'+dataname[test_index]+'_dino.csv'
            # reading csv file
            csvfile=open(filename, 'r')
            # creating a csv reader object
            csvreader = csv.reader(csvfile, delimiter=',')     
            # extracting each data row one by one
            for row in csvreader:
                dino_peptide_mascot.append(row)
            csvfile.close() 

            
        feature_table_dino=defaultdict(list)
        auc_list_dino=[]
        auc_dict_dino=defaultdict(list)
        count=0
        for i in range (1, len(dino_peptide_mascot)):
            if float(dino_peptide_mascot[i][3])<min_RT: #or round(float(dino_peptide_mascot[i][0]), mz_resolution)<min_mz: # or round(float(dino_peptide_mascot[i][0]), mz_resolution)>800:
                continue
            new_ftr=[]
            new_ftr.append(round(float(dino_peptide_mascot[i][0]), mz_resolution))
            new_ftr.append(round(float(float(dino_peptide_mascot[i][4])), 2))
            try:
                new_ftr.append(float(dino_peptide_mascot[i][13]))
                auc_dict_dino[round(float(dino_peptide_mascot[i][13]))].append(i) 
            except:
                count=count+1
                new_ftr.append(0) 
                auc_dict_dino[0].append(i) 
                

            new_ftr.append(dino_peptide_mascot[i][2]) #charge
            feature_table_dino[round(float(dino_peptide_mascot[i][0]), mz_resolution)].append(new_ftr)
            total_report[0, 2]=total_report[0, 2]+1            

        auc_list_dino=sorted(list(auc_dict_dino.keys()), reverse=True)
    #    print('probb %d'%count)
        potato_detected_dict=dict()
        human_detected_dict=dict()
        background_detected_dict=dict()
        RT_tolerance=0.2 # dino is same
        found_ftr=0
        total_feature=0
        detected_peptide[:, 5]=0
        for i in range (1, len(peptide_mascot)):            
            total_feature=total_feature+1  
            mz_exact=round(float(peptide_mascot[i][2]), mz_resolution)
            mz_range=[]
            mz_range.append(mz_exact)
                
            tolerance_mz=0.01 #(mz_exact*10.0)/10**6 # dinosaur = 0.005 
            mz_range.append(round(mz_exact-tolerance_mz, mz_resolution))
            mz_range.append(round(mz_exact+tolerance_mz, mz_resolution))
            found=0 
            pep_seq_key=str(test_index)+'-'+str(mz_exact)+str(round(float(peptide_mascot[i][7]), 2))+'|'+str(int(peptide_mascot[i][3]))+peptide_mascot[i][5]
            for j in range (0, len(mz_range)):
                mz=mz_range[j]
                if mz in feature_table_dino:
                    ftr_list=feature_table_dino[mz]
                    for k in range (0, len(ftr_list)):
                        ftr=ftr_list[k]
#                        if ftr[2]<auc_list_dino[high_conf_limit]:
#                            continue
                        
                        peak_RT=ftr[1]
                        if (round(float(peptide_mascot[i][7])-RT_tolerance, 2) <= peak_RT) and (peak_RT<=round(float(peptide_mascot[i][7])+RT_tolerance, 2))and int(ftr[3])==int(peptide_mascot[i][3]):
                            found=1
                            detected_peptide[i, 5]=1
                            found_ftr=found_ftr+1
                            ftr_matched_auc[i, 2]=ftr[2]
                            
                            if peptide_mascot[i][8]=='potato': #-SRM
                                potato_detected_dict[peptide_mascot[i][5]]='found'
                                dino_potato_peptide_quantity[pep_seq_key].append(ftr[2])
                            elif peptide_mascot[i][8]=='human': #-SRM
                                human_detected_dict[peptide_mascot[i][5]]='found'
                                dino_human_peptide_quantity[pep_seq_key].append(ftr[2])
                            else:
                                background_detected_dict[peptide_mascot[i][5]]='found'
                                dino_background_peptide_quantity[pep_seq_key].append(ftr[2])
                                
#                            break
#
#                    if found==1:
#                        break
        print('%g'%((len(list(background_detected_dict.keys()))/total_background)*100))

#        print(' %g, %g'%((len(list(human_detected_dict.keys()))/158)*100, (len(list(potato_detected_dict.keys()))/115)*100))


#candidate_peptide_quantity=dino_human_peptide_quantity
#candidate_peptide_quantity=dino_potato_peptide_quantity
candidate_peptide_quantity=dino_background_peptide_quantity

candidate_peptide_quantity_sample=[]
for i in range (0, 12):
    candidate_peptide_quantity_sample.append(defaultdict(list)) 
    
for pep_seq_key in candidate_peptide_quantity.keys():
#    print(candidate_peptide_quantity[pep_seq_key])
    intensity=np.mean(candidate_peptide_quantity[pep_seq_key]) #sum,avg,max? # multiple hit to the same psm
    candidate_peptide_quantity[pep_seq_key]=intensity
    ms_file=int((pep_seq_key.split('-'))[0])
    mz_rt_z_seq=(pep_seq_key.split('-'))[1]
    if ms_file>=53:
        candidate_peptide_quantity_sample[11][mz_rt_z_seq].append(intensity)
    elif ms_file>=49:
        candidate_peptide_quantity_sample[10][mz_rt_z_seq].append(intensity)
    elif ms_file>=45:
        candidate_peptide_quantity_sample[9][mz_rt_z_seq].append(intensity)
    elif ms_file>=41:
        candidate_peptide_quantity_sample[8][mz_rt_z_seq].append(intensity)
    elif ms_file>=37:
        candidate_peptide_quantity_sample[7][mz_rt_z_seq].append(intensity)
    elif ms_file>=33:
        candidate_peptide_quantity_sample[6][mz_rt_z_seq].append(intensity)
    elif ms_file>=29:
        candidate_peptide_quantity_sample[5][mz_rt_z_seq].append(intensity)
    elif ms_file>=25:
        candidate_peptide_quantity_sample[4][mz_rt_z_seq].append(intensity)
    elif ms_file>=18:
        candidate_peptide_quantity_sample[3][mz_rt_z_seq].append(intensity)
    elif ms_file>=11:
        candidate_peptide_quantity_sample[2][mz_rt_z_seq].append(intensity)
    elif ms_file>=4:
        candidate_peptide_quantity_sample[1][mz_rt_z_seq].append(intensity)
    elif ms_file>=0:
        candidate_peptide_quantity_sample[0][mz_rt_z_seq].append(intensity)
    
    

temp_hold=candidate_peptide_quantity_sample
candidate_peptide_quantity_sample=[]
for i in range (0, 12):
    candidate_peptide_quantity_sample.append(defaultdict(list))   
for sample_id in range (0, 12):
    feature_list=temp_hold[sample_id].keys()
    for feature in feature_list:
        considered_intensity = np.max(temp_hold[sample_id][feature]) # alignment over runs
        seq=(feature.split('|'))[1]        
        if considered_intensity>0:
            candidate_peptide_quantity_sample[sample_id][seq].append(considered_intensity)

temp_hold=candidate_peptide_quantity_sample
candidate_peptide_quantity_sample=[]
for i in range (0, 12):
    candidate_peptide_quantity_sample.append(defaultdict(list))   
for sample_id in range (0, 12):
    feature_list=temp_hold[sample_id].keys()
    for feature in feature_list:
        considered_intensity = np.max(temp_hold[sample_id][feature]) # z,seq
        seq=feature[1:len(feature)] 
        z=int(feature[0])
        if considered_intensity>0:
            candidate_peptide_quantity_sample[sample_id][seq].append(considered_intensity)

feature_auc_per_sample=defaultdict(list) 
total_intensity=0
for sample_id in range (0, 12):
    feature_list=candidate_peptide_quantity_sample[sample_id].keys()
    for feature in feature_list:
        considered_intensity = np.sum(candidate_peptide_quantity_sample[sample_id][feature]) # seq
        if considered_intensity>0:
            feature_auc_per_sample[feature].append([sample_id, considered_intensity])
        total_intensity=total_intensity+considered_intensity
            

peptide_slopes=[]
feature_list=list(feature_auc_per_sample.keys())
for j in range(0, len(feature_list)):
    feature=feature_list[j]
    sample_list=feature_auc_per_sample[feature]
    if len(sample_list)<2:
        continue
    #calculate slope
    x_series=[]
    y_series=[]
    for i in range (0, len(sample_list)):
        x_series.append(sample_list[i][0])
        y_series.append(sample_list[i][1])

    for i in range (0, len(y_series)):
        y_series[i]=math.log(y_series[i]) #(y_series[i]-min_intensity)/(max_intensity-min_intensity)
    
#    print(j)
#    print(y_series)
    slope, intercept, r, p, se = scipy.stats.linregress(x_series, y_series) #scipy.stats.linregress(x_series[start_file:end_file], y_series[start_file:end_file])
    peptide_slopes.append([round(slope,2), feature])

slope_dist_dino=[]
peptide_slopes_dino=sorted(peptide_slopes)
for i in range (0, len(peptide_slopes)):
    slope_dist_dino.append(peptide_slopes_dino[i][0])
    
plt.hist(slope_dist_dino, bins=10, alpha=0.9, density=True, label='dino', histtype='step') #
plt.legend(loc='upper right')
plt.show(block=False)
##############################################################################################################

mq_background_peptide_quantity=defaultdict(list)
mq_human_peptide_quantity=defaultdict(list)
mq_potato_peptide_quantity=defaultdict(list)
for test_index in range (0, 57):
    for runtime in range (0, 1): #
        filename ='/data/fzohora/dilution_series_syn_pep/feature_list/mascot/peptides_in_the_dilution_series.csv'
        peptide_list = [] 
        csvfile=open(filename, 'r')
        csvreader = csv.reader(csvfile)     
        for row in csvreader:
            peptide_list.append(row)
        csvfile.close()
        
           
        threshold_score=25
        f=open(datapath+'feature_list/mascot/'+dataname[test_index]+'_mascot_db_search', 'rb')
        peptide_mascot=pickle.load(f)
        f.close()    
#        
        temp_peptide_mascot=[]
        temp_peptide_mascot.append(peptide_mascot[0])
        for i in range (1, len(peptide_mascot)):
            if float(peptide_mascot[i][4])<=threshold_score:
                continue
            if round(float(peptide_mascot[i][7]), 2) < min_RT:#  or round(float(peptide_mascot[i][2]), mz_resolution)<min_mz: #or round(float(peptide_mascot[i][2]), mz_resolution)>800 or 
                continue
            temp_peptide_mascot.append(peptide_mascot[i])
        peptide_mascot=temp_peptide_mascot
        temp_peptide_mascot=0

        total_report=np.zeros((1, 5)) # 0 = openMS, 1=maxQuant, 2=dino, 3=DeepIso, 4= peaks
        ftr_matched_auc=np.zeros((len(peptide_mascot), 5)) # 0 = openMS, 1=maxQuant, 2=dino, 3=DeepIso, 4= peaks
        detected_peptide=np.zeros((len(peptide_mascot), 7)) # 0 = our, 1 = peaks, 2 = maxquant, 3= charge by Peaks, 4=peaks id, 5=dino, 6=openMS

        peptide_mascot[0].append('type')
        count=np.zeros((6))
        potato_detected_dict=dict()
        human_detected_dict=dict()
        background_detected_dict=dict()
        for i in range (1, len(peptide_mascot)):
            if peptide_mascot[i][5] in spiked_peptide_dict:
                peptide_mascot[i].append(spiked_peptide_dict[peptide_mascot[i][5]])
                if spiked_peptide_dict[peptide_mascot[i][5]]=='potato':
                    count[0]=count[0]+1
                    potato_detected_dict[peptide_mascot[i][5]]='found'
                elif spiked_peptide_dict[peptide_mascot[i][5]]=='human' :
                    count[1]=count[1]+1
                    human_detected_dict[peptide_mascot[i][5]]='found'
            else:
                peptide_mascot[i].append('background')
                background_detected_dict[peptide_mascot[i][5]]='found'
                count[5]=count[5]+1  
                
        total_potato=len(potato_detected_dict.keys())
        total_human=len(human_detected_dict.keys()) 
        total_background=len(background_detected_dict.keys()) 
        
    ######################################################################
        filename ='/data/fzohora/dilution_series_syn_pep/feature_list/maxQ/'+dataname[test_index]+'_3.csv'
         
        # initializing the titles and peptide_mascot list
        MQ_peptide_mascot = [] 
        # reading csv file
        csvfile=open(filename, 'r')
        # creating a csv reader object
        csvreader = csv.reader(csvfile)     
        # extracting each data row one by one
        for row in csvreader:
            MQ_peptide_mascot.append(row)
        csvfile.close()
        
        auc_dict_MQ=defaultdict(list)
        feature_table_mq=defaultdict(list)
        auc_list_MQ=[]
        count=0
        for i in range (1, len(MQ_peptide_mascot)):
            if float(MQ_peptide_mascot[i][4])<min_RT: #  or round(float(MQ_peptide_mascot[i][1]), mz_resolution)<min_mz: #or round(float(MQ_peptide_mascot[i][1]), mz_resolution)>800 :
                continue
            
            new_ftr=[]
            new_ftr.append(round(float(MQ_peptide_mascot[i][1]), mz_resolution))
            new_ftr.append(round(float(float(MQ_peptide_mascot[i][4])), 2))
            try:
                new_ftr.append(float(MQ_peptide_mascot[i][5]))
                auc_dict_MQ[round(float(MQ_peptide_mascot[i][5]))].append(i) 
            except:
                count=count+1
                new_ftr.append(0)        
                auc_dict_MQ[0].append(i) 

            new_ftr.append(MQ_peptide_mascot[i][0]) #charge
            feature_table_mq[round(float(MQ_peptide_mascot[i][1]), mz_resolution)].append(new_ftr)
            total_report[0, 1]=total_report[0, 1]+1   

        auc_list_MQ=sorted(list(auc_dict_MQ.keys()), reverse=True)
    #    print('probb %d'%count)
    
        potato_detected_dict=dict()
        human_detected_dict=dict()
        background_detected_dict=dict()
        RT_tolerance=0.2 # dino is same
        found_ftr=0
        total_feature=0
        detected_peptide[:, 2]=0
        for i in range (1, len(peptide_mascot)):
        #    if float(peptide_mascot[i][7])<100000:
        #        continue
                
            total_feature=total_feature+1  
            mz_exact=round(float(peptide_mascot[i][2]), mz_resolution)
            mz_range=[]
            mz_range.append(mz_exact)
                
            tolerance_mz=0.01 #(mz_exact*10.0)/10**6 # dinosaur = 0.005 
            mz_range.append(round(mz_exact-tolerance_mz, mz_resolution))
            mz_range.append(round(mz_exact+tolerance_mz, mz_resolution))
            found=0 
            pep_seq_key=str(test_index)+'-'+str(mz_exact)+str(round(float(peptide_mascot[i][7]), 2))+'|'+str(int(peptide_mascot[i][3]))+peptide_mascot[i][5]
            for j in range (0, len(mz_range)):
                mz=mz_range[j]
                if mz in feature_table_mq:
                    ftr_list=feature_table_mq[mz]
                    for k in range (0, len(ftr_list)):
                        ftr=ftr_list[k]
#                        if ftr[2]<auc_list_MQ[high_conf_limit]:
#                            continue
                        peak_RT=ftr[1]
                        if (round(float(peptide_mascot[i][7])-RT_tolerance, 2) <= peak_RT) and (peak_RT<=round(float(peptide_mascot[i][7])+RT_tolerance, 2)) and int(ftr[3])==int(peptide_mascot[i][3]):

                            found=1
                            detected_peptide[i, 2]=1
                            found_ftr=found_ftr+1
                            ftr_matched_auc[i, 1]=ftr[2]
                            
                            if peptide_mascot[i][8]=='potato': #-SRM
                                potato_detected_dict[peptide_mascot[i][5]]='found'
                                mq_potato_peptide_quantity[pep_seq_key].append(ftr[2])
                            elif peptide_mascot[i][8]=='human': #-SRM
                                human_detected_dict[peptide_mascot[i][5]]='found'
                                mq_human_peptide_quantity[pep_seq_key].append(ftr[2])
                            else:
                                background_detected_dict[peptide_mascot[i][5]]='found'
                                mq_background_peptide_quantity[pep_seq_key].append(ftr[2])
                                
#                            break
#
#                    if found==1:
#                        break
        print('%g'%((len(list(background_detected_dict.keys()))/total_background)*100))
#        print('%d'%len(list(human_detected_dict.keys())))
#        print(' %g, %g'%((len(list(human_detected_dict.keys()))/158)*100, (len(list(potato_detected_dict.keys()))/115)*100))

#candidate_peptide_quantity=mq_human_peptide_quantity
#candidate_peptide_quantity=mq_potato_peptide_quantity
candidate_peptide_quantity=mq_background_peptide_quantity

candidate_peptide_quantity_sample=[]
for i in range (0, 12):
    candidate_peptide_quantity_sample.append(defaultdict(list)) 
    
for pep_seq_key in candidate_peptide_quantity.keys():
#    print(candidate_peptide_quantity[pep_seq_key])
    intensity=np.mean(candidate_peptide_quantity[pep_seq_key]) #sum,avg,max? # multiple hit to the same psm
    candidate_peptide_quantity[pep_seq_key]=intensity
    ms_file=int((pep_seq_key.split('-'))[0])
    mz_rt_z_seq=(pep_seq_key.split('-'))[1]
    if ms_file>=53:
        candidate_peptide_quantity_sample[11][mz_rt_z_seq].append(intensity)
    elif ms_file>=49:
        candidate_peptide_quantity_sample[10][mz_rt_z_seq].append(intensity)
    elif ms_file>=45:
        candidate_peptide_quantity_sample[9][mz_rt_z_seq].append(intensity)
    elif ms_file>=41:
        candidate_peptide_quantity_sample[8][mz_rt_z_seq].append(intensity)
    elif ms_file>=37:
        candidate_peptide_quantity_sample[7][mz_rt_z_seq].append(intensity)
    elif ms_file>=33:
        candidate_peptide_quantity_sample[6][mz_rt_z_seq].append(intensity)
    elif ms_file>=29:
        candidate_peptide_quantity_sample[5][mz_rt_z_seq].append(intensity)
    elif ms_file>=25:
        candidate_peptide_quantity_sample[4][mz_rt_z_seq].append(intensity)
    elif ms_file>=18:
        candidate_peptide_quantity_sample[3][mz_rt_z_seq].append(intensity)
    elif ms_file>=11:
        candidate_peptide_quantity_sample[2][mz_rt_z_seq].append(intensity)
    elif ms_file>=4:
        candidate_peptide_quantity_sample[1][mz_rt_z_seq].append(intensity)
    elif ms_file>=0:
        candidate_peptide_quantity_sample[0][mz_rt_z_seq].append(intensity)
    
    

temp_hold=candidate_peptide_quantity_sample
candidate_peptide_quantity_sample=[]
for i in range (0, 12):
    candidate_peptide_quantity_sample.append(defaultdict(list))   
for sample_id in range (0, 12):
    feature_list=temp_hold[sample_id].keys()
    for feature in feature_list:
        considered_intensity = np.max(temp_hold[sample_id][feature]) # alignment over runs
        seq=(feature.split('|'))[1]        
        if considered_intensity>0:
            candidate_peptide_quantity_sample[sample_id][seq].append(considered_intensity)

temp_hold=candidate_peptide_quantity_sample
candidate_peptide_quantity_sample=[]
for i in range (0, 12):
    candidate_peptide_quantity_sample.append(defaultdict(list))   
for sample_id in range (0, 12):
    feature_list=temp_hold[sample_id].keys()
    for feature in feature_list:
        considered_intensity = np.max(temp_hold[sample_id][feature]) # z,seq
        seq=feature[1:len(feature)] 
        z=int(feature[0])
        if considered_intensity>0:
            candidate_peptide_quantity_sample[sample_id][seq].append(considered_intensity)

feature_auc_per_sample=defaultdict(list) 
total_intensity=0
for sample_id in range (0, 12):
    feature_list=candidate_peptide_quantity_sample[sample_id].keys()
    for feature in feature_list:
        considered_intensity = np.sum(candidate_peptide_quantity_sample[sample_id][feature]) # seq
        if considered_intensity>0:
            feature_auc_per_sample[feature].append([sample_id, considered_intensity])
        total_intensity=total_intensity+considered_intensity
            
#feature_auc_per_sample=defaultdict(list)    
#for sample_id in range (0, 12):
#    feature_list=temp_hold[sample_id].keys()
#    for feature in feature_list:
#        considered_intensity = np.mean(temp_hold[sample_id][feature]) # alignment over runs
#        if considered_intensity>0:
#            feature_auc_per_sample[feature].append([sample_id, considered_intensity])
            



peptide_slopes=[]
feature_list=list(feature_auc_per_sample.keys())
for j in range(0, len(feature_list)):
    feature=feature_list[j]
    sample_list=feature_auc_per_sample[feature]
    if len(sample_list)<2:
        continue
    #calculate slope
    x_series=[]
    y_series=[]
    for i in range (0, len(sample_list)):
        x_series.append(sample_list[i][0])
        y_series.append(sample_list[i][1])

    for i in range (0, len(y_series)):
        y_series[i]=math.log(y_series[i]) #(y_series[i]-min_intensity)/(max_intensity-min_intensity)
    
#    print(j)
#    print(y_series)
    slope, intercept, r, p, se = scipy.stats.linregress(x_series, y_series) #scipy.stats.linregress(x_series[start_file:end_file], y_series[start_file:end_file])
    peptide_slopes.append([round(slope,2), feature])

slope_dist_mq=[]
peptide_slopes_mq=sorted(peptide_slopes)
for i in range (0, len(peptide_slopes)):
    slope_dist_mq.append(peptide_slopes_mq[i][0])
    
plt.hist(slope_dist_mq, bins=10, alpha=0.9, density=True, label='mq', histtype='step') #
plt.legend(loc='upper right')
plt.show(block=False)
##########################################################################
openMS_background_peptide_quantity=defaultdict(list)    
openMS_human_peptide_quantity=defaultdict(list)
openMS_potato_peptide_quantity=defaultdict(list)
for test_index in range (0, 57):
    for runtime in range (0, 1): #
        filename ='/data/fzohora/dilution_series_syn_pep/feature_list/mascot/peptides_in_the_dilution_series.csv'
        peptide_list = [] 
        csvfile=open(filename, 'r')
        csvreader = csv.reader(csvfile)     
        for row in csvreader:
            peptide_list.append(row)
        csvfile.close()
        

        threshold_score=25
        f=open(datapath+'feature_list/mascot/'+dataname[test_index]+'_mascot_db_search', 'rb')
        peptide_mascot=pickle.load(f)
        f.close()    
#        
        temp_peptide_mascot=[]
        temp_peptide_mascot.append(peptide_mascot[0])
        for i in range (1, len(peptide_mascot)):
            if float(peptide_mascot[i][4])<=threshold_score:
                continue
            if round(float(peptide_mascot[i][7]), 2) < min_RT:#  or round(float(peptide_mascot[i][2]), mz_resolution)<min_mz: #or round(float(peptide_mascot[i][2]), mz_resolution)>800 or 
                continue
            temp_peptide_mascot.append(peptide_mascot[i])
        peptide_mascot=temp_peptide_mascot
        temp_peptide_mascot=0

        total_report=np.zeros((1, 5)) # 0 = openMS, 1=maxQuant, 2=dino, 3=DeepIso, 4= peaks
        ftr_matched_auc=np.zeros((len(peptide_mascot), 5)) # 0 = openMS, 1=maxQuant, 2=dino, 3=DeepIso, 4= peaks
        detected_peptide=np.zeros((len(peptide_mascot), 7)) # 0 = our, 1 = peaks, 2 = maxquant, 3= charge by Peaks, 4=peaks id, 5=dino, 6=openMS

        peptide_mascot[0].append('type')
        count=np.zeros((6))
        potato_detected_dict=dict()
        human_detected_dict=dict()
        background_detected_dict=dict()
        for i in range (1, len(peptide_mascot)):
            if peptide_mascot[i][5] in spiked_peptide_dict:
                peptide_mascot[i].append(spiked_peptide_dict[peptide_mascot[i][5]])
                if spiked_peptide_dict[peptide_mascot[i][5]]=='potato':
                    count[0]=count[0]+1
                    potato_detected_dict[peptide_mascot[i][5]]='found'
                elif spiked_peptide_dict[peptide_mascot[i][5]]=='human' :
                    count[1]=count[1]+1
                    human_detected_dict[peptide_mascot[i][5]]='found'
            else:
                peptide_mascot[i].append('background')
                background_detected_dict[peptide_mascot[i][5]]='found'
                count[5]=count[5]+1  
                
        total_potato=len(potato_detected_dict.keys())
        total_human=len(human_detected_dict.keys()) 
        total_background=len(background_detected_dict.keys()) 

    ########################################################################
        f=open(datapath+'feature_list/openMS/'+dataname[test_index]+'_openMS_features', 'rb') # open('/media/fzohora/USB20FD/raw/mzml/'+dataname[test_index]+'_openMS_features', 'rb')
        ft_openMS=pickle.load(f, encoding='latin1')
        f.close()   
        auc_dict_op=defaultdict(list)
        feature_table_openMS=defaultdict(list)
        auc_list_openMS=[]
        count=0
        for i in range (0, ft_openMS.shape[0]):
            if float(ft_openMS[i][1])<min_RT: #  or round(ft_openMS[i][0], mz_resolution)<min_mz: # or round(ft_openMS[i][0], mz_resolution)>800:
                continue
            
            new_ftr=[]
            new_ftr.append(round(ft_openMS[i][0], mz_resolution))
            new_ftr.append(round(ft_openMS[i][1], 2))
            try:
                new_ftr.append(float(ft_openMS[i][3]))
                auc_dict_op[round(float(ft_openMS[i][3]))].append(i) #auc_list_openMS.append(float(ft_openMS[i][3])) 

            except:
                count=count+1
                new_ftr.append(0)        
                auc_dict_op[0].append(i) 

            new_ftr.append(ft_openMS[i][2]) #charge
            feature_table_openMS[round(float(ft_openMS[i][0]), mz_resolution)].append(new_ftr)
            total_report[0, 0]=total_report[0, 0]+1   

        auc_list_openMS=sorted(list(auc_dict_op.keys()), reverse=True)
    #    print('probb %d'%count)

        RT_tolerance=0.2 # dino is same
        found_ftr=0
        total_feature=0
        detected_peptide[:, 6]=0
        potato_detected_dict=dict()
        human_detected_dict=dict()
        background_detected_dict=dict()

        for i in range (1, len(peptide_mascot)):
                
            total_feature=total_feature+1  
            mz_exact=round(float(peptide_mascot[i][2]), mz_resolution)
            mz_range=[]
            mz_range.append(mz_exact)
                
            tolerance_mz=0.01 #(mz_exact*10.0)/10**6 # dinosaur = 0.005 
            mz_range.append(round(mz_exact-tolerance_mz, mz_resolution))
            mz_range.append(round(mz_exact+tolerance_mz, mz_resolution))
            found=0 
            pep_seq_key=str(test_index)+'-'+str(mz_exact)+str(round(float(peptide_mascot[i][7]), 2))+'|'+str(int(peptide_mascot[i][3]))+peptide_mascot[i][5]
            for j in range (0, len(mz_range)):
                mz=mz_range[j]
                if mz in feature_table_openMS:
                    ftr_list=feature_table_openMS[mz]
                    for k in range (0, len(ftr_list)):
                        ftr=ftr_list[k]
#                        if ftr[2]<auc_list_openMS[high_conf_limit]:
#                            continue
                        peak_RT=ftr[1]
                        if (round(float(peptide_mascot[i][7])-RT_tolerance, 2) <= peak_RT) and (peak_RT<=round(float(peptide_mascot[i][7])+RT_tolerance, 2)) and int(ftr[3])==int(peptide_mascot[i][3]):
                        
                            found=1
                            detected_peptide[i, 6]=1
                            found_ftr=found_ftr+1
                            ftr_matched_auc[i, 0]=ftr[2]
                            
                            if peptide_mascot[i][8]=='potato': #-SRM
                                potato_detected_dict[peptide_mascot[i][5]]='found'
                                openMS_potato_peptide_quantity[pep_seq_key].append(ftr[2])
                            elif peptide_mascot[i][8]=='human': #-SRM
                                human_detected_dict[peptide_mascot[i][5]]='found'
                                openMS_human_peptide_quantity[pep_seq_key].append(ftr[2])
                            else:
                                background_detected_dict[peptide_mascot[i][5]]='found'
                                openMS_background_peptide_quantity[pep_seq_key].append(ftr[2])
                                
#                            break
#
#                    if found==1:
#                        break
        print('%g'%((len(list(background_detected_dict.keys()))/total_background)*100))
#        print('%d'%len(list(human_detected_dict.keys())))
#        print(' %g, %g'%((len(list(human_detected_dict.keys()))/158)*100, (len(list(potato_detected_dict.keys()))/115)*100))
        
#candidate_peptide_quantity=openMS_human_peptide_quantity
#candidate_peptide_quantity=openMS_potato_peptide_quantity
candidate_peptide_quantity=openMS_background_peptide_quantity


candidate_peptide_quantity_sample=[]
for i in range (0, 12):
    candidate_peptide_quantity_sample.append(defaultdict(list)) 
    
for pep_seq_key in candidate_peptide_quantity.keys():
#    print(candidate_peptide_quantity[pep_seq_key])
    intensity=np.mean(candidate_peptide_quantity[pep_seq_key]) #sum,avg,max? # multiple hit to the same psm
    candidate_peptide_quantity[pep_seq_key]=intensity
    ms_file=int((pep_seq_key.split('-'))[0])
    mz_rt_z_seq=(pep_seq_key.split('-'))[1]
    if ms_file>=53:
        candidate_peptide_quantity_sample[11][mz_rt_z_seq].append(intensity)
    elif ms_file>=49:
        candidate_peptide_quantity_sample[10][mz_rt_z_seq].append(intensity)
    elif ms_file>=45:
        candidate_peptide_quantity_sample[9][mz_rt_z_seq].append(intensity)
    elif ms_file>=41:
        candidate_peptide_quantity_sample[8][mz_rt_z_seq].append(intensity)
    elif ms_file>=37:
        candidate_peptide_quantity_sample[7][mz_rt_z_seq].append(intensity)
    elif ms_file>=33:
        candidate_peptide_quantity_sample[6][mz_rt_z_seq].append(intensity)
    elif ms_file>=29:
        candidate_peptide_quantity_sample[5][mz_rt_z_seq].append(intensity)
    elif ms_file>=25:
        candidate_peptide_quantity_sample[4][mz_rt_z_seq].append(intensity)
    elif ms_file>=18:
        candidate_peptide_quantity_sample[3][mz_rt_z_seq].append(intensity)
    elif ms_file>=11:
        candidate_peptide_quantity_sample[2][mz_rt_z_seq].append(intensity)
    elif ms_file>=4:
        candidate_peptide_quantity_sample[1][mz_rt_z_seq].append(intensity)
    elif ms_file>=0:
        candidate_peptide_quantity_sample[0][mz_rt_z_seq].append(intensity)
    
    

temp_hold=candidate_peptide_quantity_sample
candidate_peptide_quantity_sample=[]
for i in range (0, 12):
    candidate_peptide_quantity_sample.append(defaultdict(list))   
for sample_id in range (0, 12):
    feature_list=temp_hold[sample_id].keys()
    for feature in feature_list:
        considered_intensity = np.max(temp_hold[sample_id][feature]) # alignment over runs
        seq=(feature.split('|'))[1]        
        if considered_intensity>0:
            candidate_peptide_quantity_sample[sample_id][seq].append(considered_intensity)

temp_hold=candidate_peptide_quantity_sample
candidate_peptide_quantity_sample=[]
for i in range (0, 12):
    candidate_peptide_quantity_sample.append(defaultdict(list))   
for sample_id in range (0, 12):
    feature_list=temp_hold[sample_id].keys()
    for feature in feature_list:
        considered_intensity = np.max(temp_hold[sample_id][feature]) # z,seq
        seq=feature[1:len(feature)] 
        z=int(feature[0])
        if considered_intensity>0:
            candidate_peptide_quantity_sample[sample_id][seq].append(considered_intensity)

feature_auc_per_sample=defaultdict(list) 
total_intensity=0
for sample_id in range (0, 12):
    feature_list=candidate_peptide_quantity_sample[sample_id].keys()
    for feature in feature_list:
        considered_intensity = np.sum(candidate_peptide_quantity_sample[sample_id][feature]) # seq
        if considered_intensity>0:
            feature_auc_per_sample[feature].append([sample_id, considered_intensity])
        total_intensity=total_intensity+considered_intensity
            

peptide_slopes=[]
feature_list=list(feature_auc_per_sample.keys())
for j in range(0, len(feature_list)):
    feature=feature_list[j]
    sample_list=feature_auc_per_sample[feature]
    if len(sample_list)<2:
        continue
    #calculate slope
    x_series=[]
    y_series=[]
    for i in range (0, len(sample_list)):
        x_series.append(sample_list[i][0])
        y_series.append(sample_list[i][1])

    for i in range (0, len(y_series)):
        y_series[i]=math.log(y_series[i]) #(y_series[i]-min_intensity)/(max_intensity-min_intensity)
    
#    print(j)
#    print(y_series)
    slope, intercept, r, p, se = scipy.stats.linregress(x_series, y_series) #scipy.stats.linregress(x_series[start_file:end_file], y_series[start_file:end_file])
    peptide_slopes.append([round(slope,2), feature])

slope_dist_openMS=[]
peptide_slopes_openMS=sorted(peptide_slopes)
for i in range (0, len(peptide_slopes)):
    slope_dist_openMS.append(peptide_slopes_openMS[i][0])
    
plt.hist(slope_dist_openMS, bins=10, alpha=0.9, density=True, label='openMS', histtype='step') #
plt.legend(loc='upper right')
plt.show(block=False)
