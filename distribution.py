from __future__ import division
from __future__ import print_function
import math
import csv
#from time import time
import pickle
import numpy as np
from sklearn import metrics
#from collections import deque
from collections import defaultdict
#import sys
#import copy
#import scipy.misc
#import scipy.stats
#import bisect
import matplotlib.pyplot as plt

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
min_RT=10.0
#for test_index in range (0, len(dataname)): # 
for test_index in range (44,  45): #len(dataname)):
#    print(dataname[test_index])
    high_conf_limit=10000
    ####################################################################
    ###########################################################################            
    for runtime in range (0, 1): #
        threshold_score=0
        f=open(datapath+'feature_list/mascot/'+dataname[test_index]+'_mascot_db_search', 'rb')
        peptide_mascot=pickle.load(f)
        f.close()    

        temp_peptide_mascot=[]
        temp_peptide_mascot.append(peptide_mascot[0])
        for i in range (1, len(peptide_mascot)):
            if float(peptide_mascot[i][4])<=threshold_score:
                continue
            if round(float(peptide_mascot[i][4]), 2) < min_RT: # or round(float(peptide_mascot[i][2]), mz_resolution)>=(2000-0.50) or round(float(peptide_mascot[i][2]), mz_resolution)<400: #or round(float(peptide_mascot[i][2]), mz_resolution)>800 or 
                continue
            temp_peptide_mascot.append(peptide_mascot[i])
        peptide_mascot=temp_peptide_mascot
        temp_peptide_mascot=0
        
        
        total_report=np.zeros((1, 5)) # 0 = openMS, 1=maxQuant, 2=dino, 3=DeepIso, 4= peaks
        
        ########################################################################
        detected_peptide=np.zeros((len(peptide_mascot), 7)) # 0 = our, 1 = peaks, 2 = maxquant, 3= charge by Peaks, 4=peaks id, 5=dino, 6=openMS
#        f=open(datapath+'/feature_list/deepIsoV2_'+dataname[test_index]+'_featureTable_v6r1_cv5_ev2r4_merged_auc','rb') #        
#        feature_table,auc_list=pickle.load(f)
#        f.close()   
#
        f=open(datapath+'/feature_list/deepIsoV2_'+dataname[test_index]+'_featureTable_v6r1_cv5_ev2r4_merged','rb') #'_featureTable_v3r4', 'wb')
        feature_table=pickle.load(f)
        f.close() 

##        conf=100
        ftr_matched_auc=np.zeros((len(peptide_mascot), 5)) # 0 = openMS, 1=maxQuant, 2=dino, 3=DeepIso, 4= peaks
#        ####################################################
#        RT_tolerance=0.2 # dino is same
#        found_ftr=0
#        detected_peptide[:, 0]=0
#        ftr_matched_auc[:, 3]=0
#        total_feature=0
#        detected_ftr_list=[]
#        detected_ftr_list.append([])
#        for i in range (1, len(peptide_mascot)):
#            detected_ftr_list.append([])
#            found=0
#            mz_exact=round(float(peptide_mascot[i][2]), mz_resolution)
#            total_feature=total_feature+1           
#            mz_range=[]
#            mz_range.append(mz_exact)    
#            tolerance_mz=0.01 #(mz_exact*10.0)/10**6  # dinosaur = 0.005 
#            mz_range.append(round(mz_exact-tolerance_mz, mz_resolution))
#            mz_range.append(round(mz_exact+tolerance_mz, mz_resolution))
#            for j in range (0, len(mz_range)):
#                mz=mz_range[j]
#                if mz in feature_table:
#                    ftr_list=feature_table[mz]
#                    for k in range (0, len(ftr_list)):
#                        ftr=ftr_list[k]
#                        ftr_z=int(ftr[len(ftr)-1][0])    
#                        peak_RT=ftr[0][1][0] 
#                        if (round(float(peptide_mascot[i][7])-RT_tolerance, 2) <= peak_RT) and (peak_RT<=round(float(peptide_mascot[i][7])+RT_tolerance, 2)): # and ftr_z==int(peptide_mascot[i][3]):
#                            found=1
#                            found_ftr=found_ftr+1
#                            detected_peptide[i, 0]=1
#                            ftr_matched_auc[i, 3]=ftr[len(ftr)-1][1]
#                            detected_ftr_list[i].append([np.abs(round(float(peptide_mascot[i][7])-peak_RT, 2)), np.abs(round(mz_exact-ftr[0][0], 2)), ftr])
#                            feature_table[mz][k][len(ftr)-1].append('f')
#                            
##                            break
##
##                    if found==1:
##                        break
#                        
#
#    #    print(found_ftr/total_feature)
#    #    
#        detected_peptide[:, 0]=0
#        ftr_matched_auc[:, 3]=0
#        for i in range (1, len(peptide_mascot)):
#            if len(detected_ftr_list[i])>0:
#                ftr=detected_ftr_list[i][0][2]
#                max_auc=ftr[len(ftr)-1][1] #ftr[0][1][3]
#                ftr_matched_auc[i, 3]=ftr[len(ftr)-1][1]
#                for j in range (1, len(detected_ftr_list[i])):
#                    ftr=detected_ftr_list[i][j][2]
#                    if ftr[len(ftr)-1][1] >max_auc:
#                        max_auc=ftr[len(ftr)-1][1] #ftr[0][1][3]
#                        ftr_matched_auc[i, 3]=ftr[len(ftr)-1][1]
#                        
#                detected_peptide[i, 0]=1
#         
    #    ##################################################################
    
        
        ##########################PEAKS###################################
        logfile=open(datapath+'feature_list/'+dataname[test_index]+'_combineIsotopes_featureList.csv', 'rb')
        peptide_feature=np.loadtxt(logfile, delimiter=',')
        logfile.close()

        feature_table_peaks=defaultdict(list)
    #    auc_list_peaks=[]
        auc_dict_peaks=defaultdict(list)
        for i in range (0, peptide_feature.shape[0]):
            if peptide_feature[i, 5]==-1 or round(peptide_feature[i, 13], 2)<min_RT: # or round(peptide_feature[i, 0], mz_resolution)>=(2000-0.50)  or round(peptide_feature[i, 0], mz_resolution) < 400: # or round(peptide_feature[i, 0], mz_resolution)>800:
                continue
            
            new_ftr=[]
            new_ftr.append(round(peptide_feature[i, 0], mz_resolution))
            new_ftr.append( round(peptide_feature[i, 13], 2))
            new_ftr.append( int(peptide_feature[i, 3]))
            new_ftr.append(peptide_feature[i, 4])
            new_ftr.append(i)
            auc_dict_peaks[round(float(peptide_feature[i, 4]))].append(i) # auc
            feature_table_peaks[round(peptide_feature[i, 0], mz_resolution)].append(new_ftr)
            total_report[0, 4]=total_report[0, 4]+1   
        auc_list_peaks=sorted(list(auc_dict_peaks.keys()), reverse=True)

        RT_tolerance=0.2 # dino is same
        found_ftr=0
        total_feature=0
        detected_peptide[:, 1]=0
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
            for j in range (0, len(mz_range)):
                mz=mz_range[j]
                if mz in feature_table_peaks:
                    ftr_list=feature_table_peaks[mz]
                    for k in range (0, len(ftr_list)):
                        ftr=ftr_list[k]
#                        if ftr[3]<auc_list_peaks[high_conf_limit]:
#                            continue
                        peak_RT=ftr[1]
                        if (round(float(peptide_mascot[i][7])-RT_tolerance, 2) <= peak_RT) and (peak_RT<=round(float(peptide_mascot[i][7])+RT_tolerance, 2)): # and int(ftr[2])==int(peptide_mascot[i][3]):
                        
                            found=1
                            detected_peptide[i, 1]=1
                            detected_peptide[i, 3]=ftr[2]
                            detected_peptide[i, 4]=ftr[4]
                            found_ftr=found_ftr+1
                            ftr_matched_auc[i, 4]=ftr[3]
                            break

                    if found==1:
                        break
        
        ########### Dinosaurs ################################################
        filename ='/data/fzohora/dilution_series_syn_pep/feature_list/dino/'+dataname[test_index]+'_dino.csv'
        # initializing the titles and peptide_mascot list
        dino_peptide_mascot = [] 
        # reading csv file
        csvfile=open(filename, 'r')
        # creating a csv reader object
        csvreader = csv.reader(csvfile)     
        # extracting each data row one by one
        for row in csvreader:
            dino_peptide_mascot.append(row)
        csvfile.close() 

        feature_table_dino=defaultdict(list)
        auc_list_dino=[]
        auc_dict_dino=defaultdict(list)
        count=0
        for i in range (1, len(dino_peptide_mascot)):
            if float(dino_peptide_mascot[i][3])<min_RT: # or round(float(dino_peptide_mascot[i][0]), mz_resolution)>=(2000-0.50) or round(float(dino_peptide_mascot[i][0]), mz_resolution)<400: # or round(float(dino_peptide_mascot[i][0]), mz_resolution)>800:
                continue
            
            ftr_z=int(dino_peptide_mascot[i][2])
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
            for j in range (0, len(mz_range)):
                mz=mz_range[j]
                if mz in feature_table_dino:
                    ftr_list=feature_table_dino[mz]
                    for k in range (0, len(ftr_list)):
                        ftr=ftr_list[k]
#                        if ftr[2]<auc_list_dino[high_conf_limit]:
#                            continue
                        ftr_z=int(ftr[len(ftr)-1])
                        peak_RT=ftr[1]
                        if (round(float(peptide_mascot[i][7])-RT_tolerance, 2) <= peak_RT) and (peak_RT<=round(float(peptide_mascot[i][7])+RT_tolerance, 2)): # and ftr_z==int(peptide_mascot[i][3]):
                            found=1
                            detected_peptide[i, 5]=1
                            found_ftr=found_ftr+1
                            ftr_matched_auc[i, 2]=ftr[2]
                            break

                    if found==1:
                        break

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
        for i in range (0, len(MQ_peptide_mascot)):
            if float(MQ_peptide_mascot[i][4])<min_RT: # or round(float(MQ_peptide_mascot[i][1]), mz_resolution)>=(2000-0.50) or round(float(MQ_peptide_mascot[i][1]), mz_resolution)<400: #or round(float(MQ_peptide_mascot[i][1]), mz_resolution)>800 :
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
            for j in range (0, len(mz_range)):
                mz=mz_range[j]
                if mz in feature_table_mq:
                    ftr_list=feature_table_mq[mz]
                    for k in range (0, len(ftr_list)):
                        ftr=ftr_list[k]
#                        if ftr[2]<auc_list_MQ[high_conf_limit]:
#                            continue
                        peak_RT=ftr[1]
                        if (round(float(peptide_mascot[i][7])-RT_tolerance, 2) <= peak_RT) and (peak_RT<=round(float(peptide_mascot[i][7])+RT_tolerance, 2)): # and int(ftr[len(ftr)-1])==int(peptide_mascot[i][3]):

                            found=1
                            detected_peptide[i, 2]=1
                            found_ftr=found_ftr+1
                            ftr_matched_auc[i, 1]=ftr[2]
                            break

                    if found==1:
                        break
                        
    ########################################################################
        f=open(datapath+'feature_list/openMS/'+dataname[test_index]+'_openMS_features', 'rb') # open('/media/fzohora/USB20FD/raw/mzml/'+dataname[test_index]+'_openMS_features', 'rb')
        ft_openMS=pickle.load(f, encoding='latin1')
        f.close()   
        auc_dict_op=defaultdict(list)
        feature_table_openMS=defaultdict(list)
        auc_list_openMS=[]
        count=0
        for i in range (0, ft_openMS.shape[0]):
            if float(ft_openMS[i][1])<min_RT: #  or round(ft_openMS[i][0], mz_resolution)>=(2000-0.50) or round(ft_openMS[i][0], mz_resolution)<400: # or round(ft_openMS[i][0], mz_resolution)>800:
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
        for i in range (1, len(peptide_mascot)):
                
            total_feature=total_feature+1  
            mz_exact=round(float(peptide_mascot[i][2]), mz_resolution)
            mz_range=[]
            mz_range.append(mz_exact)
                
            tolerance_mz=0.01 #(mz_exact*10.0)/10**6 # dinosaur = 0.005 
            mz_range.append(round(mz_exact-tolerance_mz, mz_resolution))
            mz_range.append(round(mz_exact+tolerance_mz, mz_resolution))
            found=0 
            for j in range (0, len(mz_range)):
                mz=mz_range[j]
                if mz in feature_table_openMS:
                    ftr_list=feature_table_openMS[mz]
                    for k in range (0, len(ftr_list)):
                        ftr=ftr_list[k]
#                        if ftr[2]<auc_list_openMS[high_conf_limit]:
#                            continue
                        peak_RT=ftr[1]
                        if (round(float(peptide_mascot[i][7])-RT_tolerance, 2) <= peak_RT) and (peak_RT<=round(float(peptide_mascot[i][7])+RT_tolerance, 2)): # and int(ftr[len(ftr)-1])==int(peptide_mascot[i][3]):
                        
                            found=1
                            detected_peptide[i, 6]=1
                            found_ftr=found_ftr+1
                            ftr_matched_auc[i, 0]=ftr[2]
                            break

                    if found==1:
                        break


        pearson_coeff=np.zeros((len(detected_peptide), 5)) 
        pearson_index=0
        for i in range(1, len(detected_peptide)):
            if detected_peptide[i, 0]==1 and detected_peptide[i, 1]==1 and detected_peptide[i, 2]==1 and detected_peptide[i, 5]==1 and detected_peptide[i, 6]==1:
                pearson_coeff[pearson_index, :]=ftr_matched_auc[i,:]
                pearson_index=pearson_index+1
                
        pearson_coeff=pearson_coeff[0:pearson_index]
        
#---------------------------auc ---------------------------------
    key_list=feature_table.keys()
    auc_dict=defaultdict(list)
    for mz in key_list:
        ftr_list=sorted(feature_table[mz])
        for k in range (0, len(ftr_list)):
            ftr=ftr_list[k]
            auc_ftr=ftr[len(ftr)-1][1]
            if auc_ftr==0:
                continue
            auc_dict[round(math.log(auc_ftr, 10), 2)].append(auc_ftr)

    
    
    
    key_list=sorted(auc_dict.keys())
    auc_dist=np.zeros((len(key_list)))
    for i in range(0, len(key_list)):
        auc_dist[i]=len(auc_dict[key_list[i]])
    
    
    identified_auc_dict=defaultdict(list)
    for k in range (0, ftr_matched_auc.shape[0]):
        auc_ftr=ftr_matched_auc[k, 3]
        if auc_ftr==0:
            continue
        identified_auc_dict[round(math.log(auc_ftr, 10), 2)].append(auc_ftr)

    identified_key_list=sorted(identified_auc_dict.keys())
    identified_auc_dist=np.zeros((len(key_list)))
    for i in range(0, len(key_list)):
        if key_list[i] in identified_key_list:
            identified_auc_dist[i]=len(identified_auc_dict[key_list[i]])
        else:
            identified_auc_dist[i]=0
    
#    plt.plot(key_list, identified_auc_dist) #, auc_dist)
#    plt.plot(key_list, auc_dist)
    plt.plot(key_list, auc_dist, key_list, identified_auc_dist)
    plt.axis([0, max(key_list), 0, max(auc_dist)])
    plt.show(block=False)
    plt.savefig('DeepIsoV2_auc.png')
    

    
    plt.bar(identified_key_list, identified_auc_dist)
#    plt.axis([0, max(identified_key_list), 0, max(identified_auc_dist)])
    plt.show(block=False)


######## m/z vs frequency #############

#    key_list=feature_table.keys()
#    mz_dict=defaultdict(list)
#    total_feature=0
#    for mz in key_list:
#        ftr_list=sorted(feature_table[mz])
#        for k in range (0, len(ftr_list)):
#            ftr=ftr_list[k]
#            ftr_mz=ftr[0][0]
#            mz_dict[round(ftr_mz, 3)].append(ftr_mz)
#            total_feature=total_feature+1
#
#
#    key_list=sorted(mz_dict.keys())
#    mz_dist=np.zeros((len(key_list)))
#    for i in range(0, len(key_list)):
#        mz_dist[i]=len(mz_dict[key_list[i]])/total_feature

#----------------------------------------------------------------------------------
    mz_dist_mascot=[]    
    for i in range (1, len(peptide_mascot)):    
        mz_dist_mascot.append(round(float(peptide_mascot[i][2]), 3))


    key_list=feature_table_dino.keys()
    mz_dist_dino=[]
    for mz in key_list:
        ftr_list=sorted(feature_table_dino[mz])
        for k in range (0, len(ftr_list)):
            ftr=ftr_list[k]
            ftr_mz=round(ftr[0], 3)
            mz_dist_dino.append(ftr_mz)

    key_list=feature_table_mq.keys()
    mz_dist_mq=[]
    for mz in key_list:
        ftr_list=sorted(feature_table_mq[mz])
        for k in range (0, len(ftr_list)):
            ftr=ftr_list[k]
            ftr_mz=round(ftr[0], 3) 
            mz_dist_mq.append(ftr_mz)

    key_list=feature_table_openMS.keys()
    mz_dist_openMS=[]
    for mz in key_list:
        ftr_list=sorted(feature_table_openMS[mz])
        for k in range (0, len(ftr_list)):
            ftr=ftr_list[k]
            ftr_mz=round(ftr[0], 3) 
            mz_dist_openMS.append(ftr_mz)


    key_list=feature_table_peaks.keys()
    mz_dist_peaks=[]
    for mz in key_list:
        ftr_list=sorted(feature_table_peaks[mz])
        for k in range (0, len(ftr_list)):
            ftr=ftr_list[k]
            ftr_mz=round(ftr[0], 3) 
            mz_dist_peaks.append(ftr_mz)

    
    key_list=feature_table.keys()
    mz_dist=[]
    count_noise=np.zeros((2))
    for mz in key_list:
        ftr_list=sorted(feature_table[mz])
        for k in range (0, len(ftr_list)):
            ftr=ftr_list[k]
            ftr_mz=round(ftr[0][0], 3) 
            mz_dist.append(ftr_mz)
            if ftr_mz<500:
                count_noise[0]=count_noise[0]+1
            elif ftr_mz>1900:
                count_noise[1]=count_noise[1]+1    
        
        
        
    SMALL_SIZE = 12
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
        
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.hist(mz_dist, 20, alpha=0.9, density=True, label='DeepIsoV2', histtype='step')
    plt.hist(mz_dist_mascot, 20, alpha=0.9, density=True, label='Identified', histtype='step')
    plt.legend(loc='upper right')
    plt.show(block=False)
    plt.savefig('DeepIsoV2_mz.png')
    
    plt.hist(mz_dist_dino, 20, alpha=0.9, density=True, label='Dinosaurs', histtype='step')
    plt.hist(mz_dist_mascot, 20, alpha=0.9, density=True, label='Identified', histtype='step')
    plt.legend(loc='upper right')
    plt.show(block=False)
    plt.savefig('Dino_mz.png')
    

    plt.hist(mz_dist_mq, 20, alpha=0.9, density=True, label='MaxQuant', histtype='step')
    plt.hist(mz_dist_mascot, 20, alpha=0.9, density=True, label='Identified', histtype='step')
    plt.legend(loc='upper right')
    plt.show(block=False)
    plt.savefig('MQ_mz.png')
    
    plt.hist(mz_dist_openMS, 20, alpha=0.9, density=True, label='OpenMS', histtype='step')
    plt.hist(mz_dist_mascot, 20, alpha=0.9, density=True, label='Identified', histtype='step')
    plt.legend(loc='upper right')
#    plt.show(block=False)
    plt.savefig('OpenMS_mz.png')
    
    plt.hist(mz_dist_peaks, 20, alpha=0.9, density=True, label='PEAKS', histtype='step')
    plt.hist(mz_dist_mascot, 20, alpha=0.9, density=True, label='Identified', histtype='step')
    plt.legend(loc='upper right')
    plt.show(block=False)
    plt.savefig('PEAKS_mz.png')
#----------------------------Mass----------------------------------------------------------------------
    proton_mass= 1.00727567
    mass_dist_mascot=[]    
    for i in range (1, len(peptide_mascot)): 
        charge=float(peptide_mascot[i][3])
        mass=round((round(float(peptide_mascot[i][2]), 3)-proton_mass)*charge, 3)
        mass_dist_mascot.append(mass)


    key_list=feature_table_dino.keys()
    mass_dist_dino=[]
    for mz in key_list:
        ftr_list=sorted(feature_table_dino[mz])
        for k in range (0, len(ftr_list)):
            ftr=ftr_list[k]
            ftr_mz=round(ftr[0], 3) #ftr[0][0]
            charge=float(ftr[3])
            mass=round((ftr_mz-proton_mass)*charge, 3)        
#            if ftr_mz>1200:
#                continue
#            if ftr[len(ftr)-1][0]==1:
#                continue
            mass_dist_dino.append(mass)

    key_list=feature_table_mq.keys()
    mass_dist_mq=[]
    for mz in key_list:
        ftr_list=sorted(feature_table_mq[mz])
        for k in range (0, len(ftr_list)):
            ftr=ftr_list[k]
            ftr_mz=round(ftr[0], 3) #ftr[0][0]
            charge=float(ftr[3])
            mass=round((ftr_mz-proton_mass)*charge, 3)        
            mass_dist_mq.append(mass)

    key_list=feature_table_openMS.keys()
    mass_dist_openMS=[]
    for mz in key_list:
        ftr_list=sorted(feature_table_openMS[mz])
        for k in range (0, len(ftr_list)):
            ftr=ftr_list[k]
            ftr_mz=round(ftr[0], 3) #ftr[0][0]
            charge=float(ftr[3])
            mass=round((ftr_mz-proton_mass)*charge, 3)                    
            mass_dist_openMS.append(mass)


    key_list=feature_table_peaks.keys()
    mass_dist_peaks=[]
    for mz in key_list:
        ftr_list=sorted(feature_table_peaks[mz])
        for k in range (0, len(ftr_list)):
            ftr=ftr_list[k]
            ftr_mz=round(ftr[0], 3) #ftr[0][0]
            charge=float(ftr[2])
            mass=round((ftr_mz-proton_mass)*charge, 3)        
            mass_dist_peaks.append(mass)

    
    key_list=feature_table.keys()
    mass_dist=[]
    for mz in key_list:
        ftr_list=sorted(feature_table[mz])
        for k in range (0, len(ftr_list)):
            ftr=ftr_list[k]
            ftr_mz=round(ftr[0][0], 3) #ftr[0][0]
            charge=ftr[len(ftr)-1][0]
            mass=round((ftr_mz-proton_mass)*charge, 3)        
            mass_dist.append(mass)
    
        
    plt.hist(mass_dist, 20, alpha=0.9, density=True, label='DeepIsoV2', histtype='step')
    plt.hist(mass_dist_mascot, 20, alpha=0.9, density=True, label='Identified', histtype='step')
    plt.legend(loc='upper right')
    plt.show(block=False)
    plt.savefig('DeepIsoV2_mass.png')
    
    plt.hist(mass_dist_dino, 20, alpha=0.9, density=True, label='Dinosaurs', histtype='step')
    plt.hist(mass_dist_mascot, 20, alpha=0.9, density=True, label='Identified', histtype='step')
    plt.legend(loc='upper right')
    plt.show(block=False)
    plt.savefig('Dinosaurs_mass.png')

    plt.hist(mass_dist_mq, 20, alpha=0.9, density=True, label='MaxQuant', histtype='step')
    plt.hist(mass_dist_mascot, 20, alpha=0.9, density=True, label='Identified', histtype='step')
    plt.legend(loc='upper right')
    plt.show(block=False)
    plt.savefig('MaxQuant_mass.png')
    
    plt.hist(mass_dist_openMS, 20, alpha=0.9, density=True, label='OpenMS', histtype='step')
    plt.hist(mass_dist_mascot, 20, alpha=0.9, density=True, label='Identified', histtype='step')
    plt.legend(loc='upper right')
    plt.show(block=False)
    plt.savefig('OpenMS_mass.png')
    
    plt.hist(mass_dist_peaks, 20, alpha=0.9, density=True, label='PEAKS', histtype='step')
    plt.hist(mass_dist_mascot, 20, alpha=0.9, density=True, label='Identified', histtype='step')
    plt.legend(loc='upper right')
    plt.show(block=False)
    plt.savefig('PEAKS_mass.png')


#----------------------------- RT ---------------------------------------------------
    RT_dist_mascot=[]    
    for i in range (1, len(peptide_mascot)):    
        RT_dist_mascot.append(round(float(peptide_mascot[i][7]), 2))


    key_list=feature_table_dino.keys()
    RT_dist_dino=[]
    for mz in key_list:
        ftr_list=sorted(feature_table_dino[mz])
        for k in range (0, len(ftr_list)):
            ftr=ftr_list[k]
            ftr_RT=round(ftr[1], 2) 
            RT_dist_dino.append(ftr_RT)

    key_list=feature_table_mq.keys()
    RT_dist_mq=[]
    for mz in key_list:
        ftr_list=sorted(feature_table_mq[mz])
        for k in range (0, len(ftr_list)):
            ftr=ftr_list[k]
            ftr_RT=round(ftr[1], 2) #ftr[0][0]
            RT_dist_mq.append(ftr_RT)

    key_list=feature_table_openMS.keys()
    RT_dist_openMS=[]
    for mz in key_list:
        ftr_list=sorted(feature_table_openMS[mz])
        for k in range (0, len(ftr_list)):
            ftr=ftr_list[k]
            ftr_RT=round(ftr[1], 2) #ftr[0][0]
            RT_dist_openMS.append(ftr_RT)


    key_list=feature_table_peaks.keys()
    RT_dist_peaks=[]
    for mz in key_list:
        ftr_list=sorted(feature_table_peaks[mz])
        for k in range (0, len(ftr_list)):
            ftr=ftr_list[k]
            ftr_RT=round(ftr[1], 2) #ftr[0][0]
            RT_dist_peaks.append(ftr_RT)

    
    key_list=feature_table.keys()
    RT_dist=[]
    for mz in key_list:
        ftr_list=sorted(feature_table[mz])
        for k in range (0, len(ftr_list)):
            ftr=ftr_list[k]
            ftr_RT=round(ftr[0][1][0], 2) #ftr[0][0]
            RT_dist.append(ftr_RT)
    
        
    plt.hist(RT_dist, 20, alpha=0.9, density=True, label='DeepIsoV2', histtype='step')
    plt.hist(RT_dist_mascot, 20, alpha=0.9, density=True, label='Identified', histtype='step')
    plt.legend(loc='upper right')
    plt.show(block=False)
    plt.savefig('DeepIsoV2_RT.png')
    
    plt.hist(RT_dist_dino, 20, alpha=0.9, density=True, label='Dinosaurs', histtype='step')
    plt.hist(RT_dist_mascot, 20, alpha=0.9, density=True, label='Identified', histtype='step')
    plt.legend(loc='upper right')
    plt.show(block=False)
    plt.savefig('Dinosaurs_RT.png')

    plt.hist(RT_dist_mq, 20, alpha=0.9, density=True, label='MaxQuant', histtype='step')
    plt.hist(RT_dist_mascot, 20, alpha=0.9, density=True, label='Identified', histtype='step')
    plt.legend(loc='upper left')
    plt.show(block=False)
    plt.savefig('MaxQuant_RT.png')
    
    plt.hist(RT_dist_openMS, 20, alpha=0.9, density=True, label='OpenMS', histtype='step')
    plt.hist(RT_dist_mascot, 20, alpha=0.9, density=True, label='Identified', histtype='step')
    plt.legend(loc='upper right')
    plt.show(block=False)
    plt.savefig('OpenMS_RT.png')

    plt.hist(RT_dist_peaks, 20, alpha=0.9, density=True, label='PEAKS', histtype='step')
    plt.hist(RT_dist_mascot, 20, alpha=0.9, density=True, label='Identified', histtype='step')
    plt.legend(loc='upper right')
    plt.show(block=False)
    plt.savefig('PEAKS_RT.png')
