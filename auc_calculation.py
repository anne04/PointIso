from __future__ import division
from __future__ import print_function
#import math
#import csv
#from time import time
import pickle
import numpy as np
#from sklearn import metrics
#from collections import deque
from collections import defaultdict
#import sys
#import copy
#import scipy.misc
#import scipy.stats
import bisect
import gzip

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
for test_index in range (0, 57):
    print(dataname[test_index])
    f=gzip.open(datapath+'feature_list/'+'pointCloud_'+dataname[test_index]+'_ms1_record_mz5', 'rb')
    sorted_mz_list, maxI=pickle.load(f)
    f.close()   


    f=gzip.open(datapath+'feature_list/pointCloud_'+dataname[test_index]+'_RT_index_new_mz5', 'rb')
    RT_index=pickle.load(f)
    f.close()  
    
    RT_list=sorted(RT_index.keys())
    max_RT=RT_list[len(RT_list)-1]
    min_RT=10    
    RT_index_array=dict()
    for i in range (0, len(RT_list)):
        RT_index_array[round(RT_list[i], 2)]=i
    max_mz=0
    min_mz=1000
    for i in range (0, len(sorted_mz_list)):
        mz_I_list=sorted_mz_list[i]
        mz=mz_I_list[len(mz_I_list)-1]
        if mz>max_mz:
            max_mz=mz
        mz=mz_I_list[0]
        if mz<min_mz:
            min_mz=mz

    rt_search_index=0
    while(RT_list[rt_search_index]<min_RT):
        rt_search_index=rt_search_index+1
    print('preprocess done')
#    f=open(datapath+'/feature_list/deepIsoV2_'+dataname[test_index]+'_featureTable_v6r1_cv5_ev2r6b','rb') #_merged
    f=open(datapath+'/feature_list/deepIsoV2_'+dataname[test_index]+'_featureTable_v6r1_cv5_ev2r6b_merged','rb') #_merged
    feature_table=pickle.load(f)
    #    print('feature table restored')
    auc_dict=defaultdict(list)
    debug=0
    max_num_iso=0
    auc_list=[]
    mz_list=list(feature_table.keys())
    for i in range (0, len(mz_list)):
#            print(i)
        ftr_list=feature_table[mz_list[i]]
#        total_report[0, 3]=total_report[0, 3]+len(ftr_list)
        for f in range (0, len(ftr_list)):
            ftr=ftr_list[f]
            ftr_auc=0
            noise_flag=0
            for iso in range (0, len(ftr)-1):
                isotope=ftr[iso]
                mz_point=isotope[0]
                mz_tolerance=round((mz_point*10.0)/10**6,4) # mz_resolution)
                #mz_s=round(min(isotope[1][4])-mz_tolerance-pow(0.1, 3)+pow(0.1, 3+1)*5, 4) #  
                mz_s=round(mz_point-mz_tolerance-pow(0.1, 3)+pow(0.1, 3+1)*5, 4)
                #mz_e=round(max(isotope[1][4])+mz_tolerance+pow(0.1, 3+1)*4, 4) 
                mz_e=round(mz_point+mz_tolerance+pow(0.1, 3+1)*4, 4)
                RT_peak=round(isotope[1][0], 2)        
                # 7 step before, peak, 7 step after
    #                            count=count+1                          
                #RT_s=max(RT_index_array[RT_peak]-7, 0) #   
                RT_s=RT_index_array[isotope[1][1]] #
                #RT_e=min(RT_index_array[RT_peak]+7, len(RT_list)) # max( RT_index_array[isotope[1][2]], min(RT_s+RT_window, len(RT_list)) ) #  
                RT_e=RT_index_array[isotope[1][2]] #
                mz_dict=[]
                rt_row=0
                for RT_idx in range (RT_s,RT_e):
                    mz_dict.append([])  
                    if RT_idx<0 or RT_idx>(len(RT_list)-1):
                        rt_row=rt_row+1
                        continue
                                           
                    mz_value=mz_s

                    find_mz_idx_start= bisect.bisect_left(sorted_mz_list[RT_idx], mz_value)
                    if len(sorted_mz_list[RT_idx])==find_mz_idx_start or round(sorted_mz_list[RT_idx][find_mz_idx_start], 4)>mz_e:
                        rt_row=rt_row+1
                        continue
                    mz_value=round(sorted_mz_list[RT_idx][find_mz_idx_start] , 4)
    #                        print('rt %d, mz_start %g'%(RT_idx, mz_value))
                    
                    
                    datapoint=RT_index[round(RT_list[RT_idx], 2)][mz_value]    
                    intensity=datapoint[0] #((-0)/(maxI-0))*255 #round(, 2) # scale it to the grey value
                    mz_dict[rt_row].append(intensity)
                  
                    next_mz_idx=int(datapoint[1])+1         
                    mz_value= round(sorted_mz_list[RT_idx][next_mz_idx], 4)
                    while mz_value<=mz_e:                    
                        datapoint=RT_index[round(RT_list[RT_idx], 2)][mz_value]    
                        intensity=datapoint[0] #((-0)/(maxI-0))*255 #round(, 2) # scale it to the grey value
                        mz_dict[rt_row].append(intensity)
                        next_mz_idx=int(datapoint[1])+1         
                        mz_value= round(sorted_mz_list[RT_idx][next_mz_idx], 4)
                    # after this is done, we have the list of mz, for this RT
                    rt_row=rt_row+1

                          
                stripe_x=np.zeros(rt_row)
                stripe_y=np.zeros(rt_row)
                rt_row=0
                for RT_idx in range (RT_s,RT_e):
                    stripe_x[rt_row]=RT_list[RT_idx]
                    if len(mz_dict[rt_row])>0:
                        stripe_y[rt_row]=np.sum(mz_dict[rt_row])
                    rt_row=rt_row+1
                    
                try:
                    this_auc=np.sum(stripe_y) #np.max(stripe_y) #metrics.auc(stripe_x, stripe_y) #
                    ftr_auc=ftr_auc+this_auc        
                except:
                    ftr_auc=ftr_auc+0
    ###########################
        
            feature_table[mz_list[i]][f][len(ftr)-1].append(ftr_auc)
            auc_dict[round(ftr_auc)].append((i, f))
            

    auc_list=sorted(list(auc_dict.keys()), reverse=True)
    
    f=gzip.open(datapath+'/feature_list/deepIsoV2_'+dataname[test_index]+'_featureTable_v6r1_cv5_ev2r6b_merged_auc_exact_mz_fullRT','wb') #with 10ppm, rullRT
#    f=gzip.open(datapath+'/feature_list/deepIsoV2_'+dataname[test_index]+'_featureTable_v6r1_cv5_ev2r6b_merged_auc_NO_mztolerance_fullwidth','wb') #with 0 ppm, +-7RT
#    f=gzip.open(datapath+'/feature_list/deepIsoV2_'+dataname[test_index]+'_featureTable_v6r1_cv5_ev2r6b_merged_auc_fullwidth_fullRT','wb')
#    f=gzip.open(datapath+'/feature_list/deepIsoV2_'+dataname[test_index]+'_featureTable_v6r1_cv5_ev2r6b_merged_auc_NO_mztolerance_fullwidth_fullRT','wb')
    pickle.dump([feature_table,auc_list], f, protocol=2)
    f.close()
