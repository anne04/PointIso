from __future__ import division
from __future__ import print_function
#import math
import numpy as np
import csv
#import scipy.misc
import pickle
#import pandas as pd
#from time import time
##from shapely.geometry import Polygon
#import gc
from collections import defaultdict


path='/data/fzohora/dilution_series_syn_pep/'  #'/media/anne/Study/bsi/dilution_series_syn_peptide/feature_list/' #'/data/fzohora/water_raw_ms1/'
dataname=['130124_dilA_1_01','130124_dilA_1_02','130124_dilA_1_03','130124_dilA_1_04', 
'130124_dilA_2_01','130124_dilA_2_02','130124_dilA_2_03','130124_dilA_2_04','130124_dilA_2_05','130124_dilA_2_06','130124_dilA_2_07',
'130124_dilA_3_01','130124_dilA_3_02','130124_dilA_3_03','130124_dilA_3_04','130124_dilA_3_05','130124_dilA_3_06','130124_dilA_3_07',
'130124_dilA_4_01','130124_dilA_4_02','130124_dilA_4_03','130124_dilA_4_04','130124_dilA_4_05','130124_dilA_4_06','130124_dilA_4_07',
'130124_dilA_5_01','130124_dilA_5_02','130124_dilA_5_03','130124_dilA_5_04',
'130124_dilA_6_01','130124_dilA_6_02','130124_dilA_6_03','130124_dilA_6_04',
'130124_dilA_7_01','130124_dilA_7_02','130124_dilA_7_03','130124_dilA_7_04',
rte'130124_dilA_8_01','130124_dilA_8_02','130124_dilA_8_03','130124_dilA_8_04',
'130124_dilA_9_01','130124_dilA_9_02','130124_dilA_9_03','130124_dilA_9_04',
'130124_dilA_10_01','130124_dilA_10_02', '130124_dilA_10_03', '130124_dilA_10_04', 
'130124_dilA_11_01', '130124_dilA_11_02', '130124_dilA_11_03', '130124_dilA_11_04', 
'130124_dilA_12_01', '130124_dilA_12_02', '130124_dilA_12_03', '130124_dilA_12_04'] 
delim=','
mz_resolution=5
mz_unit=0.00001
RT_window=15
isotope_gap=np.zeros((10))
isotope_gap[0]=0.00001
isotope_gap[1]=1.00000
isotope_gap[2]=0.50000
isotope_gap[3]=0.33333
isotope_gap[4]=0.25000
isotope_gap[5]=0.20000
isotope_gap[6]=0.16667
isotope_gap[7]=0.14286
isotope_gap[8]=0.12500
isotope_gap[9]=0.11111

for data_index in (25, 26, 27, 28,  29, 30, 31, 32,  33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 52): 
#for data_index in (0, 25, 26, 27, 28): # 19, 20, 21
    print(dataname[data_index])
    count=0
    print('trying to load ms1 record')
    

    f=open(path+'feature_list/pointCloud_'+dataname[data_index]+'_ms1_record_mz5', 'rb')
    sorted_mz_list,  maxI=pickle.load(f)
    f.close()  
    
    f=open(path+'feature_list/pointCloud_'+dataname[data_index]+'_RT_index_new_mz5', 'rb')
    RT_index=pickle.load(f)
    f.close()   
    print('done!')
    
    RT_list=sorted(RT_index.keys())
    
    RT_index_array=dict()
    for i in range (0, len(RT_list)):
        RT_index_array[round(RT_list[i], 2)]=i

    filename_feature=dataname[data_index]+'.raw.fea.isotopes.csv' # feature_file[data_index]
    rows= [] 
    csvfile=open(path+'feature_list/PEAKs/'+dataname[data_index]+'.raw.fea.csv', 'r')
    csvreader = csv.reader(csvfile)     
    for row in csvreader:
        rows.append(row)
    csvfile.close() 
    total_feature=len(rows)-1
    print('total features by peaks: %d'%total_feature)
#    total_feature=feature_count[data_index]
    peptide_feature=np.zeros((total_feature, 16)) 
    #0=mz, 1=rtstsr, 2=rtend, 3=z, 4=auc, 5=kept or removed or non_overlapping_feature_id, 
    #6=end_mz, 7=min_rt, 8=max_rt, 9=endof2ndisotope, 10=start of second isotope, 11=overlapped/not, 
    # 12=maxI, 13 = meanRT, 14=isotope_no, 15=maxquant_id, #16=total datapoints, 17, RT_extent
    min_RT=10.0
    avoid=[]
    max_datapoints=0 #poz
    min_datapoints=10000 #poz
    poz_datapoints=[]
    poz_datapoints_window=[]
    f = open(path+'feature_list/PEAKs/'+filename_feature, 'r')
    line=f.readline()
    line=f.readline()
    i=0
    while line!='':
        temp=line.split(',')
        id=temp[0] # mz, rtstsr, rtend, z, auc
#        print('id:%d'%int(id))
        mz_value=round(float(temp[2]), mz_resolution) #mz
        mz_tolerance=(mz_value*10.0)/10**6
        mz_start=round(mz_value-mz_tolerance, mz_resolution) # 10ppm correction
        first_mz=mz_start
        mz_end=round(mz_value+mz_tolerance, mz_resolution) #mz_start+0.3333+0.3333
        
        min_RT=round(float(temp[8]), 2) #st
        max_RT=round(float(temp[9]), 2) #en
        
        z=int(temp[5]) #charge

       
        isotope_no=0
        count=0
        ################################
#        if min_RT==max_RT: #CHECK
#            continue
        
        i=0
        if min_RT in RT_index:
            min_RT=min_RT
        else:
            while RT_list[i]<min_RT: 
                i=i+1
            min_RT=RT_list[i] #inside
        
        i=0
        if max_RT in RT_index:
            max_RT=max_RT
        else:
            while RT_list[i]<max_RT:
                i=i+1
            max_RT=RT_list[i-1] #inside        
        
        RT_index_start=RT_index_array[round(min_RT, 2)]      
        first_RT=RT_index_start
        RT_index_end=RT_index_array[round(max_RT, 2)]  
        
        # now you know the (mz_start and mz_end) and (RT_start and RT_end). Therefore, just fill out the datapoints
        
        count2=0
        for RT_idx in range (RT_index_start, RT_index_end+1):
            mz_value=mz_start
            while mz_value<=mz_end and (mz_value not in  RT_index[round(RT_list[RT_idx], 2)]): 
                mz_value=round(mz_value+mz_unit, mz_resolution)
                
            #got it.
            if mz_value>mz_end:
                continue
                
            datapoint=RT_index[round(RT_list[RT_idx], 2)][mz_value]
            datapoint.append(z)
            RT_index[round(RT_list[RT_idx], 2)][mz_value]=datapoint
            count=count+1
            if (RT_idx-first_RT)<15:
                count2=count2+1
            next_mz_idx=int(datapoint[1])+1         
            mz_value= sorted_mz_list[RT_idx][next_mz_idx]
            
            while mz_value<=mz_end:
                datapoint=RT_index[round(RT_list[RT_idx], 2)][mz_value]
                datapoint.append(z)
                RT_index[round(RT_list[RT_idx], 2)][mz_value]=datapoint
                count=count+1
                if (RT_idx-first_RT)<15:
                    count2=count2+1
                
                next_mz_idx=int(datapoint[1])+1
                mz_value= sorted_mz_list[RT_idx][next_mz_idx]    
        ################################
        
        line=f.readline() 
        while line!='':
            temp=line.split(',')
            if temp[0]!=id:
                break
            #else this isotope belongs to this same peptide        
            isotope_no=isotope_no+1
            mz_start=round(mz_start+isotope_gap[z], mz_resolution)
            mz_end=round(mz_end+isotope_gap[z], mz_resolution) #mz_start+0.3333+0.3333

            
            min_RT=round(float(temp[8]), 2) #st
            max_RT=round(float(temp[9]), 2) #en            
        
            i=0
            if min_RT in RT_index:
                min_RT=min_RT
            else:
                while RT_list[i]<min_RT: 
                    i=i+1
                min_RT=RT_list[i] #inside
            
            i=0
            if max_RT in RT_index:
                max_RT=max_RT
            else:
                while RT_list[i]<max_RT:
                    i=i+1
                max_RT=RT_list[i-1] #inside        
            
            RT_index_start=RT_index_array[round(min_RT, 2)]       
            RT_index_end=RT_index_array[round(max_RT, 2)]  
            
            # now you know the (mz_start and mz_end) and (RT_start and RT_end). Therefore, just fill out the datapoints
            
            
            for RT_idx in range (RT_index_start, RT_index_end+1):
                mz_value=mz_start
                while mz_value<=mz_end and (mz_value not in  RT_index[round(RT_list[RT_idx], 2)]): 
                    mz_value=round(mz_value+mz_unit, mz_resolution)
                
                if mz_value>mz_end:
                    continue
                #got it.
                datapoint=RT_index[round(RT_list[RT_idx], 2)][mz_value]
                datapoint.append(z)
                RT_index[round(RT_list[RT_idx], 2)][mz_value]=datapoint
                count=count+1
                if (RT_idx-first_RT)<15 and round(mz_start-first_mz, mz_resolution)<=2.99999:
                    count2=count2+1
                next_mz_idx=int(datapoint[1])+1
                
                mz_value= sorted_mz_list[RT_idx][ next_mz_idx]
                while mz_value<=mz_end:  
                    datapoint=RT_index[round(RT_list[RT_idx], 2)][mz_value]
                    datapoint.append(z)
                    RT_index[round(RT_list[RT_idx], 2)][mz_value]=datapoint
                    count=count+1
                    if (RT_idx-first_RT)<15  and round(mz_start-first_mz, mz_resolution)<=2.99999:
                        count2=count2+1
                    next_mz_idx=int(datapoint[1])+1
                    mz_value= sorted_mz_list[RT_idx][next_mz_idx]    
            ################################
            line=f.readline()  
        #one feature done
        poz_datapoints.append(count)
        poz_datapoints_window.append(count2)
            
    f.close() 

    print('max datapoints %g, min datapoints %g, avg %g, mode %g'%(max(poz_datapoints), min(poz_datapoints), np.mean(poz_datapoints), max(set(poz_datapoints), key=poz_datapoints.count)))  
    print('window: max datapoints %g, min datapoints %g, avg %g, mode %g'%(max(poz_datapoints_window), min(poz_datapoints_window), np.mean(poz_datapoints_window), max(set(poz_datapoints_window), key=poz_datapoints_window.count)))  

    f=open(path+'feature_list/pointCloud_'+dataname[data_index]+'_RT_index_new_mz5', 'wb')
    pickle.dump(RT_index,  f, protocol=3)
    f.close()

        


