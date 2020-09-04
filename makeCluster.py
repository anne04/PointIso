#scp -r fzohora@ming-gpu-1.cs.uwaterloo.ca:/data/fzohora/dilution_series_syn_pep/feature_list/130124_dilA_11_01_ms1_record  /home/anne/Desktop/bsi/update/130124_dilA_11_01_ms1_record
#scp -r fzohora@ming-gpu-1.cs.uwaterloo.ca:/data/fzohora/dilution_series_syn_pep/feature_list/130124_dilA_11_01_combineIsotopes_info  /home/anne/Desktop/bsi/update/130124_dilA_11_01_combineIsotopes_info
from __future__ import division
from __future__ import print_function
#import tensorflow as tf
#import math
#from time import time
import pickle
import numpy as np
from collections import deque
from collections import defaultdict
import copy
#import scipy.misc
#import scipy.stats as stats
#import sys
#from sklearn import metrics
import bisect
import gc

isotope_gap=np.zeros((10))
isotope_gap[0]=0.00001
isotope_gap[1]=1.0000
isotope_gap[2]=0.5000
isotope_gap[3]=0.3333
isotope_gap[4]=0.2500
isotope_gap[5]=0.2000
isotope_gap[6]=0.1667
isotope_gap[7]=0.1429
isotope_gap[8]=0.1250
isotope_gap[9]=0.1111



RT_window=15
mz_resolution=4
total_class=10
RT_unit=0.01
mz_unit=0.0001

modelpath='/data/fzohora/dilution_series_syn_pep/model/'
datapath='/data/fzohora/dilution_series_syn_pep/'      #'/data/fzohora/water/' #'/media/anne/Study/study/PhD/bsi/update/data/water/'  #
dataname=['130124_dilA_1_01','130124_dilA_1_02', '130124_dilA_1_03', '130124_dilA_1_04',
'130124_dilA_2_01','130124_dilA_2_02','130124_dilA_2_03','130124_dilA_2_04','130124_dilA_2_05','130124_dilA_2_06','130124_dilA_2_07',
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

#test_index=int(sys.argv[1])

######################################################################
fc_size=4
num_class=10
state_size = fc_size
num_neurons= num_class #mz_window*RT_window

#for test_index in (23,28,31,34,40,44,48,52,56): #(2,7,14,23, 28, 31,34,40,44,48,52,56): # 
for test_index in range (9,10):  #
#for test_index in range (10,len(dataname)):
    ####################################################################
    print('scanning test ms: '+dataname[test_index])
    print('trying to load ms1 record')
    f=open(datapath+'feature_list/'+'pointCloud_'+dataname[test_index]+'_ms1_record_mz5', 'rb')
    RT_mz_I_dict, sorted_mz_list, maxI=pickle.load(f)
    f.close()   
    print('done!')
    gc.collect()
    #############################
    f=open(datapath+'feature_list/pointCloud_'+dataname[test_index]+'_RT_index_new_mz5', 'rb')
    RT_index=pickle.load(f)
    f.close()  
    
    print('data restore done')

    ###########################
    #scan ms1_block and record the cnn outputs in list_dict[z]: hash table based on m/z
    #for each m/z
    RT_list = np.sort(list(RT_mz_I_dict.keys()))
    max_RT=RT_list[len(RT_list)-1]
    min_RT=10    

    sorted_mz_list=[]
    RT_index_array=dict()
    for i in range(0, len(RT_list)):
        RT_index_array[round(RT_list[i], 2)]=i
        sorted_mz_list.append(sorted(RT_mz_I_dict[RT_list[i]].keys()))  

        
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
    print("%g %g"%(max_mz,min_mz))
    rt_search_index=0
    while(RT_list[rt_search_index]<=min_RT):
        if RT_list[rt_search_index]==min_RT:
            break
        rt_search_index=rt_search_index+1 

    total_mz=int(round((max_mz-min_mz+mz_unit)/mz_unit, mz_resolution)) 
    total_RT=len(RT_list)-rt_search_index


#    f=open(datapath+'LC_MS/'+dataname[test_index]+'_pointNet_seg_list_dict_mz5_v3r2_withFP_'+'400', 'rb') 
    f=open(datapath+'LC_MS/'+dataname[test_index]+'_pointNet_seg_list_dict_mz5_v6r1_'+'400', 'rb')
    list_dict, starting_mz_value=pickle.load(f)
    f.close()
    for mz_dict in (600,  800,  1000,  1200,  1400, 1600,1800):
#        f=open(datapath+'LC_MS/'+dataname[test_index]+'_pointNet_seg_list_dict_mz5_v3r2_withFP_'+str(mz_dict), 'rb') #v3r2 
        f=open(datapath+'LC_MS/'+dataname[test_index]+'_pointNet_seg_list_dict_mz5_v6r1_'+str(mz_dict), 'rb') #v3r2
        list_dict_next, starting_mz_value=pickle.load(f)
        f.close()        
        for z in range (1, 10):
            list_dict[z].update(list_dict_next[z])

#    f=open(datapath+'LC_MS/'+dataname[test_index]+'_pointNet_seg_list_dict_test', 'rb') 
#    list_dict, stripe_index = pickle.load(f)
#    f.close()
    gc.collect()
#    list_dict_hold=copy.deepcopy(list_dict)
    isotope_cluster=defaultdict(list)
#    list_dict=copy.deepcopy(list_dict_hold)
    #save_list_dict=copy.deepcopy(list_dict[2]) 
    #list_dict[2]=copy.deepcopy(save_list_dict)
    for z in range (1, 10):
        new_list_dict=dict()
        new_mz_resolution=3
        new_mz_unit=0.001
        list_mz=np.sort(list(list_dict[z].keys()))
        for i in range (0, len(list_mz)):
            if len(list_dict[z][list_mz[i]])==0:
                list_dict[z].pop(list_mz[i])
                continue
            if round(list_mz[i], new_mz_resolution) not in new_list_dict:
                new_list_dict[round(list_mz[i], new_mz_resolution)]=[] 
      
            while len(list_dict[z][list_mz[i]])>0:
                item_1=list_dict[z][list_mz[i]].popleft()
                item_2=list_dict[z][list_mz[i]].popleft()
                if item_2==-1:
                    new_list_dict[round(list_mz[i], new_mz_resolution)].append([item_1])
                else:
                    new_list_dict[round(list_mz[i], new_mz_resolution)].append([item_1, item_2])    
                    list_dict[z][list_mz[i]].popleft() #-1
            list_dict[z].pop(list_mz[i])
        # new_list_dict[mz] has a RT list as: [[10.0],[10.05],[10.10,10.40],...] but not sorted
        # prepare a dict to find longest pairs of ranges 
        list_mz=np.sort(list(new_list_dict.keys()))
        
        for i in range (0, len(list_mz)):
            temp_list=new_list_dict[list_mz[i]]
            temp_list=sorted(temp_list) # [[10],[10.5],[15.01,15.08],[15.01,15.10],[15.02,15.08]]
            RT_pairs=dict()
            for j in range (0, len(temp_list)):
                inside_list=temp_list[j]
                if inside_list[0] not in RT_pairs:
                    RT_pairs[inside_list[0]]=[]
                
                if len(inside_list)==2:
                    RT_pairs[inside_list[0]]=[inside_list[1]]
                    
            # RT_pairs[10]=[]
            # RT_pairs[10.5]=[]
            # RT_pairs[15.01]=[15.08, 15.10]
            # RT_pairs[15.02]=[15.08]
            # ...
            # insert them into new lict_dict
            RT_queue=deque()
            RT_dict_keys=sorted(RT_pairs.keys())
            for k in RT_dict_keys:
                rt_list=RT_pairs[k]
                
                if len(rt_list)>0:
                    RT_queue.append([k, max(rt_list)])
                else:
                    RT_queue.append([k])
            
            # RT_queue=deque([10],[10.5], [15.01,15.10], [15.02,15.08]) # this is sorted
            # now check if succesive [k, max(rt_list)] are mergable or not
            non_overlapping_RT_list=[]
            rt_item=RT_queue.pop() #right
            while len(RT_queue)>0:
                rt_item_pred=RT_queue.pop() #right
                a=min(rt_item_pred)
                b=max(rt_item_pred)
                c=min(rt_item)
                d=max(rt_item)
                if a<=d and b>=c: #overlap
                #if they are mergable
                    min_rt_item=min(min(rt_item), min(rt_item_pred))
                    max_rt_item=min(max(rt_item), max(rt_item_pred))
                    if min_rt_item==max_rt_item:
                        rt_item=[min_rt_item]
                    else:
                        rt_item=[min_rt_item, max_rt_item]
                # if not mergable
                else:
                    non_overlapping_RT_list.append(rt_item)    
                    rt_item=rt_item_pred


            non_overlapping_RT_list.append(rt_item)    
            # now insert them as before
            non_overlapping_RT_list=sorted(non_overlapping_RT_list)
            list_dict[z][list_mz[i]]=deque()        
            for items in non_overlapping_RT_list:
                if len(items)==1:
                    list_dict[z][list_mz[i]].append(items[0])
                    list_dict[z][list_mz[i]].append(-1)
                else:
                    list_dict[z][list_mz[i]].append(items[0])
                    list_dict[z][list_mz[i]].append(items[1])
                    list_dict[z][list_mz[i]].append(-1)
                
            
    # process the list_dict to merge those -1 which results from skiping one index in RT list only
        list_mz=np.sort(list(list_dict[z].keys()))
        max_dict=len(list_mz)
        for i in range (0, max_dict):#
            mz=round(list_mz[i], new_mz_resolution)
            list_RT_range=list_dict[z][mz] # get list of RT range 
            if len(list_RT_range)==0: #remove the empty list
                list_dict[z].pop(mz)
                continue
            list_dict[z][mz] = deque()
            limit=len(list_RT_range)
            seq_running=0
            rt_pred=list_RT_range.popleft()
            j=1
            while j < limit-1:
                rt_current=list_RT_range.popleft()
                j=j+1
                if rt_current==-1:
                    # check if there is just one RT index missing
                    rt_next=list_RT_range.popleft()
                    j=j+1
                    if RT_index_array[rt_next]-RT_index_array[rt_pred]<=5:  #A # those which are ==1 are considered consecutive and merged during formation of list_dict #changed from 1
                        if seq_running==0:
                            list_dict[z][mz].append(rt_pred)
                            rt_pred=rt_next
                            list_dict[z][mz].append(rt_pred)
                            seq_running=1
                        else:        
                            list_dict[z][mz].pop()
                            rt_pred=rt_next
                            list_dict[z][mz].append(rt_pred)
                    elif seq_running==1:
                        seq_running=0
                        list_dict[z][mz].append(-1)
                        rt_pred=rt_next
                    else:
                        rt_pred=rt_next
                        
                elif seq_running==0:
                    list_dict[z][mz].append(rt_pred)
                    rt_pred=rt_current
                    list_dict[z][mz].append(rt_pred)
                    seq_running=1
                    
                elif seq_running==1:
                    list_dict[z][mz].pop()
                    rt_pred=rt_current
                    list_dict[z][mz].append(rt_pred)
                    
            if seq_running==1:
                list_dict[z][mz].append(-1)
            


        # remove the false detections caused by saying YES ahead of time
        # remove those traces whose extent is only one consecutive scans
        #to enclose the ranges in a [start,end,-1] format
        count=0
        list_keys=np.sort(list(list_dict[z].keys()))
        max_dict=len(list_keys)
        for i in range (0, max_dict):#
            mz=round(list_keys[i], new_mz_resolution)
            list_RT_range=list_dict[z][mz] # get list of RT range 
            if len(list_RT_range)==0:
                list_dict[z].pop(mz)
                continue
            
            list_dict[z][mz] = deque()
            limit=len(list_RT_range)
            j=0
            while j < limit:
                rt_st=round(list_RT_range.popleft(), 2)
                rt_end=round(list_RT_range.popleft() , 2)       
                list_RT_range.popleft() #remove the -1 sign
                
                y=0
                for RT_idx in range (RT_index_array[rt_st], RT_index_array[rt_end]+1):
                    intensity=0
                    #################
                    mz_value=round(mz-pow(0.1, new_mz_resolution)+pow(0.1, new_mz_resolution+1)*5, mz_resolution)
                    mz_end=round(mz+pow(0.1, mz_resolution)*4, mz_resolution)
                    find_mz_idx_start= bisect.bisect_left(sorted_mz_list[RT_idx], mz_value)
                    if len(sorted_mz_list[RT_idx])==find_mz_idx_start or round(sorted_mz_list[RT_idx][find_mz_idx_start], mz_resolution)>mz_end:
                        continue
                    mz_value=round(sorted_mz_list[RT_idx][find_mz_idx_start] , mz_resolution) 
                    datapoint=RT_index[RT_list[RT_idx]][mz_value]
                    intensity=intensity+datapoint[0]
                    
                    next_mz_idx=int(datapoint[1])+1         
                    mz_value= round(sorted_mz_list[RT_idx][next_mz_idx], mz_resolution)
                    
                    while mz_value<=mz_end:
                        datapoint=RT_index[RT_list[RT_idx]][mz_value]
                        intensity=intensity+datapoint[0]                        
                        next_mz_idx=int(datapoint[1])+1         
                        mz_value= round(sorted_mz_list[RT_idx][next_mz_idx], mz_resolution)                    
                    ###################
                    y=y+intensity
                
                # B, changed from 2. 1 = two consecutives are kept, 2 = three consecutives are kept. less than that -- discard 
                if RT_index_array[rt_end]-RT_index_array[rt_st]>=1 and y>0: # and np.amax(ms1[RT_index_array[rt_st]-rt_search_index: RT_index_array[rt_end]-rt_search_index+1, int((mz-min_mz)/mz_unit)])>0: #2=to remove those detections which occurs only two consecutive scans
                    list_dict[z][mz].append([rt_st, rt_end, -1])
                else:
                    count=count+1 #just for debug to see how many traces were false detections like that
                j=j+3
        
#        list_dict_hold_2=copy.deepcopy(list_dict)
        merge_isotopes=dict() #based on id 
        list_keys=np.sort(list(list_dict[z].keys()))
        
        if len(list_keys)==1:
            list_dict[z].pop(round(list_keys[0],  new_mz_resolution))
        list_keys=np.sort(list(list_dict[z].keys())) 
        max_dict=len(list_keys)-1
        i=0
        j=0
        k=0
        for i in range (0, max_dict):
            mz_pred=round(list_keys[i], new_mz_resolution)
            mz=round(list_keys[i+1], new_mz_resolution)
            dynamic_mz_unit=round((mz_pred*10.0)/10**6, new_mz_resolution) # mz_pred is 700.1204, mz is 700.1274 or less than that, then fine to merge them
            if mz<=round(mz_pred+dynamic_mz_unit, new_mz_resolution):
                mz_pred_RT_list=list(list_dict[z][mz_pred])
                list_dict[z][mz_pred]=mz_pred_RT_list #it has made list from dict
                mz_RT_list=list(list_dict[z][mz])
                list_dict[z][mz]=mz_RT_list #it has made list from dict
                k=0
                for j in range (0, len(mz_pred_RT_list)):
                    a=round(mz_pred_RT_list[j][0], 2) #actual floating point RT value
                    b=round(mz_pred_RT_list[j][1], 2) #actual  floating point RT value
                    id=mz_pred_RT_list[j][2]
                    #mz_pred is the actual floating point mz value
                    mz_point1= mz_pred
                    rt_1_s=RT_index_array[a] 
                    rt_1_e=RT_index_array[b] 
                    y=0
                    max_intensity=0 #RT_index[RT_list[rt_1_s]][mz_point1][0]
                    peak_RT_1=RT_list[rt_1_s]
                    for RT_idx in range (rt_1_s, rt_1_e+1):
                        intensity=0
                        ################# 
                        mz_value=round(mz_point1 -pow(0.1, new_mz_resolution)+pow(0.1, mz_resolution)*5, mz_resolution)
                        mz_end=round(mz_point1 +pow(0.1, mz_resolution)*4, mz_resolution)
                        find_mz_idx_start= bisect.bisect_left(sorted_mz_list[RT_idx], mz_value)
                        if len(sorted_mz_list[RT_idx])==find_mz_idx_start or round(sorted_mz_list[RT_idx][find_mz_idx_start], mz_resolution)>mz_end:
                            continue
                        mz_value=round(sorted_mz_list[RT_idx][find_mz_idx_start] , mz_resolution) 
                        datapoint=RT_index[RT_list[RT_idx]][mz_value]
                        intensity=intensity+datapoint[0]
                        
                        next_mz_idx=int(datapoint[1])+1         
                        mz_value= round(sorted_mz_list[RT_idx][next_mz_idx], mz_resolution)
                        
                        while mz_value<=mz_end:
                            datapoint=RT_index[RT_list[RT_idx]][mz_value]
                            intensity=intensity+datapoint[0]                        
                            next_mz_idx=int(datapoint[1])+1         
                            mz_value= round(sorted_mz_list[RT_idx][next_mz_idx], mz_resolution)                    
                        ###################                        
                        if intensity>max_intensity:
                            max_intensity=intensity
                            peak_RT_1=RT_list[RT_idx]
                            
                        y=y+intensity
                    
                    weight_pred_mz=round(y, 2)
                        
                    #find the next overlapped 
                    p=k
                    max_overlapped_area=-1
                    max_overlapped_index=-1
                    while p < len(mz_RT_list):
                        c=round(mz_RT_list[p][0], 2)
                        d=round(mz_RT_list[p][1], 2)
                        #check overlapping: if (RectA.Left < RectB.Right && RectA.Right > RectB.Left..)
                        if c>=b:
                            break
                        elif a<d and b>c: #overlap 
 
                            mz_point2= mz
                            rt_2_s=RT_index_array[c] 
                            rt_2_e=RT_index_array[d] 

                            y=0
                            max_intensity=0 #RT_index[RT_list[rt_2_s]][mz_point2][0]
                            peak_RT_2=RT_list[rt_2_s]                            
                            for RT_idx in range (rt_2_s, rt_2_e+1):
                                intensity=0
                                ################# 
                                mz_value=round(mz_point2 -pow(0.1, new_mz_resolution)+pow(0.1, mz_resolution)*5, mz_resolution)
                                mz_end=round(mz_point2 +pow(0.1, mz_resolution)*4, mz_resolution)
                                find_mz_idx_start= bisect.bisect_left(sorted_mz_list[RT_idx], mz_value)
                                if len(sorted_mz_list[RT_idx])==find_mz_idx_start or round(sorted_mz_list[RT_idx][find_mz_idx_start], mz_resolution)>mz_end:
                                    continue
                                mz_value=round(sorted_mz_list[RT_idx][find_mz_idx_start] , mz_resolution) 
                                datapoint=RT_index[RT_list[RT_idx]][mz_value]
                                intensity=intensity+datapoint[0]
                                
                                next_mz_idx=int(datapoint[1])+1         
                                mz_value= round(sorted_mz_list[RT_idx][next_mz_idx], mz_resolution)
                                
                                while mz_value<=mz_end:
                                    datapoint=RT_index[RT_list[RT_idx]][mz_value]
                                    intensity=intensity+datapoint[0]                        
                                    next_mz_idx=int(datapoint[1])+1         
                                    mz_value= round(sorted_mz_list[RT_idx][next_mz_idx], mz_resolution)                    
                                ###################                       



                                if intensity>max_intensity:
                                    max_intensity=intensity
                                    peak_RT_2=RT_list[RT_idx]
                                    
                                y=y+intensity                            

                            # C
                            if abs(RT_index_array[peak_RT_1]-RT_index_array[peak_RT_2])<=2: #changed from 2
                                overlapped_area=min(b, d)-max(a, c)
                                if overlapped_area>max_overlapped_area:
                                    max_overlapped_area=overlapped_area
                                    max_overlapped_index=p
                        p=p+1
                            
                    if max_overlapped_index==-1: #no match 
                        if id==-1:
                            new_id=len(merge_isotopes)
                            mz_weight=[weight_pred_mz]
                            peak_RT_list=[peak_RT_1]
                            merge_isotopes[new_id]=[mz_weight, a, b, -1, weight_pred_mz, [mz_pred], peak_RT_list]  
                            list_dict[z][mz_pred][j][2]=[]
                            list_dict[z][mz_pred][j][2].append(new_id)                      
    #                        list_dict[z][mz_pred][j]=[0, 0, -1] #-- pop
                        k=p
                        continue
                    # else 
                    c=round(mz_RT_list[max_overlapped_index][0], 2)
                    d=round(mz_RT_list[max_overlapped_index][1], 2)                    
                                                 
                    mz_point2= mz
                    rt_2_s=RT_index_array[c] 
                    rt_2_e=RT_index_array[d] 

                    y=0
                    max_intensity=0 #RT_index[RT_list[rt_2_s]][mz_point2][0]
                    peak_RT_2=RT_list[rt_2_s]                            
                    for RT_idx in range (rt_2_s, rt_2_e+1):
                        intensity=0
                        ################# 
                        mz_value=round(mz_point2 -pow(0.1, new_mz_resolution)+pow(0.1, mz_resolution)*5, mz_resolution)
                        mz_end=round(mz_point2 +pow(0.1, mz_resolution)*4, mz_resolution)
                        find_mz_idx_start= bisect.bisect_left(sorted_mz_list[RT_idx], mz_value)
                        if len(sorted_mz_list[RT_idx])==find_mz_idx_start or round(sorted_mz_list[RT_idx][find_mz_idx_start], mz_resolution)>mz_end:
                            continue
                        mz_value=round(sorted_mz_list[RT_idx][find_mz_idx_start] , mz_resolution) 
                        datapoint=RT_index[RT_list[RT_idx]][mz_value]
                        intensity=intensity+datapoint[0]
                        
                        next_mz_idx=int(datapoint[1])+1         
                        mz_value= round(sorted_mz_list[RT_idx][next_mz_idx], mz_resolution)
                        
                        while mz_value<=mz_end:
                            datapoint=RT_index[RT_list[RT_idx]][mz_value]
                            intensity=intensity+datapoint[0]                        
                            next_mz_idx=int(datapoint[1])+1         
                            mz_value= round(sorted_mz_list[RT_idx][next_mz_idx], mz_resolution)                    
                        ###################                       



                        if intensity>max_intensity:
                            max_intensity=intensity
                            peak_RT_2=RT_list[RT_idx]
                            
                        y=y+intensity  

                    weight_mz=y
                    intensity_2=weight_mz
                    ################################
                                                    
                    if id==-1:
                        intensity_1=weight_pred_mz

                        #########################
                        if intensity_1>intensity_2:
                            grp_rt_st=a
                            grp_rt_end=b
                            auc=intensity_1
                        else:
                            grp_rt_st=c
                            grp_rt_end=d
                            auc=intensity_2
                            
                        new_id=len(merge_isotopes)
                        mz_weight=[weight_pred_mz, weight_mz]
                        peak_RT_list=[peak_RT_1, peak_RT_2]
                        merge_isotopes[new_id]=[mz_weight, grp_rt_st, grp_rt_end, auc, intensity_1+intensity_2, [mz_pred, mz], peak_RT_list]
                        if list_dict[z][mz][max_overlapped_index][2]==-1:
                            list_dict[z][mz][max_overlapped_index][2]=[]
                            
                        list_dict[z][mz][max_overlapped_index][2].append(new_id)
                        list_dict[z][mz_pred][j][2]=[]
                        list_dict[z][mz_pred][j][2].append(new_id)
                    else: #this might need to run a loop over ids. do this for all ids
                        for pred_id in id:
                            get_current_intensity=merge_isotopes[pred_id][3]
                            if get_current_intensity<=intensity_2:
                                merge_isotopes[pred_id][1]=c
                                merge_isotopes[pred_id][2]=d
                                merge_isotopes[pred_id][3]=intensity_2
                                
                            # add new intensity and weight to the existing one
                            merge_isotopes[pred_id][4]=merge_isotopes[pred_id][4]+intensity_2
                            merge_isotopes[pred_id][5].append(mz)
                            merge_isotopes[pred_id][0].append(weight_mz)
                            merge_isotopes[pred_id][6].append(peak_RT_2)
                            if list_dict[z][mz][max_overlapped_index][2]==-1:
                                list_dict[z][mz][max_overlapped_index][2]=[]
                            
                            list_dict[z][mz][max_overlapped_index][2].append(pred_id)
                     
                    if max_overlapped_index==-1:
                        k=p
                    else:
                        k=max_overlapped_index
    #            if id==0: #for debug
    #                break
            elif i==0 or mz_pred>round(list_keys[i-1]+dynamic_mz_unit, new_mz_resolution):
#                list_dict[z].pop(mz_pred)
                mz_pred_RT_list=list(list_dict[z][mz_pred])
                for j in range (0, len(mz_pred_RT_list)):
                    a=round(mz_pred_RT_list[j][0], 2) #actual floating point RT value
                    b=round(mz_pred_RT_list[j][1], 2) #actual  floating point RT value
                    id=mz_pred_RT_list[j][2]
                    #mz_pred is the actual floating point mz value
                    mz_point1= mz_pred
                    rt_1_s=RT_index_array[a] 
                    rt_1_e=RT_index_array[b] 
                    y=0
                    max_intensity=0 #RT_index[RT_list[rt_1_s]][mz_point1][0]
                    peak_RT_1=RT_list[rt_1_s]
                    for RT_idx in range (rt_1_s, rt_1_e+1):
                        intensity=0
                        ################# 
                        mz_value=round(mz_point1 -pow(0.1, new_mz_resolution)+pow(0.1, mz_resolution)*5, mz_resolution)
                        mz_end=round(mz_point1 +pow(0.1, mz_resolution)*4, mz_resolution)
                        find_mz_idx_start= bisect.bisect_left(sorted_mz_list[RT_idx], mz_value)
                        if len(sorted_mz_list[RT_idx])==find_mz_idx_start or round(sorted_mz_list[RT_idx][find_mz_idx_start], mz_resolution)>mz_end:
                            continue
                        mz_value=round(sorted_mz_list[RT_idx][find_mz_idx_start] , mz_resolution) 
                        datapoint=RT_index[RT_list[RT_idx]][mz_value]
                        intensity=intensity+datapoint[0]
                        next_mz_idx=int(datapoint[1])+1         
                        mz_value= round(sorted_mz_list[RT_idx][next_mz_idx], mz_resolution)

                        while mz_value<=mz_end:
                            datapoint=RT_index[RT_list[RT_idx]][mz_value]
                            intensity=intensity+datapoint[0]
                            next_mz_idx=int(datapoint[1])+1         
                            mz_value= round(sorted_mz_list[RT_idx][next_mz_idx], mz_resolution)
                        ###################
                        if intensity>max_intensity:
                            max_intensity=intensity
                            peak_RT_1=RT_list[RT_idx]
                        y=y+intensity

                    weight_pred_mz=round(y, 2)
                    new_id=len(merge_isotopes)
                    mz_weight=[weight_pred_mz]
                    peak_RT_list=[peak_RT_1]
                    merge_isotopes[new_id]=[mz_weight, a, b, -1, weight_pred_mz, [mz_pred], peak_RT_list]  
                    list_dict[z][mz_pred][j][2]=[]
                    list_dict[z][mz_pred][j][2].append(new_id)



        if len(list_keys)!=0:
            i=i+1                        
            mz=round(list_keys[i], new_mz_resolution)
            mz_RT_list=list(list_dict[z][mz])
            list_dict[z][mz]=mz_RT_list
            for j in range (0, len(mz_RT_list)):
                if mz_RT_list[j][2]==-1:
                    a=round(mz_RT_list[j][0], 2)
                    b=round(mz_RT_list[j][1], 2)     

                    mz_point1= mz
                    rt_1_s=RT_index_array[a] 
                    rt_1_e=RT_index_array[b] 
                    y=0
                    max_intensity=0 #RT_index[RT_list[rt_1_s]][mz_point1][0]
                    peak_RT_1=RT_list[rt_1_s]
                    for RT_idx in range (rt_1_s, rt_1_e+1):
                        intensity=0
                        ################# 
                        mz_value=round(mz_point1 -pow(0.1, new_mz_resolution)+pow(0.1, mz_resolution)*5, mz_resolution)
                        mz_end=round(mz_point1 +pow(0.1, mz_resolution)*4, mz_resolution)
                        find_mz_idx_start= bisect.bisect_left(sorted_mz_list[RT_idx], mz_value)
                        if len(sorted_mz_list[RT_idx])==find_mz_idx_start or round(sorted_mz_list[RT_idx][find_mz_idx_start], mz_resolution)>mz_end:
                            continue
                        mz_value=round(sorted_mz_list[RT_idx][find_mz_idx_start] , mz_resolution) 
                        datapoint=RT_index[RT_list[RT_idx]][mz_value]
                        intensity=intensity+datapoint[0]
                        
                        next_mz_idx=int(datapoint[1])+1         
                        mz_value= round(sorted_mz_list[RT_idx][next_mz_idx], mz_resolution)
                        
                        while mz_value<=mz_end:
                            datapoint=RT_index[RT_list[RT_idx]][mz_value]
                            intensity=intensity+datapoint[0]                        
                            next_mz_idx=int(datapoint[1])+1         
                            mz_value= round(sorted_mz_list[RT_idx][next_mz_idx], mz_resolution)                    
                        ###################                       



                        if intensity>max_intensity:
                            max_intensity=intensity
                            peak_RT_1=RT_list[RT_idx]
                            
                        y=y+intensity
                    
                    weight_mz=y
                    
                    new_id=len(merge_isotopes)
                    mz_weight=[weight_mz]
                    peak_RT_list=[peak_RT_1]
                    merge_isotopes[new_id]=[mz_weight, a, b, -1, weight_mz, [mz], peak_RT_list]  
                    list_dict[z][mz][j][2]=[]
                    list_dict[z][mz][j][2].append(new_id)                                      
    #                list_dict[z][mz][j]=[0, 0, -1]

        print('merge isotopes done')

        isotope_table=defaultdict(list)
        for i in range (0, len(merge_isotopes)):
            mz_weight_list=merge_isotopes[i][0]
            max_weight=-1
            mz_index=-1
            for j in range(0, len(mz_weight_list)):
                if mz_weight_list[j]>=max_weight:
                    max_weight=mz_weight_list[j]
                    mz_index=j
            isotope_table[round(merge_isotopes[i][5][mz_index], new_mz_resolution)].append([merge_isotopes[i][6][mz_index], merge_isotopes[i][1],merge_isotopes[i][2],merge_isotopes[i][4],merge_isotopes[i][5]])

        isotope_mz_list=sorted(isotope_table.keys())

        isotope_table_temp=defaultdict(list)
        for i in isotope_mz_list:
            isotope_table[i]=sorted(isotope_table[i])
            j=0
            while (j<len(isotope_table[i])):
                isotope_table_temp[i].append(isotope_table[i][j])
                if j+1>=len(isotope_table[i]):
                    break
                for k in range (j+1,  len(isotope_table[i])):
                    if (isotope_table[i][j][0]!=isotope_table[i][k][0]):
                        break
                j=k
                
                
        isotope_table=copy.deepcopy(isotope_table_temp)
        isotope_table_temp=0

    #    #AUC
    #    print('AUC calculation of isotopes')
    #    isotope_mz_list=sorted(isotope_table.keys())
    #    for mz in isotope_mz_list:
    #        iso_list=isotope_table[mz]
    #        for i in range (0, len(iso_list)):
    #            mz_point=int(round((mz-min_mz)/mz_unit))
    #            rt_s=RT_index_array[iso_list[i][1]]-rt_search_index 
    #            rt_e=RT_index_array[iso_list[i][2]]-rt_search_index 
    #            y=np.array((np.copy(ms1[rt_s:rt_e+1, mz_point])/255)*maxI)
    #            x=np.array(RT_list[RT_index_array[iso_list[i][1]]:RT_index_array[iso_list[i][2]]+1])
    #            AUC_iso=metrics.auc(x, y)
    #            isotope_table[mz][i][3]=AUC_iso



        # form cluster of isotopes to feed into the isotope grouping module
        DEBUG=0
        mz_list=sorted(isotope_table.keys())
        tolerance_RT=2 #D
    #    mz_tolerance=2
        for mz in mz_list:
            iso_list_mz=isotope_table[mz]
            for i in range (0, len(iso_list_mz)):
                current_iso=iso_list_mz[i]
                current_mz=mz
                if current_iso[0]==-1:
                    continue
                current_peak=current_iso[0]
                found1=0
                id=len(isotope_cluster)
                next_mz_exact=round(current_mz+isotope_gap[z], new_mz_resolution)
                next_mz_range=[]
                next_mz_range.append(next_mz_exact)
                mz_tolerance_10ppm=round((next_mz_exact*10.0)/10**6, new_mz_resolution)
                mz_left_limit=round(next_mz_exact-mz_tolerance_10ppm, new_mz_resolution)
                mz_right_limit=round(next_mz_exact+mz_tolerance_10ppm, new_mz_resolution)
#                mz_tolerance=int(round(mz_tolerance_10ppm/new_mz_unit, new_mz_resolution))

                find_mz_idx_start= bisect.bisect_left(mz_list, mz_left_limit)
                while len(mz_list)!=find_mz_idx_start and round(mz_list[find_mz_idx_start], new_mz_resolution)<=mz_right_limit:     
                    next_mz_range.append(round(mz_list[find_mz_idx_start], new_mz_resolution))
                    find_mz_idx_start=find_mz_idx_start+1
                # next_mz might be a range
                k=0
                while(k<len(next_mz_range)):
                    next_mz= next_mz_range[k]            
                    if next_mz in isotope_table:
                        found2=0
                        iso_list_next_mz=isotope_table[next_mz]
                        
                        for j in range (0, len(iso_list_next_mz)):
                            next_iso=iso_list_next_mz[j]
                            if next_iso[0]==-1:
                                continue
                            if RT_index_array[next_iso[0]]>RT_index_array[current_peak]+tolerance_RT:
                                break
                            
                            if RT_index_array[current_peak]-tolerance_RT<=RT_index_array[next_iso[0]] and RT_index_array[next_iso[0]]<=RT_index_array[current_peak]+tolerance_RT: # and current_iso[3] >= ((next_iso[3]*3)/4) :
                               # within tolerance. Check RT range
                                a=current_iso[1]
                                b=current_iso[2]
                                c=next_iso[1]
                                d=next_iso[2]
                                if a<=d and b>=c: #overlapped
                                    found2=1
                                    break
                                
                        if found2==1:
                            found1=1
                            isotope_table[next_mz][j]=[-1] #remove it
                            # add pred_iso to cluster
                            isotope_cluster[id].append([current_mz, current_iso])
                            current_iso=next_iso
                            current_peak=current_iso[0]
                            current_mz=next_mz
                            ############
                            next_mz_exact=round(current_mz+isotope_gap[z], new_mz_resolution)
                            next_mz_range=[]
                            next_mz_range.append(next_mz_exact)
                            mz_tolerance_10ppm=round((next_mz_exact*10.0)/10**6, new_mz_resolution)
                            mz_left_limit=round(next_mz_exact-mz_tolerance_10ppm, new_mz_resolution)
                            mz_right_limit=round(next_mz_exact+mz_tolerance_10ppm, new_mz_resolution)
            #                mz_tolerance=int(round(mz_tolerance_10ppm/new_mz_unit, new_mz_resolution))
                            find_mz_idx_start= bisect.bisect_left(mz_list, mz_left_limit)
                            while len(mz_list)!=find_mz_idx_start and round(mz_list[find_mz_idx_start], new_mz_resolution)<=mz_right_limit:     
                                next_mz_range.append(round(mz_list[find_mz_idx_start], new_mz_resolution))
                                find_mz_idx_start=find_mz_idx_start+1

                            ############
                            k=0
                        else:    
                            k=k+1
                    else:
                        k=k+1
                if found1==1:
                    # add pred_iso to cluster
                    isotope_cluster[id].append([current_mz, current_iso])
                    isotope_cluster[id].append([z]) # charge
                else: #else: insert them in to the single iso table
    #                id=len(isotope_cluster)
                    isotope_cluster[id].append([current_mz, current_iso])    
                    isotope_cluster[id].append([z])
                    
                isotope_table[mz][i]=[-1] #remove it
    #        if DEBUG==1:
    #            break


    #########################################
    print(len(isotope_cluster.keys()))
    total_cluster=len(isotope_cluster.keys())
    temp_isotope_cluster=copy.deepcopy(isotope_cluster)
    isotope_cluster=defaultdict(list)
    total_clusters=len(temp_isotope_cluster.keys())

    for i in range (0, total_clusters):
        ftr=copy.deepcopy(temp_isotope_cluster[i])
#        isotope_cluster[round(ftr[0][0], new_mz_resolution)].append(ftr) # starting m/z of the 1st isotope
        isotope_cluster[round(ftr[0][0], 2)].append(ftr) # starting m/z of the 1st isotope
        
    temp_isotope_cluster=0

    temp_isotope_cluster=copy.deepcopy(isotope_cluster)
    isotope_cluster=defaultdict(list)
    keys_list=sorted(temp_isotope_cluster.keys())
    max_num_iso=0
    total_cluster=0
    for mz in keys_list:
        ftr_list=temp_isotope_cluster[mz]
        i=0
        while i<len(ftr_list):
            ftr=copy.deepcopy(ftr_list[i])
            j=i+1
            while j<len(ftr_list): 
                ftr_2=copy.deepcopy(ftr_list[j])
                if ftr_2[0][0]==ftr[0][0] and len(ftr)==len(ftr_2) and ftr[0][1][0]==ftr_2[0][1][0]:
                    j=j+1
                else:
                    break
            isotope_cluster[round(ftr[0][0], 2)].append(ftr)
            total_cluster=total_cluster+1
            i=j
            if (len(ftr)-1)>max_num_iso:
                max_num_iso=(len(ftr)-1)        
        
    print('%d %d'%(max_num_iso, total_cluster)) 
    temp_isotope_cluster=0
    

        
    print(max_num_iso) 
#    f=open(datapath+'LC_MS/'+dataname[test_index]+'_pointNet_centroid_v2r1_clusters_mz3_v5', 'wb') 
#    f=open(datapath+'LC_MS/'+dataname[test_index]+'_pointNet_v3r2_withFP_clusters_mz3_v5', 'wb') 
    f=open(datapath+'LC_MS/'+dataname[test_index]+'_pointNet_v6r1_clusters_mz3_v5', 'wb')
    pickle.dump([isotope_cluster, max_num_iso, total_cluster], f, protocol=2)
    f.close()
    print('cluster write done')






