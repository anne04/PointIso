# nohup python -u  k0_separation_timsTOF_parallel.py [recordpath] [sample_name] [segment] [scanpath] > output.log &
''' nohup python -u k0_separation_timsTOF_parallel.py '/data/anne/timsTOF/hash_records/' 'A1_1_2042' 1 /data/anne/timsTOF/scanned_result/ > output.log & '''
from __future__ import division
from __future__ import print_function
from time import time
import pickle
import numpy as np
from collections import deque
from collections import defaultdict
import copy
#import scipy.misc
#import scipy.stats as stats
import sys
#from sklearn import metrics
import bisect
#import gc
import gzip
from operator import itemgetter

recordpath=sys.argv[1]
sample_name=sys.argv[2]
gpu_index=sys.argv[3]
segment=sys.argv[4]
scanpath=sys.argv[5]

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



target_part=int(segment) #8 # 
max_part=12
min_part=1

RT_window=10
mz_resolution=5 #
k0_resolution=5
total_class=10
RT_unit=0.01
mz_unit=0.00001
num_class=10
new_mz_resolution=3
new_mz_unit=0.001


print('trying to load ms1 record')
######################################################################
f=open(recordpath+'pointCloud_'+sample_name+'_maxI_k0', 'rb')
maxI=pickle.load(f)
f.close()


f=gzip.open(recordpath+'pointCloud_'+sample_name+'_k0_dict', 'rb')
k0_dict=pickle.load(f)
f.close()

k0_list=sorted(k0_dict.keys())
k0_index_array=dict()
for i in range (0, len(k0_list)):
    k0_index_array[k0_list[i]]=i



part=target_part
f=gzip.open(scanpath+sample_name+'_timsTOF_list_dict_RT_'+str(part), 'rb') #v3r2
list_dict_RT, part_dict=pickle.load(f)
f.close() 

max_RT=[]
min_RT=[]
max_RT.append(-1)
min_RT.append(100)
for z in range (1, num_class):
    RT_list=sorted(list_dict_RT[z].keys())

    if len(RT_list)>0:
        max_RT.append(RT_list[len(RT_list)-1])
        min_RT.append(RT_list[0])
    else:
        max_RT.append(-1)
        min_RT.append(100)

for i in range (1, 4):
    if part+i<=max_part:
        f=gzip.open(scanpath+sample_name+'_timsTOF_list_dict_RT_'+str(part+i), 'rb') #v3r2
        list_dict_next, part_dict=pickle.load(f)
        f.close()
        for z in range (1, 10):
            RT_list=sorted(list_dict_next[z].keys(), reverse=False)
            for rt_idx in range (0, len(RT_list)): #min(, RT_window*2)):
                print('rt indx %d'%rt_idx)
                rt=np.float32(RT_list[rt_idx])
                if rt in list_dict_RT[z]:
                    for mz in list_dict_next[z][rt].keys():
                        list_dict_RT[z][rt][mz].extend(list_dict_next[z][rt][mz])
                elif len(list_dict_next[z][rt].keys())>0:
                    list_dict_RT[z][rt]=defaultdict(list)                
                    for mz in list_dict_next[z][rt].keys():
                        list_dict_RT[z][rt][mz].extend(list_dict_next[z][rt][mz])
    else:
        break



print('data restore done')

RT_list=[]
RT_index_array=[]
sorted_mz_list=[]
RT_index_array.append(-1)
sorted_mz_list.append(-1)
RT_list.append(-1)

for z in range (1, num_class):
    RT_list.append([])
    RT_list[z]=sorted(list_dict_RT[z].keys())

    RT_index_array.append(dict())
    sorted_mz_list.append([])
    sorted_mz_list[z]=[]
    for i in range (0, len(RT_list[z])):
        RT_value=np.float32(RT_list[z][i])
        RT_index_array[z][RT_value]=i
        sorted_mz_list[z].append([])
        sorted_mz_list[z][i]=sorted(list_dict_RT[z][RT_value].keys())  



start_time=time()
f=gzip.open(scanpath+sample_name+'_featureTable_v2_timsTOF','rb')
feature_table=pickle.load(f)
f.close()

isotope_cluster=defaultdict(list)
mz_list=feature_table.keys()
total_feature=0
for mz in mz_list:
    ftr_list=feature_table[mz]
    for ftr in ftr_list:
        # if both points are greater than min point + starting is less than  max point0
        if (ftr[0][1][1] >= min(min_RT) and ftr[0][1][1] <= max(max_RT)) and (ftr[0][1][2] >= min(min_RT) and  ftr[0][1][2] <= max(max_RT)):
            long_mz=ftr[0][1][5]
            isotope_cluster[round(long_mz, mz_resolution)].append(ftr)
            total_feature=total_feature+1

feature_table=isotope_cluster

print('%d'%total_feature)



ppm=10
k0_tolerance=0.01 #
A=5 # #increasing A will deduct matching with smaller k0 tolerance

E=1 #gap (along k0 axis) between starting and ending of two features
k0_skip_tolerance=0.01
total_feature=0
not_found=0
total_feature_input=0




########## pick each feature and make a list_dict 
mz_list_ftr=sorted(feature_table.keys())
new_feature_table=defaultdict(list)
for mz_key_indx in range (0, len(mz_list_ftr)):
    mz_key=mz_list_ftr[mz_key_indx]
    ftr_list=feature_table[mz_key] #monoisotope
    for f in range (0, len(ftr_list)):
        total_feature_input=total_feature_input+1
        ftr=ftr_list[f]
        min_RT=1000
        max_RT=-1
        for iso_index in range (0, 1): #len(ftr)-1):
            if ftr[iso_index][1][1]<min_RT:
                min_RT=ftr[iso_index][1][1]
            if ftr[iso_index][1][2]>max_RT:
                max_RT=ftr[iso_index][1][2]

        min_RT=np.float32(min_RT)
        max_RT=np.float32(max_RT)
        if max_RT-min_RT>2: 
            continue
        mz_tolerance=round((ftr[0][0]*ppm)/10**6, new_mz_resolution)
        min_mz=round(ftr[0][0]-mz_tolerance, new_mz_resolution) # resolution 3
        z=int(ftr[len(ftr)-1][0])
        max_mz=round(ftr[len(ftr)-2][0]+mz_tolerance, new_mz_resolution) #round(max(min_mz+isotope_gap[z]+mz_tolerance, ftr[len(ftr)-2][0]+mz_tolerance), new_mz_resolution)

        ##################################################
        # a ftr with min_mz,max_mz,minRT,maxRT is found
        # make a list_dict of, key=mz, value=1/k0, starting from minRT to max RT
        cluster_for_rt=[]
        RT_s=RT_index_array[z][min_RT]
        RT_e=RT_index_array[z][max_RT]
#        i=-1
        for RT_idx in range (RT_s,  RT_e+1):
#            i=i+1
            list_dict=dict()
            mz_list=sorted_mz_list[z][RT_idx] #sorted(list_dict_RT[z][RT_list[z][RT_idx]].keys())
            find_mz_idx_start= bisect.bisect_left(mz_list, min_mz)
            if len(mz_list)==find_mz_idx_start or round(mz_list[find_mz_idx_start], mz_resolution)>max_mz:
                continue
            mz_value_long=mz_list[find_mz_idx_start]  
#                print(mz_value_long)
            while mz_value_long<=max_mz and find_mz_idx_start<len(mz_list):
                mz_value=np.round(mz_value_long, new_mz_resolution)
                # as long as they all produce the same mz_value, insert them into the same k0_list
                k0_list=deque()
                while np.round(mz_value_long, new_mz_resolution)==mz_value:
#                        print(mz_value)
#                        k0_list.append(list_dict_RT[RT_list[RT_idx]][mz_value_long])
                    temp_list=sorted(list_dict_RT[z][RT_list[z][RT_idx]][mz_value_long])
                    for k in range (0, len(temp_list)): #[k0,intensity]
                        real_intensity=(temp_list[k][1]/255)*maxI
                        k0_list.append([temp_list[k][0], real_intensity])

                    find_mz_idx_start=find_mz_idx_start+1
                    if find_mz_idx_start<len(mz_list):
                        mz_value_long=mz_list[find_mz_idx_start]
                    else:
                        break

                # merge intesity of those duplicates
                k0_list=sorted(k0_list)
                temp_list=[]
                j=0
                intensity_sum=k0_list[j][1]
                j=1
                while j < len(k0_list):
                    if k0_list[j][0]==k0_list[j-1][0]:
                        intesity_sum=intensity_sum+k0_list[j][1]
                    else:
                        temp_list.append([k0_list[j-1][0], intensity_sum])
                        intensity_sum=k0_list[j][1]

                    j=j+1

                temp_list.append([k0_list[j-1][0], intensity_sum])    
                k0_list=temp_list    
                # add continuas k0 values in to the queue separated by -1
                list_dict[mz_value] = deque()
                pred_start=k0_list[0][0]
                pred_k0=k0_list[0][0]
                current_intensity=k0_list[0][1]
#                list_dict[i][mz_value].append(k0_list[0])
                for j in range (1, len(k0_list)):
#                        if np.abs(round(k0_list[j][0], k0_resolution)-round(pred_k0, k0_resolution))<=k0_resolution: 
                    if np.abs(k0_index_array[np.float32(round(k0_list[j][0], k0_resolution))]-k0_index_array[np.float32(round(pred_k0, k0_resolution))])<=A: #A: continuation
                        pred_k0=k0_list[j][0]
                        current_intensity=current_intensity+k0_list[j][1]
                    else: # when they are far apart
#                            if pred_start!=pred_k0:
                        list_dict[mz_value].append([[pred_start, pred_k0], current_intensity, -1])
#                            else:
#                                list_dict[mz_value].append([[pred_start], current_intensity, -1])

                        pred_start=k0_list[j][0]
                        pred_k0=k0_list[j][0]
                        current_intensity=k0_list[j][1]

#                    if pred_start!=pred_k0:
                list_dict[mz_value].append([[pred_start, pred_k0], current_intensity, -1])                        
#                    else:
                    #list_dict[mz_value].append([[pred_start], current_intensity, -1])

            ##################
            # now you have list_dict in desired format: for each mz, [k0_start,k0_end,intesity]
            # now make a cluster list for this RT
            ## and run following script ############
            isotope_cluster=defaultdict(deque)
            merge_isotopes=dict() #based on id 
            list_keys=sorted(list_dict.keys())
#                if len(list_keys)==1:
#                    list_dict.pop(round(list_keys[0],  new_mz_resolution))
#                    
#                list_keys=sorted(list_dict.keys()) 
            max_dict=len(list_keys)-1
            i=0
            j=0
            k=0
            for i in range (0, max_dict):
    #            print(i)
                mz_pred=round(list_keys[i], new_mz_resolution)
                mz=round(list_keys[i+1], new_mz_resolution)
                dynamic_mz_unit=round((mz_pred*ppm)/10**6, new_mz_resolution) # mz_pred is 700.1204, mz is 700.1274 or less than that, then fine to merge them
                if mz<=round(mz_pred+dynamic_mz_unit, new_mz_resolution):
                    mz_pred_RT_list=list_dict[mz_pred]
                    mz_RT_list=list_dict[mz]
                    k=0
                    for j in range (0, len(mz_pred_RT_list)):
                        a=round(min(mz_pred_RT_list[j][0]), k0_resolution) #actual floating point RT value
                        b=round(max(mz_pred_RT_list[j][0]), k0_resolution) #actual  floating point RT value
                        id=mz_pred_RT_list[j][2]
                        #mz_pred is the actual floating point mz value
                        mz_point1= mz_pred
                        y=mz_pred_RT_list[j][1]
                        weight_pred_mz=round(y, 2)

                        #find the next overlapped 
                        p=k
                        max_overlapped_area=-1
                        max_overlapped_index=-1
                        while p < len(mz_RT_list):
                            c=round(min(mz_RT_list[p][0]), k0_resolution)
                            d=round(max(mz_RT_list[p][0]), k0_resolution)
                            #check overlapping: if (RectA.Left < RectB.Right && RectA.Right > RectB.Left..)
                            if np.abs(c-b)>k0_tolerance:
                                break
                            elif (a<=d and b>=c) or np.abs(round(((a+b)/2)-((c+d)/2), k0_resolution))<=k0_tolerance: #(np.abs(a-d)<=k0_tolerance and np.abs(b-c)<=k0_tolerance): #overlap 
                                mz_point2= mz
#                                    rt_2_s=RT_index_array[z][c] 
#                                    rt_2_e=RT_index_array[z][d] 
                                y=mz_RT_list[p][1]
                                # C
    #                            if abs(RT_index_array[np.float32(peak_RT_1)]-RT_index_array[np.float32(peak_RT_2)])<=2: #changed from 2
                                overlapped_area=min(b, d)-max(a, c)
                                if overlapped_area>max_overlapped_area:
                                    max_overlapped_area=overlapped_area
                                    max_overlapped_index=p
                            p=p+1

                        if max_overlapped_index==-1: #no match 
                            if id==-1:
                                new_id=len(merge_isotopes)
                                mz_weight=[weight_pred_mz]
    #                            peak_RT_list=[peak_RT_1]
                                merge_isotopes[new_id]=[mz_weight, a, b, -1, weight_pred_mz, [mz_pred]] #, peak_RT_list]  
                                list_dict[mz_pred][j][2]=[]
                                list_dict[mz_pred][j][2].append(new_id)                      
        #                        list_dict[mz_pred][j]=[0, 0, -1] #-- pop
                            k=p
                            continue
                        # else 
                        c=round(min(mz_RT_list[max_overlapped_index][0]),  k0_resolution)
                        d=round(max(mz_RT_list[max_overlapped_index][0]),  k0_resolution)                    

                        mz_point2= mz
#                            rt_2_s=RT_index_array[z][c] 
#                            rt_2_e=RT_index_array[z][d] 

                        y=mz_RT_list[max_overlapped_index][1] ####START HERE ##########
    #                    peak_RT_2=RT_list[rt_2_s]                            
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
    #                        peak_RT_list=[peak_RT_1, peak_RT_2]
                            merge_isotopes[new_id]=[mz_weight, grp_rt_st, grp_rt_end, auc, intensity_1+intensity_2, [mz_pred, mz]] #, peak_RT_list]
                            if list_dict[mz][max_overlapped_index][2]==-1:
                                list_dict[mz][max_overlapped_index][2]=[]

                            list_dict[mz][max_overlapped_index][2].append(new_id)
                            list_dict[mz_pred][j][2]=[]
                            list_dict[mz_pred][j][2].append(new_id)
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
    #                            merge_isotopes[pred_id][6].append(peak_RT_2)
                                if list_dict[mz][max_overlapped_index][2]==-1:
                                    list_dict[mz][max_overlapped_index][2]=[]

                                list_dict[mz][max_overlapped_index][2].append(pred_id)

                        if max_overlapped_index==-1:
                            k=p
                        else:
                            k=max_overlapped_index
        #            if id==0: #for debug
        #                break
                elif i==0 or mz_pred>round(list_keys[i-1]+dynamic_mz_unit, new_mz_resolution):
    #                list_dict.pop(mz_pred)

                    mz_pred_RT_list=list(list_dict[mz_pred])
                    for j in range (0, len(mz_pred_RT_list)):
                        a=round(min(mz_pred_RT_list[j][0]),  k0_resolution) #actual floating point RT value
                        b=round(max(mz_pred_RT_list[j][0]),  k0_resolution) #actual  floating point RT value
                        id=mz_pred_RT_list[j][2]
                        #mz_pred is the actual floating point mz value
                        mz_point1= mz_pred
#                            rt_1_s=RT_index_array[z][a] 
#                            rt_1_e=RT_index_array[z][b] 
                        y=mz_pred_RT_list[j][1]
    #                    peak_RT_1=RT_list[rt_1_s]
                        weight_pred_mz=round(y, 2)
                        new_id=len(merge_isotopes)
                        mz_weight=[weight_pred_mz]
    #                    peak_RT_list=[peak_RT_1]
                        merge_isotopes[new_id]=[mz_weight, a, b, -1, weight_pred_mz, [mz_pred]] #, peak_RT_list]  
                        list_dict[mz_pred][j][2]=[]
                        list_dict[mz_pred][j][2].append(new_id)



            if len(list_keys)!=0:
                i=i+1  
                if max_dict==0:
                    i=0
                mz=round(list_keys[i], new_mz_resolution)
                mz_RT_list=list(list_dict[mz])
                list_dict[mz]=mz_RT_list
                for j in range (0, len(mz_RT_list)):
                    if mz_RT_list[j][2]==-1:
                        a=round(min(mz_RT_list[j][0]), k0_resolution)
                        b=round(max(mz_RT_list[j][0]), k0_resolution)     
                        mz_point1= mz
    #                    rt_1_s=RT_index_array[np.float32(a)] 
    #                    rt_1_e=RT_index_array[np.float32(b)] 
                        y=mz_RT_list[j][1]
    #                    peak_RT_1=RT_list[rt_1_s]
                        weight_mz=y

                        new_id=len(merge_isotopes)
                        mz_weight=[weight_mz]
    #                    peak_RT_list=[peak_RT_1]
                        merge_isotopes[new_id]=[mz_weight, a, b, -1, weight_mz, [mz]] #, peak_RT_list]  
                        list_dict[mz][j][2]=[]
                        list_dict[mz][j][2].append(new_id)                                      
        #                list_dict[mz][j]=[0, 0, -1]

#                print('merge isotopes done')

            isotope_table=defaultdict(list)
            for i in range (0, len(merge_isotopes)):
                if merge_isotopes[i][1]==merge_isotopes[i][2]:
                    continue
                mz_weight_list=merge_isotopes[i][0]
                max_weight=-1
                mz_index=-1
                for j in range(0, len(mz_weight_list)):
                    if mz_weight_list[j]>=max_weight:
                        max_weight=mz_weight_list[j]
                        mz_index=j
                isotope_table[round(merge_isotopes[i][5][mz_index], new_mz_resolution)].append([round(merge_isotopes[i][1]+(merge_isotopes[i][2]-merge_isotopes[i][1])/2, k0_resolution), merge_isotopes[i][1],merge_isotopes[i][2],merge_isotopes[i][4],merge_isotopes[i][5]])

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
                    mz_tolerance_10ppm=round((next_mz_exact*ppm)/10**6, new_mz_resolution)
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
#                                if RT_index_array[np.float32(next_iso[0])]>RT_index_array[np.float32(current_peak)]+tolerance_RT:
#                                    break

#                                if RT_index_array[np.float32(current_peak)]-tolerance_RT<=RT_index_array[np.float32(next_iso[0])] and RT_index_array[np.float32(next_iso[0])]<=RT_index_array[np.float32(current_peak)]+tolerance_RT: # and current_iso[3] >= ((next_iso[3]*3)/4) :
                                   # within tolerance. Check RT range
                                a=current_iso[1]
                                b=current_iso[2]
                                c=next_iso[1]
                                d=next_iso[2]
                                if np.abs(current_iso[0]-next_iso[0])<=k0_tolerance or (a<=d and b>=c): # (a<=d and b>=c)or (np.abs(a-d)<=k0_tolerance and np.abs(b-c)<=k0_tolerance): #overlapped
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
                                mz_tolerance_10ppm=round((next_mz_exact*ppm)/10**6, new_mz_resolution)
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
#                            isotope_cluster[id].append([z]) # charge
                    else: #else: insert them in to the single iso table
        #                id=len(isotope_cluster)
                        isotope_cluster[id].append([current_mz, current_iso])    
#                            isotope_cluster[id].append([z])

                    isotope_table[mz][i]=[-1] #remove it
        #        if DEBUG==1:
        #            break


            #########################################
#                print(len(isotope_cluster.keys()))
            total_cluster=len(isotope_cluster.keys())
            temp_isotope_cluster=copy.deepcopy(isotope_cluster)
            isotope_cluster=defaultdict(list)
            total_clusters=len(temp_isotope_cluster.keys())

            for i in range (0, total_clusters):
                ftr=copy.deepcopy(temp_isotope_cluster[i])
                isotope_cluster[round(ftr[0][0], new_mz_resolution)].append(ftr) # starting m/z of the 1st isotope
#                    isotope_cluster[round(ftr[0][0], 2)].append(ftr) # 

            temp_isotope_cluster=0
            cluster_for_rt.append([len(isotope_cluster), isotope_cluster])
            #isotope_cluster[mz]=[[[mz,iso],[mz,iso]],[],[]]

#            gc.collect()
        ##########################################

        cluster_for_rt=sorted(cluster_for_rt, reverse=True, key=itemgetter(0))
        i=1
        if len(cluster_for_rt)>0:
            cluster_dict=copy.deepcopy(cluster_for_rt[0][1])
        while i<len(cluster_for_rt):
            cluster_for_rt_keys=sorted(cluster_for_rt[i][1].keys())
            for mz in cluster_for_rt_keys:

                for ftr_new in cluster_for_rt[i][1][mz]:
                    cluster_found=0

                    mz_tolerance_ppm=round((mz*ppm)/10**6, new_mz_resolution)
                    mz_left_limit=round(mz-mz_tolerance_ppm, new_mz_resolution)
                    mz_right_limit=round(mz+mz_tolerance_ppm, new_mz_resolution)
                    mz_range=[]
                    mz_list=sorted(cluster_dict.keys())
                    find_mz_idx_start= bisect.bisect_left(mz_list, mz_left_limit)
                    while len(mz_list)!=find_mz_idx_start and round(mz_list[find_mz_idx_start], new_mz_resolution)<=mz_right_limit:     
                        mz_range.append(round(mz_list[find_mz_idx_start], new_mz_resolution))
                        find_mz_idx_start=find_mz_idx_start+1                        


                    for mz in mz_range:
                        if mz in cluster_dict: #mono isotopes matched
                            # do merge
                            for ftr_idx in range (0, len(cluster_dict[mz])): #its a list
                                ftr=cluster_dict[mz][ftr_idx]
                                if ftr[0][0]<=ftr_new[0][0]: #iterate over the iso of ftr to see which iso of ftr match with monoiso of new_ftr
                                    a=ftr_new[0][1][1]
                                    b=ftr_new[0][1][2]
                                    found_iso=-1
                                    for isotope in range (0, len(ftr)):
                                        c=ftr[isotope][1][1]
                                        d=ftr[isotope][1][2]
                                        if (a<=d and b>=c) or np.abs(round(ftr_new[0][1][0]-ftr[isotope][1][0], 2))<=k0_tolerance: #overlap 
                                            found_iso=isotope #old ftr er kar shate mile
                                            break
                                        if np.abs(k0_index_array[np.float32(b)]-k0_index_array[np.float32(c)])<=E or np.abs(k0_index_array[np.float32(a)]-k0_index_array[np.float32(d)])<=E:
                                            found_iso=isotope #old ftr er kar shate mile
                                            break

                                    if found_iso>-1: # mono iso of new ftr matched with found_iso of ftr   
                                        ftr_i=0
                                        for isotope in range (found_iso, min((len(ftr), len(ftr_new)))):
                                            cluster_dict[mz][ftr_idx][isotope][1][1]=min(cluster_dict[mz][ftr_idx][isotope][1][1], ftr_new[ftr_i][1][1])
                                            cluster_dict[mz][ftr_idx][isotope][1][2]=max(cluster_dict[mz][ftr_idx][isotope][1][2], ftr_new[ftr_i][1][2])
                                            cluster_dict[mz][ftr_idx][isotope][1][3]=cluster_dict[mz][ftr_idx][isotope][1][3] + ftr_new[ftr_i][1][3]

                                            cluster_dict[mz][ftr_idx][isotope][1][4].extend(ftr_new[ftr_i][1][4])
                                            temp=list(set(cluster_dict[mz][ftr_idx][isotope][1][4]))
                                            cluster_dict[mz][ftr_idx][isotope][1][4]=temp
                                            cluster_dict[mz][ftr_idx][isotope][1][0]=round((cluster_dict[mz][ftr_idx][isotope][1][1]+cluster_dict[mz][ftr_idx][isotope][1][2])/2, k0_resolution)
                                            ftr_i=ftr_i+1

                                        while ftr_i < len(ftr_new):
                                            cluster_dict[mz][ftr_idx].append(ftr_new[ftr_i]) 
                                            ftr_i=ftr_i+1

#                                            cluster_dict[mz][ftr_idx].append(z)    
                                        cluster_found=1
                                        break
                                else:
                                    a=ftr[0][1][1]
                                    b=ftr[0][1][2]
                                    found_iso=-1
                                    for isotope in range (0, len(ftr_new)): #iterate over the iso of new feature to see which iso of new feature match with monoiso of ftr
                                        c=ftr_new[isotope][1][1]
                                        d=ftr_new[isotope][1][2]
                                        if (a<=d and b>=c) or np.abs(round(ftr_new[isotope][1][0]-ftr[0][1][0], 2))<=k0_tolerance: #overlap 
                                            found_iso=isotope
                                            break

                                        if np.abs(k0_index_array[np.float32(b)]-k0_index_array[np.float32(c)])<=E or np.abs(k0_index_array[np.float32(a)]-k0_index_array[np.float32(d)])<=E:
                                            found_iso=isotope #old ftr er kar shate mile
                                            break

                                    if found_iso>-1: 
                                        old_ftr_i=0
                                        for isotope in range (found_iso, min((len(ftr), len(ftr_new)))):
                                            cluster_dict[mz][ftr_idx][old_ftr_i][1][1]=min(cluster_dict[mz][ftr_idx][old_ftr_i][1][1], ftr_new[isotope][1][1])
                                            cluster_dict[mz][ftr_idx][old_ftr_i][1][2]=max(cluster_dict[mz][ftr_idx][old_ftr_i][1][2], ftr_new[isotope][1][2])
                                            cluster_dict[mz][ftr_idx][old_ftr_i][1][3]=cluster_dict[mz][ftr_idx][old_ftr_i][1][3] + ftr_new[isotope][1][3]

                                            cluster_dict[mz][ftr_idx][old_ftr_i][1][4].extend(ftr_new[isotope][1][4])
                                            temp=list(set(cluster_dict[mz][ftr_idx][old_ftr_i][1][4]))
                                            cluster_dict[mz][ftr_idx][old_ftr_i][1][4]=temp
                                            cluster_dict[mz][ftr_idx][old_ftr_i][1][0]=round((cluster_dict[mz][ftr_idx][old_ftr_i][1][1]+cluster_dict[mz][ftr_idx][old_ftr_i][1][2])/2, k0_resolution)

                                            old_ftr_i=old_ftr_i+1

                                        isotope=isotope+1
                                        while isotope < len(ftr_new):
                                            cluster_dict[mz][ftr_idx].append(ftr_new[isotope]) 
                                            isotope=isotope+1

                                        old_ftr_i=found_iso-1
                                        while old_ftr_i>=0:
                                            cluster_dict[mz][ftr_idx].appendleft(ftr_new[old_ftr_i])
                                            old_ftr_i=old_ftr_i-1

#                                            cluster_dict[mz][ftr_idx].append(z)
                                        cluster_found=1
                                        break
                                if cluster_found==1:
                                    break
                            if cluster_found==1:
                                break
                        if cluster_found==1:
                            break

                    if cluster_found==0:
                       cluster_dict[mz].append(ftr_new)
            i=i+1
        # all done. Now cluster_dict give you a list of features for the current feature in feature_table
        cluster_dict_keys=sorted(cluster_dict.keys())
        found=0

        for mz in cluster_dict_keys:
            if np.abs(round(ftr_list[f][0][0]-mz, new_mz_resolution))<=mz_tolerance:
                temp_list=[]
                k0_list=[]
                for ftr in cluster_dict[mz]:
#                        temp_list.append([len(ftr), ftr])
                    found_k0=0
                    for k0 in k0_list:
                        if np.abs(k0-ftr[0][1][0])<=k0_tolerance:
                            found_k0=1
                            break

                    if found_k0==0:        
                        temp_list.append(ftr)

                    k0_list.append(ftr[0][1][0])



                for ftr in temp_list:
                    temp_ftr=copy.deepcopy(ftr_list[f])
                    temp_ftr[0].append([ftr[0][1][1], ftr[0][1][2]])
                    new_feature_table[round(temp_ftr[0][1][5], mz_resolution)].append(temp_ftr)
                    total_feature=total_feature+1
                    found=1                         
#                        
        if found==0:
            not_found=not_found+1


    print('mz key %g, found:%d, not found:%d, out of %d'%(mz_key, total_feature, not_found, total_feature_input))
##############################################
print('ppm %d, k0_tol %g, A %d, E %d'%(ppm, k0_tolerance, A, E))
f=gzip.open(scanpath+sample_name+'_k0_matched_cluster_part_'+str(target_part), 'wb') #7-p02-3
pickle.dump(new_feature_table, f, protocol=3)
f.close()  
print('end time %g '%(time()-start_time))

