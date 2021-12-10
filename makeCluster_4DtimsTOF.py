#nohup python -u makeCluster.py [recordpath] [sample_name] [scanpath] > output.log &
'''nohup python -u makeCluster.py '/data/anne/timsTOF/hash_records/' 'A1_1_2042' '/data/anne/timsTOF/scanned_result/' > output.log & '''
from __future__ import division
from __future__ import print_function
from time import time
import pickle
import numpy as np
from collections import defaultdict
import copy
import sys
import bisect
import gzip
from operator import itemgetter

recordpath= sys.argv[1]
sample_name=sys.argv[2]
scanpath=sys.argv[3]


proton_mass= 1.00727567
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
max_part=12
RT_window=10
mz_resolution=5 #
RT_resolution=2
total_class=10
RT_unit=0.01
mz_unit=0.00001
num_class=10

new_mz_resolution=3
new_mz_unit=0.001

print('reading list_dict')


f=gzip.open(scanpath+sample_name+'_timsTOF_list_dict_'+'1', 'rb') #v3r2
list_dict, part_dict=pickle.load(f)
f.close()
for part in range (2, max_part+1):
    f=gzip.open(scanpath+sample_name+'_timsTOF_list_dict_'+str(part), 'rb')
    list_dict_next, part_dict=pickle.load(f)
    f.close()        
    for z in range (1, 10):
        for mz in list_dict_next[z].keys():
            if mz in list_dict[z]:
                list_dict[z][mz].extend(list_dict_next[z][mz])
            elif len(list_dict_next[z][mz])>0:
                list_dict[z][mz]=list_dict_next[z][mz]


print('done')


RT_index=dict()
for part in range (1, max_part+1):
    print(part)
    f=gzip.open(recordpath+sample_name+'_RT_index_part'+str(part), 'rb') 
    RT_index_temp=pickle.load(f)
    f.close()
    RT_index.update(RT_index_temp)

print('read done')
RT_list=sorted(RT_index.keys())
max_RT=RT_list[len(RT_list)-1]
min_RT=RT_list[0]

RT_index_array=dict()
for i in range (0, len(RT_list)):
    RT_value=round(RT_list[i], 2) 
    RT_index_array[RT_value]=i


#f=gzip.open(recordpath+sample_name+'_RT_index_array_x4', 'wb') #
#pickle.dump(RT_index_array, f, protocol=3)
#f.close()


#    f=gzip.open(recordpath+sample_name+'_RT_index_array', 'rb')
#    RT_index_array=pickle.load(f)
#    f.close()


max_intensity_cluster=-1
isotope_cluster=defaultdict(list)

A=2 
B=1
C=2 
D=2 
ppm=10 
for z in range (1, 10):
    new_list_dict=defaultdict(list)
    mz_list=sorted(list_dict[z].keys())
    start_time=time()
    j=0
    while j<len(mz_list):
        mz_value=round(mz_list[j], new_mz_resolution)  
        if len(list_dict[z][mz_list[j]])==0:
            list_dict[z].pop(mz_list[j])
            j=j+1
            continue

        new_list_dict[mz_value]=list_dict[z][mz_list[j]]
        list_dict[z].pop(mz_list[j])    
#            print('start %d'%len(new_list_dict[mz_value]))
        lower_mz_limit=mz_list[j]
        higher_mz_limit=mz_list[j]
        j=j+1
        while j<len(mz_list) and round(mz_list[j], new_mz_resolution)==mz_value:
            if len(list_dict[z][mz_list[j]])==0:
                list_dict[z].pop(mz_list[j])
                j=j+1
                continue

            new_list_dict[mz_value].extend(list_dict[z][mz_list[j]])
            higher_mz_limit=mz_list[j]
            list_dict[z].pop(mz_list[j])
            j=j+1


        new_list_dict[mz_value]=sorted(new_list_dict[mz_value], key=itemgetter(0))
#            print('end %d'%len(new_list_dict[mz_value]))
        #merge the duplicates or overlappings if there is any
        temp_list=[]
        k=0
        a=min(new_list_dict[mz_value][k][0])
        b=max(new_list_dict[mz_value][k][0])
        intensity_dict=new_list_dict[mz_value][k][1]

        k=1
        while k < len(new_list_dict[mz_value]):                
            c=min(new_list_dict[mz_value][k][0])
            d=max(new_list_dict[mz_value][k][0])
            if a<=d and b>=c: #overlap
                #merge the dict
                intensity_list=new_list_dict[mz_value][k][1].keys()
                for rt in intensity_list:
                    if rt in intensity_dict:
                        intensity_dict[rt]=intensity_dict[rt]+new_list_dict[mz_value][k][1][rt]
                    else:
                        intensity_dict[rt]=new_list_dict[mz_value][k][1][rt]

                a=min(a, c)
                b=max(b, d)


            else:
#                    if intensity_sum>0:         # this might be a area not recoorded due to noise
                temp_list.append([[a, b], intensity_dict, -1])
                a=min(new_list_dict[mz_value][k][0])
                b=max(new_list_dict[mz_value][k][0])
                intensity_dict=new_list_dict[mz_value][k][1]

            k=k+1
#            if intensity_sum>0:               # remove the false detections caused by saying YES ahead of time 
        temp_list.append([[a, b], intensity_dict, -1])

        new_list_dict[mz_value]=temp_list
        # now it has no duplicate or overlapping
        ###########################################
        #merge the pairs if possible
        temp_list=[]
        k=0
        a=min(new_list_dict[mz_value][k][0])
        b=max(new_list_dict[mz_value][k][0])
        intensity_dict=new_list_dict[mz_value][k][1]
        k=1
        while k < len(new_list_dict[mz_value]):                
            c=min(new_list_dict[mz_value][k][0])
            d=max(new_list_dict[mz_value][k][0])
#                print('b-%g, c-%g, %d, %d'%(b, c, RT_index_array[b], RT_index_array[c]))
            if np.abs(RT_index_array[b]-RT_index_array[c])<=A: #A: mergable 
                #merge
                intensity_list=new_list_dict[mz_value][k][1].keys()
                for rt in intensity_list:
#                        if rt in intensity_dict:
#                            intensity_dict[rt]=intensity_dict[rt]+new_list_dict[mz_value][k][1][rt]
#                        else:
                    intensity_dict[rt]=new_list_dict[mz_value][k][1][rt]
                a=min(a, c)
                b=max(b, d)


            else:
                if np.abs(RT_index_array[a]-RT_index_array[b])>=B:        # B: remove those traces whose extent is B-1 consecutive scans
                    a=round(a, RT_resolution)
                    b=round(b, RT_resolution)
                    # find RT_peak and intensity_sum
                    intensity_list=sorted(intensity_dict.keys())
                    max_intensity=intensity_sum=intensity_dict[intensity_list[0]]
                    RT_peak=intensity_list[0]
                    for m in range (1, len(intensity_list)):
                        if intensity_dict[intensity_list[m]]>max_intensity:
                            max_intensity=intensity_dict[intensity_list[m]]
                            RT_peak=intensity_list[m]
                        intensity_sum=intensity_sum+intensity_dict[intensity_list[m]]
                    if intensity_sum>0:
                        temp_list.append([[a, b], RT_peak, -1,  intensity_sum, intensity_dict]) #max_intensity,

                a=min(new_list_dict[mz_value][k][0])
                b=max(new_list_dict[mz_value][k][0])
                intensity_dict=new_list_dict[mz_value][k][1]

            k=k+1
        if np.abs(RT_index_array[a]-RT_index_array[b])>=B:        # remove those traces whose extent is B-1 consecutive scans
            a=round(a, RT_resolution)
            b=round(b, RT_resolution)
            intensity_list=sorted(intensity_dict.keys())
            max_intensity=intensity_sum=intensity_dict[intensity_list[0]]
            RT_peak=intensity_list[0]
            for m in range (1, len(intensity_list)):
                if intensity_dict[intensity_list[m]]>max_intensity:
                    max_intensity=intensity_dict[intensity_list[m]]
                    RT_peak=intensity_list[m]
                intensity_sum=intensity_sum+intensity_dict[intensity_list[m]]

            if intensity_sum>0:
                temp_list.append([[a, b], RT_peak, -1,  intensity_sum, intensity_dict]) #max_intensity,

        new_list_dict[mz_value]=temp_list
        # now no more merging
        if len(new_list_dict[mz_value])>0:
            new_list_dict[mz_value].append(round((lower_mz_limit+higher_mz_limit)/2, mz_resolution))
            list_dict[z][mz_value]=new_list_dict[mz_value]

    print('time for merging mz resolution %g'%(time()-start_time))
    start_time=time()
    merge_isotopes=dict() #based on id 
    list_keys=sorted(list_dict[z].keys())
    if len(list_keys)==1:
        list_dict[z].pop(round(list_keys[0],  new_mz_resolution))
    list_keys=sorted(list_dict[z].keys()) 
    max_dict=len(list_keys)-1
    i=0
    j=0
    k=0
    for i in range (0, max_dict):
#            print('max_dict %d'%i)
        mz_pred=round(list_keys[i], new_mz_resolution)
        mz=round(list_keys[i+1], new_mz_resolution)
        dynamic_mz_unit=round((mz_pred*ppm)/10**6, new_mz_resolution) # mz_pred is 700.1204, mz is 700.1274 or less than that, then fine to merge them
        if mz<=round(mz_pred+dynamic_mz_unit, new_mz_resolution):
            mz_pred_RT_list=list_dict[z][mz_pred]
            mz_RT_list=list_dict[z][mz]
            k=0
            for j in range (0, len(mz_pred_RT_list)-1):
                a=round(mz_pred_RT_list[j][0][0], 2) #actual floating point RT value
                b=round(mz_pred_RT_list[j][0][1], 2) #actual  floating point RT value
                id=mz_pred_RT_list[j][2]
                #mz_pred is the actual floating point mz value
                mz_point1= mz_pred
                y=mz_pred_RT_list[j][3] #RT_index[RT_list[rt_1_s]][mz_point1][0]
                peak_RT_1=mz_pred_RT_list[j][1]
                intensity_dict_1=mz_pred_RT_list[j][4]
                weight_pred_mz=round(y, 2)

                #find the next overlapped 
                p=k
                max_overlapped_area=-1
                max_overlapped_index=-1
                while p < len(mz_RT_list)-1:
#                        print('p %d'%p)
                    c=round(mz_RT_list[p][0][0], 2)
                    d=round(mz_RT_list[p][0][1], 2)
                    #check overlapping: if (RectA.Left < RectB.Right && RectA.Right > RectB.Left..)
                    if c>=b:
                        break
                    elif a<d and b>c: #overlap 
                        mz_point2= mz
                        y=mz_RT_list[p][3]
                        peak_RT_2=mz_RT_list[p][1]                            
                        # C
                        if abs(RT_index_array[np.float32(peak_RT_1)]-RT_index_array[np.float32(peak_RT_2)])<=C: #changed from 2
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
                        ############################################
                        intensity_dict=defaultdict(dict)
                        rt_keys=sorted(intensity_dict_1.keys())
                        for rt in rt_keys:
                            intensity_dict[rt][mz_pred]=intensity_dict_1[rt]
                            if max_intensity_cluster<intensity_dict[rt][mz_pred]:
                                max_intensity_cluster=intensity_dict[rt][mz_pred]
                        ############################################
                        merge_isotopes[new_id]=[mz_weight, a, b, -1, intensity_dict, [mz_pred], peak_RT_list]  
                        list_dict[z][mz_pred][j][2]=[]
                        list_dict[z][mz_pred][j][2].append(new_id)                      
#                        list_dict[z][mz_pred][j]=[0, 0, -1] #-- pop
                    k=p
                    continue
                # else 
                c=round(mz_RT_list[max_overlapped_index][0][0], 2)
                d=round(mz_RT_list[max_overlapped_index][0][1], 2)                                                                     
                mz_point2= mz
                y=mz_RT_list[max_overlapped_index][3]
                peak_RT_2=mz_RT_list[max_overlapped_index][1]                 
                intensity_dict_2=mz_RT_list[max_overlapped_index][4]
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
                    #intensity_dict_1 and intensity_dict_2 are two different mz. So they have no overlapping. 
                    #So simpley updating must work
                    ##################################
                    intensity_dict=defaultdict(dict)
                    rt_keys=sorted(intensity_dict_1.keys())
                    for rt in rt_keys:
                        intensity_dict[rt][mz_pred]=intensity_dict_1[rt]
                        if max_intensity_cluster<intensity_dict[rt][mz_pred]:
                            max_intensity_cluster=intensity_dict[rt][mz_pred]

                    rt_keys=sorted(intensity_dict_2.keys())
                    for rt in rt_keys:
                        intensity_dict[rt][mz]=intensity_dict_2[rt]
                        if max_intensity_cluster<intensity_dict[rt][mz]:
                            max_intensity_cluster=intensity_dict[rt][mz]

                    ###################################


                    merge_isotopes[new_id]=[mz_weight, grp_rt_st, grp_rt_end, auc, intensity_dict, [mz_pred, mz], peak_RT_list]
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
#                            merge_isotopes[pred_id][4]=merge_isotopes[pred_id][4]+intensity_2
                        #############################################################
                        rt_keys=sorted(intensity_dict_2.keys())
                        for rt in rt_keys:
                            merge_isotopes[pred_id][4][rt][mz]=intensity_dict_2[rt]  
                            if max_intensity_cluster<merge_isotopes[pred_id][4][rt][mz]:
                                max_intensity_cluster=merge_isotopes[pred_id][4][rt][mz]


                        ############################################################
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
            for j in range (0, len(mz_pred_RT_list)-1):
                a=round(mz_pred_RT_list[j][0][0], 2) #actual floating point RT value
                b=round(mz_pred_RT_list[j][0][1], 2) #actual  floating point RT value
                id=mz_pred_RT_list[j][2]
                #mz_pred is the actual floating point mz value
                mz_point1= mz_pred
                y=mz_pred_RT_list[j][3]
                peak_RT_1=mz_pred_RT_list[j][1]
                intensity_dict_1=mz_pred_RT_list[j][4]
                ##################################
                intensity_dict=defaultdict(dict)
                rt_keys=sorted(intensity_dict_1.keys())
                for rt in rt_keys:
                    intensity_dict[rt][mz_pred]=intensity_dict_1[rt]
                    if max_intensity_cluster<intensity_dict[rt][mz_pred]:
                        max_intensity_cluster=intensity_dict[rt][mz_pred]

                ###################################

                weight_pred_mz=round(y, 2)
                new_id=len(merge_isotopes)
                mz_weight=[weight_pred_mz]
                peak_RT_list=[peak_RT_1]
                merge_isotopes[new_id]=[mz_weight, a, b, -1, intensity_dict, [mz_pred], peak_RT_list]  
                list_dict[z][mz_pred][j][2]=[]
                list_dict[z][mz_pred][j][2].append(new_id)



    if len(list_keys)!=0:
        i=i+1                        
        mz=round(list_keys[i], new_mz_resolution)
        mz_RT_list=list(list_dict[z][mz])
        list_dict[z][mz]=mz_RT_list
        for j in range (0, len(mz_RT_list)-1):
            if mz_RT_list[j][2]==-1:
                a=round(mz_RT_list[j][0][0], 2)
                b=round(mz_RT_list[j][0][1], 2)     

                mz_point1= mz
                y=mz_RT_list[j][3]
                peak_RT_1=mz_RT_list[j][1]
                intensity_dict_1=mz_RT_list[j][4]
                ##################################
                intensity_dict=defaultdict(dict)
                rt_keys=sorted(intensity_dict_1.keys())
                for rt in rt_keys:
                    intensity_dict[rt][mz]=intensity_dict_1[rt]
                    if max_intensity_cluster<intensity_dict[rt][mz]:
                        max_intensity_cluster=intensity_dict[rt][mz]

                ###################################
                weight_mz=y
                new_id=len(merge_isotopes)
                mz_weight=[weight_mz]
                peak_RT_list=[peak_RT_1]
                merge_isotopes[new_id]=[mz_weight, a, b, -1, intensity_dict, [mz], peak_RT_list]  
                list_dict[z][mz][j][2]=[]
                list_dict[z][mz][j][2].append(new_id)                                      
#                list_dict[z][mz][j]=[0, 0, -1]

    print('merge isotopes done')
    print('time for merging %g'%(time()-start_time))
    start_time=time()
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

        long_mz_index=len(list_dict[z][round(merge_isotopes[i][5][mz_index], new_mz_resolution)])-1
        long_mz=list_dict[z][round(merge_isotopes[i][5][mz_index], new_mz_resolution)][long_mz_index]
        isotope_table[round(merge_isotopes[i][5][mz_index], new_mz_resolution)].append([merge_isotopes[i][6][mz_index], merge_isotopes[i][1],merge_isotopes[i][2],merge_isotopes[i][4],merge_isotopes[i][5], long_mz])

    isotope_mz_list=sorted(isotope_table.keys())

    isotope_table_temp=defaultdict(list)
    for i in isotope_mz_list:
        isotope_table[i]=sorted(isotope_table[i], key=itemgetter(0))
        j=0
        while (j<len(isotope_table[i])):
            if len(isotope_table[i][j])!=0:
                isotope_table_temp[i].append(isotope_table[i][j])
            if j+1>=len(isotope_table[i]):
                break
            for k in range (j+1,  len(isotope_table[i])):
                if (isotope_table[i][j][0]!=isotope_table[i][k][0]):
                    break
            j=k


    isotope_table=copy.deepcopy(isotope_table_temp)
    isotope_table_temp=0

##############################################
    key_list=sorted(isotope_table.keys())
    count=0
    RT_tol=.02 #4
#        ppm=10.0
#        tolerance=0.004
    new_isotope_table=defaultdict(list)
    for i in range (0, len(key_list)) :
        ftr_list_pred=sorted(isotope_table[key_list[i]], key=itemgetter(0))
        for k in range (0, len(ftr_list_pred)):
#                if len(ftr_list_pred[k])==0:
#                    continue
            ftr_pred=ftr_list_pred[k]  #i
            mz_new_ftr=key_list[i]
            tolerance=round((key_list[i]*ppm)/10**6, new_mz_resolution)
            p=i+1

            while (p<len(key_list) and np.abs(round(key_list[i]-key_list[p], new_mz_resolution))<=tolerance): 
#                while (p<len(key_list) and np.abs(round((round(key_list[i], 3)-proton_mass)*z, 3) - round((round(key_list[p], 3)-proton_mass)*z, 3))<=0.008): #

                ftr_list=sorted(isotope_table[key_list[p]], key=itemgetter(0))
                j=0
                while j< len(ftr_list):
                    if len(ftr_list[j])==0 or round(ftr_pred[0]-ftr_list[j][0], 2)>RT_tol:
                        j=j+1
                        continue
                    if ftr_list[j][0]>round(ftr_pred[0]+RT_tol, 2):
                        break
                    ftr=ftr_list[j]  #p
                    a=ftr_pred[1]
                    b=ftr_pred[2]
                    c=ftr[1]
                    d=ftr[2]
                    if a<=d and b>=c:
                        RT_peak_pred=ftr_pred[0]
                        RT_peak=ftr[0]
                        if np.abs(RT_peak_pred-RT_peak)<=RT_tol: #4:
                        # merge 
                            a=min(ftr_pred[1], ftr[1]) #start
                            b=max(ftr_pred[2], ftr[2]) #end
                            temp_dict=copy.deepcopy(ftr_pred[3])

                            for ftr_rt in ftr[3].keys():
                                if ftr_rt not in temp_dict:
                                   temp_dict[ftr_rt]=ftr[3][ftr_rt]
                                else:
                                    for ftr_mz in ftr[3][ftr_rt].keys():
                                        if ftr_mz not in temp_dict[ftr_rt]:
                                            temp_dict[ftr_rt][ftr_mz]=ftr[3][ftr_rt][ftr_mz]
                                        else:
                                            temp_dict[ftr_rt][ftr_mz]=temp_dict[ftr_rt][ftr_mz]+ftr[3][ftr_rt][ftr_mz]

                            max_intensity=-1
                            peak_rt=-1
                            for ftr_rt in temp_dict.keys():
                                sum_intensity=0
                                for ftr_mz in temp_dict[ftr_rt].keys():
                                    sum_intensity=sum_intensity+temp_dict[ftr_rt][ftr_mz]

                                if sum_intensity>max_intensity:
                                    max_intensity=sum_intensity
                                    peak_rt=ftr_rt

                            if max_intensity>max_intensity_cluster:
                                max_intensity_cluster=max_intensity
                            #temp_dict has the merged dict
                            new_ftr=[peak_rt, a, b, temp_dict, sorted(ftr_pred[4]+ftr[4]), round((ftr_pred[5]+ftr[5])/2, mz_resolution)]

                        ##################################


                            ftr_pred=copy.deepcopy(new_ftr)
                            isotope_table[key_list[p]].pop(j)
                            ftr_list.pop(j)
#                                isotope_table[key_list[p]][j]=[]
#                            break
                        else:
                            j=j+1
                    else:
                        j=j+1
                p=p+1        


            new_isotope_table[mz_new_ftr].append(ftr_pred)
            count=count+1

#        print(count)
    isotope_table=copy.deepcopy(new_isotope_table)

##############################################
    # form cluster of isotopes to feed into the isotope grouping module
    DEBUG=0
    mz_list=sorted(isotope_table.keys())
    tolerance_RT=D
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
                        if RT_index_array[np.float32(next_iso[0])]>RT_index_array[np.float32(current_peak)]+tolerance_RT:
                            break

                        if np.abs(RT_index_array[np.float32(current_peak)]-RT_index_array[np.float32(next_iso[0])])<=tolerance_RT: # and current_iso[3] >= ((next_iso[3]*3)/4) :
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
                isotope_cluster[id].append([z]) # charge
            else: #else: insert them in to the single iso table
#                id=len(isotope_cluster)
                isotope_cluster[id].append([current_mz, current_iso])    
                isotope_cluster[id].append([z])

            isotope_table[mz][i]=[-1] #remove it
#        if DEBUG==1:
#            break

    print('time for merging isotopes %g'%(time()-start_time))
#########################################
print(len(isotope_cluster.keys()))
total_cluster=len(isotope_cluster.keys())
temp_isotope_cluster=copy.deepcopy(isotope_cluster)
isotope_cluster=defaultdict(list)
total_clusters=len(temp_isotope_cluster.keys())

for i in range (0, total_clusters):
    ftr=copy.deepcopy(temp_isotope_cluster[i])
#        isotope_cluster[round(ftr[0][0], new_mz_resolution)].append(ftr) # starting m/z of the 1st isotope
    isotope_cluster[round(ftr[0][0], 2)].append([ftr[0][0], ftr]) # starting m/z of the 1st isotope

temp_isotope_cluster=0


temp_isotope_cluster=copy.deepcopy(isotope_cluster)
isotope_cluster=defaultdict(list)
keys_list=sorted(temp_isotope_cluster.keys())
max_num_iso=0
total_cluster=0
for mz in keys_list:
    temp_isotope_cluster[mz]=sorted(temp_isotope_cluster[mz], key=itemgetter(0))
    i=0
    while i<len(temp_isotope_cluster[mz]):
        ftr=copy.deepcopy(temp_isotope_cluster[mz][i][1])
        candidate_for_ftr=[]
        candidate_for_ftr.append([len(ftr), ftr])
        j=i+1
        while j<len(temp_isotope_cluster[mz]): 
            ftr_2=copy.deepcopy(temp_isotope_cluster[mz][j][1])
            if ftr_2[0][0]==ftr[0][0]:
                if ftr[len(ftr)-1][0]==ftr_2[len(ftr_2)-1][0] and ftr[0][1][0]==ftr_2[0][1][0]:
                    temp_isotope_cluster[mz].pop(j)
                    candidate_for_ftr.append([len(ftr_2), ftr_2])
                else:
                    j=j+1
            else:
                break

        candidate_for_ftr=sorted(candidate_for_ftr,  key=itemgetter(0),  reverse=True)
        ftr=candidate_for_ftr[0][1]
        isotope_cluster[round(ftr[0][0], 2)].append(ftr)
        total_cluster=total_cluster+1
        i=i+1
        if (len(ftr)-1)>max_num_iso:
            max_num_iso=(len(ftr)-1)        

print('%d %d'%(max_num_iso, total_cluster)) 
temp_isotope_cluster=0


print('%d %d'%(max_num_iso, total_cluster)) 
f=gzip.open(scanpath+sample_name+'_4DtimsTOF_clusters_mz3_v5c', 'wb') 
pickle.dump([isotope_cluster, max_num_iso, total_cluster], f, protocol=3)
f.close()

f=open(scanpath+sample_name+'_maxI_cluster_v5c', 'wb') 
pickle.dump(max_intensity_cluster, f, protocol=3)
f.close()


print('cluster write done')

#makeCluster_pointnet_timsTOF_v5c.py


