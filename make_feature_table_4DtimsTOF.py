# nohup python -u  make_feature_table_4DtimsTOF.py [scanpath] [sample_name] [resultpath] > output.log &
''' nohup python -u make_feature_table_4DtimsTOF.py '/data/anne/timsTOF/scanned_result/' 'A1_1_2042' '/data/anne/timsTOF/4D_result/' > output.log & '''

from __future__ import division
from __future__ import print_function
import csv
import pickle
import numpy as np
import bisect
from collections import defaultdict
import copy
import gzip


scanpath=sys.argv[1]
sample_name=sys.argv[2]
resultpath=sys.argv[3]


RT_window=15
mz_window=211
frame_width=11
mz_resolution=2
k0_resolution=5
total_class=10 # charge
RT_unit=0.01
fc_size= 4
total_frames_hor=6
num_class=total_frames_hor # number of isotopes to report
state_size = fc_size
num_neurons= num_class #mz_window*RT_window
truncated_backprop_length = 6
result_all=np.zeros((len(dataname), 12))
threshold_score=10.0
percent_feature=20

f=gzip.open(scanpath+sample_name+'_k0_matched_cluster_part_'+str(1), 'rb')
feature_table=pickle.load(f)
f.close() 
max_part=12
for part in range (2, max_part+1):
    f=gzip.open(scanpath+sample_name+'_k0_matched_cluster_part_'+str(part), 'rb')
    feature_table_next=pickle.load(f)
    f.close() 
    for mz in feature_table_next.keys():
        if mz in feature_table:
            feature_table[mz].extend(feature_table_next[mz])
        else:
            feature_table[mz]=feature_table_next[mz]



isotope_cluster=defaultdict(list)
mz_list=feature_table.keys()
total_feature=0
score_list=dict()
for mz in mz_list:
    ftr_list=feature_table[mz]
    for ftr in ftr_list:
        long_mz=ftr[0][1][5]  
        if long_mz<400 or long_mz>1600:
            continue
        isotope_cluster[round(long_mz, mz_resolution)].append(ftr)
        total_feature=total_feature+1

feature_table=isotope_cluster

print('%d'%total_feature)



#Some duplicate features are recorded in the previous step which are discarded here    
#merge features 4d
k0_tolerance=0.01
E=5
isotope_cluster=defaultdict(list)
keys_list=sorted(feature_table.keys())
max_num_iso=0
total_cluster=0
for mz in keys_list:
    ftr_list=feature_table[mz]
    i=0
    while i<len(ftr_list):
        ftr=copy.deepcopy(ftr_list[i])
        j=i+1
        a=ftr[0][2][0]
        b=ftr[0][2][1]
        matched=0
        while j<len(ftr_list):
            ftr_2=copy.deepcopy(ftr_list[j])
            # mono mz, len same, rt peak same:
            if ftr_2[0][0]==ftr[0][0] and len(ftr)==len(ftr_2) and ftr[0][1][1]==ftr_2[0][1][1] and ftr[0][1][2]==ftr_2[0][1][2] and ftr[len(ftr)-1][0]==ftr_2[len(ftr_2)-1][0]:
                # see if k0 range are mergable or not
                # if mergable then merge it and replace ftr_2 with -1
                c=ftr_2[0][2][0]
                d=ftr_2[0][2][1]

                if round(np.abs(round(((a+b)/2)-((c+d)/2), k0_resolution)), 2)<k0_tolerance or (round(np.abs(round(((a+b)/2)-((min(a, c)+max(b, d))/2), k0_resolution)), 2)<k0_tolerance and round(np.abs(round(((c+d)/2)-((min(a, c)+max(b, d))/2), k0_resolution)), 2)<k0_tolerance): #overlap 
                    min_value=min(a, c)
                    max_value=max(b, d)

                    ftr[0][2][0]=min_value
                    ftr[0][2][1]=max_value


                    feature_table[mz].pop(j)
#                            ftr_list.pop(j)
                    a=ftr[0][2][0]
                    b=ftr[0][2][1]
                else:
                    j=j+1
            else: 
                break


        isotope_cluster[round(ftr[0][1][5], mz_resolution)].append(ftr)
        total_cluster=total_cluster+1
        i=i+1



feature_table=copy.deepcopy(isotope_cluster)
print(total_cluster)
##

################ merge feature 3d##################
mz_difference=0.004 #0.004
RT_difference=0.01 #0.15
k0_tolerance=0.01
E=5
#        feature_table=copy.deepcopy(new_feature_table)
isotope_cluster=defaultdict(list)
keys_list=sorted(feature_table.keys())
max_num_iso=0
total_cluster=0
for mz in keys_list:
#            print(total_cluster)
#        feature_table[mz]=sorted(feature_table[mz], key=itemgetter(0))
    i=0
    while i<len(feature_table[mz]):
        ftr=copy.deepcopy(feature_table[mz][i])
        j=i+1
        matched=0
        a=ftr[0][1][1]
        b=ftr[0][1][2]
        while j<len(feature_table[mz]):                    
            ftr_2=copy.deepcopy(feature_table[mz][j])
            c=ftr_2[0][1][1]
            d=ftr_2[0][1][2]
            # mono mz, len same, rt peak same:
            if np.abs(((ftr[0][2][0]+ftr[0][2][1])/2)-((ftr_2[0][2][0]+ftr_2[0][2][1])/2))<k0_tolerance and np.abs(ftr_2[0][0]-ftr[0][0])<=mz_difference and ((a<=d and b>=c) or np.abs(ftr[0][1][0]-ftr_2[0][1][0])<=RT_difference): # and ftr[len(ftr)-1][0]==ftr_2[len(ftr_2)-1][0]: #and len(ftr)==len(ftr_2) and ftr[0][1][2]==ftr_2[0][1][2] 
                
                if ftr[len(ftr)-1][0]!=ftr_2[len(ftr_2)-1][0]:
                    if max(ftr[len(ftr)-1][1]) < max(ftr_2[len(ftr_2)-1][1]):
                        ftr=ftr_2
                else:        
                    for k in range (0, min(len(ftr)-1, len(ftr_2)-1)):
                        ftr[k][0]=round((ftr[k][1][5]+ftr_2[k][1][5])/2, 2)
                        ftr[k][1][5]=round((ftr[k][1][5]+ftr_2[k][1][5])/2, 5)
                        ftr[k][1][0]=round((ftr[k][1][0]+ftr_2[k][1][0])/2, 2) # peak RT
                        ftr[k][1][1]=min(ftr[k][1][1], ftr_2[k][1][1])
                        ftr[k][1][2]=min(ftr[k][1][2], ftr_2[k][1][2])
                        if k==0:
                            ftr[0][2][0]=min(ftr[0][2][0], ftr_2[0][2][0])
                            ftr[0][2][1]=min(ftr[0][2][1], ftr_2[0][2][1])


                        for mz_2 in ftr_2[k][1][3].keys():
                            if mz_2 in ftr[k][1][3]:
                                for rt in ftr_2[k][1][3][mz_2].keys():
                                    if rt in ftr[k][1][3][mz_2]:
                                        ftr[k][1][3][mz_2][rt]=ftr[k][1][3][mz_2][rt]+ftr_2[k][1][3][mz_2][rt]
                                    else:
                                        ftr[k][1][3][mz_2][rt]=ftr_2[k][1][3][mz_2][rt]
                            else:
                                for rt in ftr_2[k][1][3][mz_2].keys():
                                    ftr[k][1][3][mz_2][rt]=ftr_2[k][1][3][mz_2][rt]

                    keep_save=ftr[len(ftr)-1] #score and charge
                    if len(ftr)<len(ftr_2):
                        k=k+1
                        ftr[len(ftr)-1]=ftr_2[k]
                        k=k+1
                        while k<(len(ftr_2)-1):
                            ftr.append(ftr_2[k])
                            k=k+1

                        if max(keep_save[1])>max(ftr_2[len(ftr_2)-1][1]):
                            ftr.append(keep_save)
                        else:
                            ftr.append(ftr_2[len(ftr_2)-1])


                    if max(ftr[len(ftr)-1][1])<max(ftr_2[len(ftr_2)-1][1]):
                        ftr[len(ftr)-1]=ftr_2[len(ftr_2)-1]

                a=ftr[0][1][1]
                b=ftr[0][1][2]

                feature_table[mz].pop(j)
                #

            else: 
                j=j+1


        isotope_cluster[round(ftr[0][1][5], mz_resolution)].append(ftr)
        total_cluster=total_cluster+1
        i=i+1

#
#
feature_table=copy.deepcopy(isotope_cluster)
f=gzip.open(resultpath+sample_name+'_feature_table', 'wb') 
pickle.dump(feature_table, f, protocol=3)
f.close()  
