# python nohup -u read_pointCloud.py [filepath] [topath] [filename] > output.log &
# python nohup -u read_pointCloud.py /data/anne/dilution_series_syn_pep/ /data/anne/dilution_series_syn_pep/hash_record/ 130124_dilA_1_01 > output.log &
filepath=sys.argv[1]  output.log &
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import pickle
from collections import defaultdict


filepath=sys.argv[1] 
topath=sys.argv[2] 
filename=sys.argv[3]

delim=','
mz_resolution=5
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
    
   
print(filepath+filename)
print("reading file to convert it to a hash table")
f = open(filepath+filename+'.ms1', 'r') 
line=f.readline()
RT_mz_I_dict=defaultdict(list)
i=0
j=0
maxI=0
max_mz=0
print("conversion starts ...")
while line!='':
    if line.find('RTime')>=0:
        temp=line.split('\t')
        temp=temp[len(temp)-1]
        temp=temp.split('\n')
        temp=temp[len(temp)-2]
        RT_value=round(float(temp), 2)       #?
        line=f.readline()  
        line=f.readline()    
        line=f.readline()  
        line=f.readline() 
        j=0
#            print('found RT')
        while line!='' and line.find('S')<0:
            temp=line.split(' ')
            #print(temp[0])
            mz_value=round(float(temp[0]), mz_resolution) #?
            temp=temp[1].split('\n')
            temp=temp[len(temp)-2]
            intensity_value=round(float(temp), 4)
#                if max_mz<mz_value:
#                   max_mz=mz_value                 

            if maxI<intensity_value:
               maxI=intensity_value 

            RT_mz_I_dict[RT_value].append((mz_value, intensity_value))

            line=f.readline()  
            j=j+1
        i=i+1
    if line!='':   
        line=f.readline()    
f.close()


RT_list = np.sort(list(RT_mz_I_dict.keys()))
sorted_mz_list=[]
RT_index=defaultdict(dict)
for i in range(0, len(RT_list)):
    mz_dict=defaultdict(list)
    for j in range (0, len(RT_mz_I_dict[RT_list[i]])):
        mz_dict[round(RT_mz_I_dict[RT_list[i]][j][0], mz_resolution)].append(round(RT_mz_I_dict[RT_list[i]][j][1], 2))

    mz_keys=sorted(mz_dict.keys())
    for j in range (0, len(mz_keys)):
        RT_index[round(RT_list[i], 2)][round(mz_keys[j], mz_resolution)]=[max(mz_dict[mz_keys[j]]), j]

    sorted_mz_list.append(mz_keys)

print("conversion done. writing records. ")

f=open(topath+'feature_list/pointCloud_'+dataname[data_index]+'_RT_index_new_mz5', 'wb')
pickle.dump(RT_index,  f, protocol=3)
f.close()
    
f=open(topath+filename+'_ms1_record_mz5', 'wb')
pickle.dump([sorted_mz_list,maxI], f, protocol=3) #all mz_done
f.close()

print("writing done.")

