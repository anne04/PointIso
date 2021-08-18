# nohup python -u read_pointcloud_4DtimsTOF.py [filepath] [filename] [sample_name] [topath] > output.log &
'''nohup python -u read_pointcloud_4DtimsTOF.py '/data/anne/timsTOF/' 20180924_50ngHeLa_1.0.25.1_Hystar5.0SR1_S2-A1_1_2042.mzML  
A1_1_2042 '/data/anne/timsTOF/hash_records/' > output.log & '''

from pyteomics import mzml
import pickle
import gc
import gzip
import numpy as np

rt_resolution=2
k0_resolution=4
mz_resolution=5

filepath=sys.argv[1]
filename=sys.argv[2]
sample_name=sys.argv[3]
topath=sys.argv[4]


##########################################################
reader=mzml.MzML(filepath+filename) 
max_I=0
index_reader=0
it=reader[index_reader]
max_I_k0=0
print('read done')
for part in range (1, 13):
    dict_rt_mz_k0_i=dict()
    gc.collect()
    limit_rt=np.float32(round(part*0.5, rt_resolution))
    while (index_reader<len(reader)):
        rt_value=np.float32(round(it['scanList']['scan'][0]['scan start time']/60,rt_resolution))
        if rt_value>limit_rt : 
            break 

        if rt_value not in dict_rt_mz_k0_i:
            dict_rt_mz_k0_i[rt_value]=dict()

        mz_array=it['m/z array']
        i_array=it['intensity array']
        k0_value=np.float32(round(it['scanList']['scan'][0]['inverse reduced ion mobility'], k0_resolution))
        for index in range (0, mz_array.shape[0]):
            mz=np.float32(round(mz_array[index], mz_resolution))

            if mz not in dict_rt_mz_k0_i[rt_value]:
                dict_rt_mz_k0_i[rt_value][mz]=dict()  #defaultdict(list)

            intensity_value=np.float32(round(i_array[index], 4))
            if k0_value not in dict_rt_mz_k0_i[rt_value][mz]: 
                dict_rt_mz_k0_i[rt_value][mz][k0_value]=[intensity_value]
            else:
                dict_rt_mz_k0_i[rt_value][mz][k0_value]=[np.float32(round(dict_rt_mz_k0_i[rt_value][mz][k0_value][0]+intensity_value, 4))]

            if dict_rt_mz_k0_i[rt_value][mz][k0_value][0]>max_I_k0:
               max_I_k0=dict_rt_mz_k0_i[rt_value][mz][k0_value]

        it=reader[index_reader]
        index_reader=index_reader+1

    print('read done with index %d rt_value %g'%(index_reader, rt_value))

    for rt_value in dict_rt_mz_k0_i.keys():
        for mz_value in dict_rt_mz_k0_i[rt_value].keys():
            total_intensity=0

            for k0_value in dict_rt_mz_k0_i[rt_value][mz_value].keys():

                total_intensity=total_intensity+dict_rt_mz_k0_i[rt_value][mz_value][k0_value][0]

            dict_rt_mz_k0_i[rt_value][mz_value][6]=np.float32(round(total_intensity, 4))
            if max_I<total_intensity:
                max_I=total_intensity



    f=gzip.open(topath+sample_name+'_RT_index_part'+str(part), 'wb') 
    pickle.dump(dict_rt_mz_k0_i, f, protocol=3)
    f.close()
    print('write done')
    if (index_reader>=len(reader)):
        break
print('all done %d'%len(reader))

f=open(topath+'pointCloud_'+sample_name+'_maxI', 'wb') 
pickle.dump(max_I, f, protocol=3)
f.close()


f=open(topath+'pointCloud_'+sample_name+'_maxI_k0', 'wb') 
pickle.dump(max_I_k0, f, protocol=3)
f.close()

print('write done')
    
