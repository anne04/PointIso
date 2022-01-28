from pyteomics import mzml #, auxiliary 
from collections import defaultdict
import pickle
#import gc
import gzip
#import joblib
import numpy as np

rt_resolution=5
k0_resolution=4
mz_resolution=5
path='/data/fzohora/timsTOF/'  #'/media/anne/Study/bsi/dilution_series_syn_peptide/feature_list/' #'/data/fzohora/water_raw_ms1/'
dataname=['A1_1_2042','A2_1_2043','A3_1_2044','A4_1_2045','A5_1_2046','A6_1_2047', 'A7_1_2048', 'A8_1_2049','A9_1_2050', 'A10_1_2051','A11_1_2052', 'A12_1_2053', 'B1_1_2054',  'B2_1_2055', 'B3_1_2056', 'B4_1_2057']
data_index=0

for data_index in range (1, len(dataname)):
    print(dataname[data_index])
    ##########################################################
    reader=mzml.MzML('/media/anne/Dataset/HeLa_5min_raw/20180924_50ngHeLa_1.0.25.1_Hystar5.0SR1_S2-'+dataname[data_index]+'.mzML') #mzml.MzML('/data/fzohora/timsTOF/20180924_50ngHeLa_1.0.25.1_Hystar5.0SR1_S2-'+dataname[data_index]+'.mzML')
    index_reader=0
    max_I_k0=0
    print('read done')
    dict_scan_k0=defaultdict(list)
    while (index_reader<len(reader) and int(reader[index_reader]['spectrum title'].split('=')[1].split(" ")[0])<2):
        it=reader[index_reader]
        scan=int(it['spectrum title'].split('=')[2].split('"')[0])
        k0_value=np.float32(round(it['scanList']['scan'][0]['inverse reduced ion mobility'], k0_resolution))
        dict_scan_k0[scan].append(k0_value)        
        index_reader=index_reader+1
  
    f=gzip.open('/home/anne/Desktop/bsi/timsTOF/PXD010012/mq/'+dataname[data_index]+'_scan_k0', 'wb') #gzip.open('/data/fzohora/timsTOF/'+dataname[data_index]+'_RT_k0', 'wb')
    pickle.dump(dict_scan_k0, f, protocol=3)
    f.close()
    print('write done')
