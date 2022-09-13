from __future__ import division
from __future__ import print_function
import csv
import pickle
import numpy as np
from collections import defaultdict
import scipy.stats
import bisect
import gzip


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
for test_index in range (0, 57):
    f=gzip.open(datapath+'/feature_list/deepIsoV2_'+dataname[test_index]+'_featureTable_v6r1_cv5_ev2r6b_merged_auc_exact_mz_fullRT','rb') #[-1.43,2.44] human-SRM 
    feature_table,auc_list=pickle.load(f)
    f.close()     
    count=0
    mz_list=list(feature_table.keys())
    for i in range (0, len(mz_list)):
        ftr_list=feature_table[mz_list[i]]
        count=count+len(ftr_list)

    total_feature = count
    print('total feature: ',total_feature)

    feature_list=[]
    mz_list=list(feature_table.keys())
    for i in range (0, len(mz_list)):
        ftr_list=feature_table[mz_list[i]]
        for f in range (0, len(ftr_list)):
            ftr=ftr_list[f]
            item=[]
            item.append(ftr[0][0]) # mono-isotopic m/z
            item.append(ftr[0][1][0]) # max intensity RT point
            item.append(ftr[len(ftr)-1][0]) # charge
            item.append(ftr[0][1][1]) # start RT point
            item.append(ftr[0][1][2]) # end RT point
            item.append(len(ftr)-1) # number of isotopes
            item.append(ftr[len(ftr)-1][2]) # total intensity
            item.append(np.max(ftr[len(ftr)-1][1])) # feature score
            item.append(ftr[len(ftr)-1][1]) # last layer softmax output
            
            other_iso = ""
            for iso in range (1, len(ftr)-1): 
                other_iso = other_iso + "iso "+ str(iso) +" - mz:" + str(ftr[iso][0]) +"; RT Peak:"+str(ftr[iso][1][0]) +"; RT start:"+str(ftr[iso][1][1])+"; RT end:"+str(ftr[iso][1][2])+"|"
            item.append(other_iso)
            feature_list.append(item)

    print(len(feature_list))

    feature_filename = '/home/fzohora/bsi/feature_list_csv/'+dataname[test_index]+'_feature_list_detailed.csv'
    f=open(feature_filename, 'w', encoding='UTF8', newline='') #'/cluster/home/t116508uhn/test.csv'
    writer = csv.writer(f)
    # write the header
    writer.writerow(['mono-isotopic m/z', 'max intensity RT', 'charge', 'start RT', 'end RT', 'number of isotopes','total intensity', 'feature score','last layer softmax output','all isotopes'])
    writer.writerows(feature_list)
    f.close()



