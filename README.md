The *.raw files are downloaded from ProteomeXchange database and ProteoWizerd 3.0.1817 is used to convert the raw file into .ms1 format. After that we have to run the uploaded python scripts as explained below:

Syntax for running the python scripts from linux (Ubuntu) terminal is provided below in the order of execution: 
-----------------------------------------------------------------------------------------------------------------------------------------------------------
1. read_pointCloud.py: This script read the *.ms1 file from location 'filepath' having name 'sample_name' and convert the LC-MS map in that file to a hash table holding the datapoint triplets (RT, m/z, I). The hash table is saved at location 'topath'. The syntax and example of related command to run in terminal is provided below:  
$ nohup python -u read_pointCloud.py [filepath] [topath] [sample_name] > output.log &  
Example:  
$ nohup python -u read_pointCloud.py /data/anne/dilution_series_syn_pep/ /data/anne/dilution_series_syn_pep/hash_record/ 130124_dilA_1_01 > output.log &
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
2. isoDetecting_scan_MS1_pointNet.py: This script loads the LC-MS map of sample 'sample_name' saved at location 'recordpath' and scans it using the trained model saved at 'modelpath' location. The scanned result is saved at location 'scanpath'. GPU index is to be mentioned as parameter. Also, this script should scan multiple segments of the LC-MS map in parallel. That is why, we provide another parameter 'start_mz', so that IsoDetecting module starts scanning at that particular m/z value and covers next 200 m/z. If LC-MS map is ranged from 400 to 2000 m/z, then the 'start_mz' of 7 parallel scripts should be: 400, 600, 800, 1000, 1200, 1400, 1800. The syntax and example of related command to run in terminal is provided below:  
$nohup python -u isoDetecting_scan_MS1_pointNet.py [recordpath] [sample_name] [modelpath] [gpu_index] [start_mz] [scanpath] > output.log &  
Example:  
$nohup python -u isoDetecting_scan_MS1_pointNet.py /data/anne/dilution_series_syn_pep/hash_record/ 130124_dilA_1_01 /data/anne/pointIso/3D_model/ 0 400 /data/anne/dilution_series_syn_pep/scanned_result/ > output.log & 
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
3. makeCluster.py: This script clusters the equidistant isotopes of same charge together. This combines the scanned result generated by multiple parallel scripts as mentioned above. The cluster list is saved at the location 'scanpath'. The syntax and example of related command to run in terminal is provided below:  
$nohup python -u makeCluster.py [recordpath] [filename] [scanpath] > output.log &    
Example:  
$nohup python -u makeCluster.py /data/anne/dilution_series_syn_pep/hash_record/ 130124_dilA_1_01 /data/anne/dilution_series_syn_pep/scanned_result/ > output.log &
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
4. IsoGrouping_reportFeature_ev2r4.py: This script process the cluster list generated in previous step by IsoGrouping module and prints the feature table. Feature table is saved at location 'resultpath'. The syntax and example of related command to run in terminal is provided below:  
$nohup python -u IsoGrouping_reportFeature_ev2r4.py [recordpath] [scanpath] [modelpath] [filename] [resultpath] [gpu_index] > output.log &  
Example:  
$ nohup python -u IsoGrouping_reportFeature_ev2r4.py /data/anne/dilution_series_syn_pep/hash_record/ /data/anne/dilution_series_syn_pep/scanned_result/  /data/anne/pointIso/3D_model/  130124_dilA_1_01 /data/anne/pointIso/3D_result/ 0 > output.log & 
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
