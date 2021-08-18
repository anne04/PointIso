# PointIso

Order of running the files:
 
2. $ nohup python -u read_pointCloud.py [filepath] [topath]
   Here filepath is the location of .ms1 file and topath is the location where the hash table having the triplets (RT, m/z, I) of datapoints is saved. 
   A sample command should look like below (here the outputs are printed in a output.log file):
   $ nohup python -u read_pointCloud.py /data/anne/dilution_series_syn_pep/130124_dilA_1_01.ms1 /data/anne/dilution_series_syn_pep/130124_dilA_1_01_ms1_record_mz5 >      output.log &
   
3. prepare_RT_index.py

4. isoDetecting_scan_MS1_pointNet.py
5. makeCluster.py
6. IsoGrouping_reportFeature_ev2r4.py
