# nohup python -u IsoGrouping_reportFeature_ev2r4.py recordpath scanpath modelpath filename resultpath gpu_index > output.log &
'''nohup python -u IsoGrouping_reportFeature_ev2r4.py /data/anne/dilution_series_syn_pep/hash_record/ /data/anne/dilution_series_syn_pep/scanned_result/  
/data/anne/pointIso/3D_model/  130124_dilA_1_01 /data/anne/pointIso/3D_result/ 0 > output.log & '''


from __future__ import print_function, division
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
import pickle
import math
import copy
#from time import time
import bisect
#import sys
#import scipy.misc
#import csv
from collections import defaultdict
import os
from sklearn import metrics
recordpath=
scanpath=
modelpath=
filename=
resultpath= 
gpu_index=

gpu=gpu_index
os.environ["CUDA_VISIBLE_DEVICES"]=gpu
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

truncated_backprop_length = 5
#print('%s, learn rate %g'%(log_no,learn_rate))
total_frames_hor=truncated_backprop_length
total_hops_horizontal= total_frames_hor//truncated_backprop_length 
num_class=total_frames_hor # number of isotopes to report
drop_out_k=0.5
RT_window=15
mz_window=frame_width=3 #7
new_mz_unit=0.001
RT_unit=0.01
new_mz_resolution=3
mz_resolution=4
mz_unit=0.0001
learning_rate= .07 #0.08 -- gave best so far
num_epochs= 200

log_no='deepIsoV2_isoGrouping_auc_exact_v2_r4' 
batch_size=128
print('%s, learn rate %g'%(log_no,learning_rate))

take_zero=1
activation_func=2
#val_start=100
total_frames_hor=truncated_backprop_length
total_hops_horizontal= total_frames_hor//truncated_backprop_length 
drop_out_k=0.5


#######################################################################
num_class=total_frames_hor
num_neurons= num_class #mz_window*RT_window
#state_size = 4 #
fc_size = 8 #one
learning_rate= 0.05 #-- gave best so far
def weight_variable(shape, variable_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=variable_name)

def bias_variable(shape, variable_name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=variable_name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv2d_seq(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, fc_size, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.device('/gpu:'+ gpu):  #sys.argv[1]): # 
    batchX_placeholder = tf.placeholder(tf.float32, [None, RT_window, mz_window*truncated_backprop_length]) #image block to consider for one run of training by back propagation
    batchAUC_placeholder = tf.placeholder(tf.float32, [None,total_frames_hor])
    keep_prob = tf.placeholder(tf.float32)
    keep_prob_seq = tf.placeholder(tf.float32)
    learn_rate=tf.placeholder(tf.float32) 
    # each image is 15 x 7
    W_conv0 = weight_variable([3, 2 , 1, 8], 'W_conv0')#v10: 
    b_conv0 = bias_variable([8], 'b_conv0') #15-3+1=13,3-2+1=2
    # pool - 7, 1

    W_conv1 = weight_variable([3, 1 , 8, 16], 'W_conv1')#v10: 7-3+1, 1-1+1= 5,1 
    b_conv1 = bias_variable([16], 'b_conv1') #for each of feature maps
    # pool - 3, 1


    W_auc=weight_variable([1 , 8], 'W_auc') #
    b_auc = bias_variable([8], 'b_auc')

    #2 x 1

    W_fc1 = weight_variable([3 * 1 * 16 + 8 , 16], 'W_fc1') # + 4
    b_fc1 = bias_variable([16], 'b_fc1')



    W_out = weight_variable([16, fc_size], 'W_out') #8
    b_out = bias_variable([fc_size], 'b_out')

    W_seq_conv_0 = weight_variable([1, fc_size*2 ,  1, 8], 'W_seq_conv_0') #32 done
    b_seq_conv_0 = bias_variable([8], 'b_seq_conv_0')  #1,4


    W_fc2 = weight_variable([1*4*8, 128], 'W_fc2') # 8+8+8+8+8=4
    b_fc2 = bias_variable([128], 'b_fc2')
    
    W_fc3 = weight_variable([128, 64], 'W_fc3') 
    b_fc3 = bias_variable([64], 'b_fc3')    
#    
    W_z=weight_variable([64 , 1], 'W_z') # scaling neuron
    b_z = bias_variable([1], 'b_z')


    W_fc4 = weight_variable([64+1, 32], 'W_fc4') 
    b_fc4 = bias_variable([32], 'b_fc4')      
    
    W2 = tf.Variable(np.random.rand(32, num_class),dtype=tf.float32) #final output
    b2 = tf.Variable(np.zeros((1,num_class)), dtype=tf.float32) #final output

    #param_loader = tf.train.Saver({'W_conv0': W_conv0, 'W_conv1': W_conv1, 'W_conv2': W_conv2, 'W_conv3': W_conv3, 'W_fc1':W_fc1, 'W_out':W_out, 'b_conv0':b_conv0, 'b_conv1':b_conv1, 'b_conv2':b_conv2, 'b_conv3':b_conv3, 'b_fc1':b_fc1, 'b_out':b_out})

    batchY_placeholder = tf.placeholder(tf.float32, [None, num_class])
    batchZ_placeholder = tf.placeholder(tf.float32, [None, 1])
    
#    init_state = tf.placeholder(tf.float32, [None, state_size])

    # Forward pass
#    current_state = init_state
    states_series = []
    for j in range (0, truncated_backprop_length):
        ##############################
        x_image = tf.reshape(batchX_placeholder[:, : , mz_window*j : mz_window* (j+1)], [-1, RT_window, mz_window, 1]) #flatten to 2d: row: RT, column: mz

        h_conv0 = tf.tanh(conv2d(x_image, W_conv0) + b_conv0) # now the input is: (15-8+1) x (211-22+1) x 16 = 8 x 190 x 16           
        h_pool0 = max_pool_2x2(h_conv0)    

        h_conv1 = tf.tanh(conv2d(h_pool0, W_conv1) + b_conv1) # now the input is: (8-4+1) x (190-6+1) x 16 = 5 x 185 x 16
        h_pool1 = max_pool_2x2(h_conv1)

#        h_conv2 = tf.tanh(conv2d(h_pool1, W_conv2) + b_conv2) # now the input is: (8-4+1) x (190-6+1) x 16 = 5 x 185 x 16
    #    h_pool2 = max_pool_2x2(h_conv2)


#        h_conv3 = tf.tanh(conv2d(h_conv2, W_conv3) + b_conv3) # now the input is: (5-3+1) x (185-4+1) x 8 = 3  x 182  x 8
        h_conv2_flat = tf.reshape(h_pool1, [-1, 3 * 1  * 16])
    #    h_conv3_flat_drop = tf.nn.dropout(h_conv3_flat, keep_prob)

        h_auc = tf.tanh(tf.matmul(batchAUC_placeholder[:, j:j+1], W_auc) + b_auc)
        h_conv_auc=tf.concat([h_conv2_flat, h_auc], 1) 
                
        h_fc1 = tf.tanh(tf.matmul(h_conv_auc, W_fc1) + b_fc1)

#        h_fc1 = tf.tanh(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
        h_fc1_dropout=tf.nn.dropout(h_fc1, keep_prob)
#        h_fc1_dropout_z = tf.concat([h_fc1_dropout, batchZ_placeholder], 1)
        frame_out= tf.tanh(tf.matmul(h_fc1_dropout, W_out) + b_out) # finally this will connect with RNN
        ##############################
        current_FC  = frame_out #h_fc2 #tf.nn.dropout(h_fc2, keep_prob) #  [batch_size, fc_size])
        states_series.append(current_FC) #next_state
        
        
        
#        current_state = cand_next_state #next_state
    state_concatenated=states_series[0]
    for j in range (1, truncated_backprop_length):
        state_concatenated=tf.concat([state_concatenated, states_series[j]], 1) # row --> batch
    
    state_concatenated=tf.reshape(state_concatenated, [-1, 1, fc_size*total_frames_hor, 1]) 
    
    h_seq_conv_0=tf.tanh(conv2d_seq(state_concatenated, W_seq_conv_0) + b_seq_conv_0)   
#    h_seq_conv_1=tf.tanh(conv2d(h_seq_conv_0, W_seq_conv_1) + b_seq_conv_1)       
    h_seq_conv_1_flat=tf.reshape(h_seq_conv_0, [-1, 1*4 * 8])

#    h_fc2 = tf.tanh(tf.matmul(state_concatenated, W_fc2) + b_fc2)

    h_fc2 = tf.tanh(tf.matmul(h_seq_conv_1_flat, W_fc2) + b_fc2)
    h_fc3 = tf.tanh(tf.matmul(tf.nn.dropout(h_fc2, keep_prob_seq), W_fc3) + b_fc3)  
    h_fc3_drop=tf.nn.dropout(h_fc3, keep_prob)
    
    h_z = tf.tanh(tf.matmul(h_fc3_drop, W_z) + b_z)
  
    h_fc4 = tf.tanh(tf.matmul(tf.concat([h_fc3_drop, tf.multiply(h_z, batchZ_placeholder)], 1), W_fc4) + b_fc4)    
    
    logit = tf.matmul(tf.nn.dropout(h_fc4, keep_prob_seq), W2) + b2 

    prediction = tf.argmax(tf.nn.softmax(logit), 1)

    loss=tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=batchY_placeholder)
        
    total_loss = tf.reduce_mean(loss)

    train_step = tf.contrib.opt.NadamOptimizer(learn_rate).minimize(total_loss) # tf.train.AdamOptimizer(learn_rate).minimize(total_loss) #tf.train.AdagradOptimizer(learn_rate).minimize(total_loss) # # tf.contrib.opt.NadamOptimizer(learn_rate).minimize(total_loss) #tf.train.AdagradOptimizer(learn_rate).minimize(total_loss) #

config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
#config=tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver = tf.train.Saver(max_to_keep=7)
#sess.run(tf.global_variables_initializer())
#saver.save(sess, modelpath+'init-model_'+log_no+'.ckpt')
#######################################
#saver.restore(sess, modelpath+'trained-model_'+log_no+'_best_loss.ckpt')
saver.restore(sess, modelpath+'trained-model_'+log_no+'_best.ckpt')
##########################

#################### report feature module#####################################
print(filename) 
f=open(scanpath+filename+'_pointIso_clusters', 'rb') # 98.35    
isotope_cluster, max_num_iso,total_clusters=pickle.load(f)
f.close()
print('making cluster list')
mz_list=sorted(isotope_cluster.keys())
cluster_list=[]
for i in range (0, len(mz_list)): #len(mz_list)
    ftr_list=isotope_cluster[mz_list[i]]
    for j in range (0, len(ftr_list)):
        ftr=ftr_list[j]        
        cluster_list.append(ftr)




f=open(recordpath+filename+'_ms1_record_mz5', 'rb')
RT_mz_I_dict, sorted_mz_list, maxI=pickle.load(f)
f.close()   
print('done!')



print('data restore done')
#scan ms1_block and record the cnn outputs in list_dict[z]: hash table based on m/z
#for each m/z

RT_list=sorted(RT_mz_I_dict.keys())
max_RT=RT_list[len(RT_list)-1]
min_RT=10    
RT_index_array=dict()
for i in range (0, len(RT_list)):
    RT_index_array[round(RT_list[i], 2)]=i

#    f=open(datapath+'feature_list/pointCloud_'+dataname[val_index]+'_RT_index_new_mz5', 'rb')
#    RT_index=pickle.load(f)
#    f.close()  
RT_index= RT_mz_I_dict       

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

rt_search_index=0
while(RT_list[rt_search_index]<min_RT):
    rt_search_index=rt_search_index+1
print('preprocess done')

############## match feature module ###########################################


########################################################


total_clusters=len(cluster_list)
print('making done %d'%(total_clusters))
cluster_length=np.zeros((total_clusters))
count=0
for i in range (0, len(cluster_list)): #len(mz_list)
    cluster_length[count]=len(cluster_list[i])-1                   
    count=count+1

start_iso=np.zeros((total_clusters))
current_iso=np.zeros((total_clusters))
feature_table=defaultdict(list)
batch_size_val=100000 #total_clusters
total_batch_val=math.ceil(total_clusters/batch_size_val)
DEBUG=0
total_feature=0
cluster_count=0
case_count=0
for batch_idx in range (0, total_batch_val):
    print(batch_idx)
    start_cluster=batch_idx*batch_size_val
    end_cluster=min(start_cluster+batch_size_val, total_clusters)
    cluster_count=cluster_count+end_cluster-start_cluster
    cluster_left=1
#                    _current_state = np.zeros((batch_size_val, state_size))    
    while(cluster_left):
        # for each cluster, assign frames from start_iso to total_frames_hor, to the cut_block
        # make the batch
        count=0
        batch_ms1_val=np.zeros((batch_size_val, RT_window,frame_width*total_frames_hor))  
        batch_auc_val=np.zeros((batch_size_val, total_frames_hor))   
        ftr_z=np.zeros((batch_size_val, 1))
        for c in range (start_cluster, end_cluster):
            if cluster_length[c]<=0:
                continue

################################################
            feature=cluster_list[c]
            RT_peak=round(feature[int(current_iso[c])][1][0], 2)        
            # 7 step before, peak, 7 step after
#                            count=count+1                          
            RT_s=max(RT_index_array[RT_peak]-7, 0)
            RT_e=min(RT_s+RT_window, len(RT_list)) #ex
            charge=int(feature[len(feature)-1][0])   
#                            num_isotopes=int(feature[len(feature)-1][1])
#                            feature_width=min(total_frames_hor, num_isotopes)
            mz_point=round(feature[int(current_iso[c])][0], new_mz_resolution)   #round(peptide_feature[ftr, 0], new_mz_resolution)
            mz_tolerance=round((mz_point*2.0)/10**6, mz_resolution)        

            auc_list=[]
            auc_cal=[]
            cut_block=np.zeros((total_frames_hor, RT_window, mz_window))
            for iso in range (0, total_frames_hor):
                mz_s=round(mz_point-mz_tolerance-pow(0.1, new_mz_resolution)+pow(0.1, new_mz_resolution+1)*5, mz_resolution)
                mz_e=round(mz_point+mz_tolerance+pow(0.1, new_mz_resolution+1)*4, mz_resolution)
                sum_area=0
                mz_dict=[]
                rt_row=0
                auc_iso_list=[]
                for RT_idx in range (RT_s,RT_e):
                    mz_dict.append(defaultdict(list))  
                    auc_iso_list.append([]) 
                    if RT_idx<0 or RT_idx>(len(RT_list)-1):
                        rt_row=rt_row+1
                        continue

                    mz_value=mz_s

                    find_mz_idx_start= bisect.bisect_left(sorted_mz_list[RT_idx], mz_value)
                    if len(sorted_mz_list[RT_idx])==find_mz_idx_start or round(sorted_mz_list[RT_idx][find_mz_idx_start], mz_resolution)>mz_e:
                        rt_row=rt_row+1
                        continue
                    mz_value=round(sorted_mz_list[RT_idx][find_mz_idx_start] , mz_resolution)                        

                    datapoint=RT_index[round(RT_list[RT_idx], 2)][mz_value]    
                    intensity=((datapoint[0]-0)/(maxI-0))*255 #round(, 2) # scale it to the grey value
                    mz_dict[rt_row][round(mz_value, 2)].append(intensity)
                    auc_iso_list[rt_row].append(datapoint[0])
                    next_mz_idx=int(datapoint[1])+1         
                    mz_value= round(sorted_mz_list[RT_idx][next_mz_idx], mz_resolution)
                    while mz_value<=mz_e:                    
                        datapoint=RT_index[round(RT_list[RT_idx], 2)][mz_value]    
                        intensity=((datapoint[0]-0)/(maxI-0))*255 #round(, 2) # scale it to the grey value
                        mz_dict[rt_row][round(mz_value, 2)].append(intensity)
                        next_mz_idx=int(datapoint[1])+1         
                        mz_value= round(sorted_mz_list[RT_idx][next_mz_idx], mz_resolution)
                        auc_iso_list[rt_row].append(datapoint[0])
                    # after this is done, we have the list of mz, for this RT
                    rt_row=rt_row+1

                RT_idx=RT_s
                stripe_x=np.zeros(rt_row)
                stripe_y=np.zeros(rt_row)
                for rt_row in range (0, len(mz_dict)):
                    mz_keys=sorted(mz_dict[rt_row].keys())
                    if len(mz_keys)==0:
                        RT_idx=RT_idx+1
                        continue
                    stripe_x[rt_row]=RT_list[RT_idx]
                    if len(auc_iso_list[rt_row])>0:
                        stripe_y[rt_row]=np.max(auc_iso_list[rt_row])

                    for mz_value in mz_keys:
                        mz_poz=int(round((mz_value-mz_s)/0.01, 2))                            
                        cut_block[iso, rt_row, mz_poz]=max(mz_dict[rt_row][mz_value])
                        sum_area=sum_area+cut_block[iso, rt_row, mz_poz]
                    RT_idx=RT_idx+1
                try:
                    this_auc=metrics.auc(stripe_x, stripe_y) #np.sum(stripe_y)#
                    auc_cal.append(this_auc)        
                except:
                    auc_cal.append(0)

                auc_list.append(sum_area)            
                mz_point=round(mz_point+isotope_gap[charge], new_mz_resolution)
    ###########################                        
            for fr in range (0, total_frames_hor):
                batch_ms1_val[count, :, fr*frame_width:(fr+1)*frame_width]=cut_block[fr, :, :]    
                batch_auc_val[count, fr]=auc_cal[fr] #auc_list[fr]    
            ftr_z[count, 0]=charge
            count=count+1
        # one batch made

        if count==0:
            break

        # now run the model
        current_batch_size=count
        print('current_batch_size %d'%current_batch_size)

#                        _current_state = np.zeros((current_batch_size, state_size))                            
        batchX = np.round(batch_ms1_val[0:current_batch_size, :, :], 2) 
        _prediction_batch= sess.run(
            prediction,
            feed_dict={
                batchX_placeholder:batchX,
                batchZ_placeholder:ftr_z[0:current_batch_size] ,
                batchAUC_placeholder:batch_auc_val[0:current_batch_size, :],
#                                init_state:_current_state, 
                keep_prob:1.0, 
                keep_prob_seq:1.0
            })          

        # now ck the prediction for each cluster and set the start iso for next run accordingly
        count=-1
        for c in range (start_cluster, end_cluster):
            if cluster_length[c]<=0:
                continue
            count=count+1            
            _prediction= _prediction_batch[count] 

            if _prediction>0:
                if _prediction==total_frames_hor-1 and cluster_length[c]>total_frames_hor:
                    current_iso[c]=current_iso[c]+_prediction
                    cluster_length[c]=cluster_length[c]-_prediction
#                                    _current_state = np.zeros((batch_size_val, state_size))
                else:
                    if _prediction<cluster_length[c]:
                        end_iso=int(current_iso[c]+_prediction+1) #(ex)
                    else:
                        end_iso=int(current_iso[c]+cluster_length[c]) #(ex)

                    new_ftr=[]
                    ftr=cluster_list[c]
                    for isotope in range (int(start_iso[c]) ,end_iso):
                        new_ftr.append(ftr[isotope])

                    new_ftr.append([ftr[len(ftr)-1][0]]) # charge
                    feature_table[round(new_ftr[0][0], 2)].append(new_ftr)
                    total_feature=total_feature+1                    

                    start_iso[c]=end_iso                        
                    current_iso[c]=start_iso[c]
                    cluster_length[c]=cluster_length[c]-_prediction-1
#                                    _current_state = np.zeros((batch_size_val, state_size))
            else:
                end_iso=int(current_iso[c]+_prediction+1)
                if current_iso[c]!=start_iso[c]: # it was continuing 
                    new_ftr=[]
                    ftr=cluster_list[c]
                    for isotope in range (int(start_iso[c]),end_iso):
                        new_ftr.append(ftr[isotope])

                    new_ftr.append([ftr[len(ftr)-1][0]]) # charge
                    feature_table[round(new_ftr[0][0], 2)].append(new_ftr)
                    total_feature=total_feature+1                    

                start_iso[c]=end_iso                        
                current_iso[c]=start_iso[c]
                cluster_length[c]=cluster_length[c]-_prediction-1
#                                _current_state = np.zeros((batch_size_val, state_size))

############## merge features apart from each other with just 0.004 m/z and 0.01 RT ###########################################
key_list=sorted(feature_table.keys())
count=0
RT_tol=.01 #4
tolerance=0.004
for mz in key_list:

    ftr_list=sorted(feature_table[mz])
    new_ftr_list=[]
    for k in range (0, len(ftr_list)):
        if len(ftr_list[k])==0:
            continue
        ftr_pred=ftr_list[k]
#            tolerance=round((ftr_pred[0][0]*ppm)/10**6, new_mz_resolution)
        z_pred=ftr_pred[len(ftr_pred)-1][0]
        for j in range (k+1, len(ftr_list)):
            if len(ftr_list[j])==0:
                continue                
            if ftr_list[j][0][0]>round(ftr_pred[0][0]+tolerance, new_mz_resolution):
                break
            mono_mz_pred=ftr_pred[0][0]
#                tolerance=round((mono_mz_pred*ppm)/10**6, new_mz_resolution)        
            ftr=ftr_list[j]
            z_ftr=ftr[len(ftr)-1][0]
            mono_mz=ftr[0][0]
            mono_mz_pred=ftr_pred[0][0]
            if mono_mz<=round(mono_mz_pred+tolerance, new_mz_resolution) and z_pred==z_ftr:
                a=ftr_pred[0][1][1]
                b=ftr_pred[0][1][2]
                c=ftr[0][1][1]
                d=ftr[0][1][2]
                if a<=d and b>=c:
                    RT_peak_pred=ftr_pred[0][1][0]
                    RT_peak=ftr[0][1][0]
                    if np.abs(RT_peak_pred-RT_peak)<=RT_tol: #4
                        # merge 
                        min_isotope=min(len(ftr)-1,len(ftr_pred)-1)
                        new_ftr=[]
                        for iso_index in range (0, min_isotope):
                            a=min(ftr_pred[iso_index][1][1], ftr[iso_index][1][1]) #start
                            b=max(ftr_pred[iso_index][1][2], ftr[iso_index][1][2]) #end

                            if ftr_pred[iso_index][1][3]<ftr[iso_index][1][3]: 
                                mz_new_ftr=ftr[iso_index][0]
                                peak_rt=RT_peak
                            else:
                                mz_new_ftr=ftr_pred[iso_index][0]
                                peak_rt=RT_peak_pred



                            new_ftr.append([mz_new_ftr, [peak_rt, a, b, ftr_pred[iso_index][1][3]+ftr[iso_index][1][3], ftr_pred[iso_index][1][4]+ftr[iso_index][1][4]]])

                        if len(ftr)-1 >min_isotope:
                            for iso_idx in range (iso_index+1, len(ftr)-1):  
                                new_ftr.append(ftr[iso_idx])
                        elif len(ftr_pred)-1 >min_isotope:
                            for iso_idx in range (iso_index+1, len(ftr_pred)-1):  
                                new_ftr.append(ftr_pred[iso_idx])
                        new_ftr.append([z_ftr])
                        ftr_pred=copy.deepcopy(new_ftr)
                        # replace the
#                            if ftr_pred[0][1][3]<ftr[0][1][3]: 
#                                ftr_pred=copy.deepcopy(ftr)

                        ftr_list[j]=[]
#        ftr_pred[0][1][3]=sum_intensity
        new_ftr_list.append(ftr_pred)                
    feature_table[mz]=new_ftr_list
    count=count+len(new_ftr_list)

###############        
key_list=sorted(feature_table.keys())
count=0
new_feature_table=defaultdict(list)
for i in range (0, len(key_list)) :
    ftr_list_pred=sorted(feature_table[key_list[i]])
    for k in range (0, len(ftr_list_pred)):
        if len(ftr_list_pred[k])==0:
            continue
        ftr_pred=ftr_list_pred[k]
#            tolerance=round((ftr_pred[0][0]*ppm)/10**6, new_mz_resolution)
        z_pred=ftr_pred[len(ftr_pred)-1][0]
        not_finished=1
        p=i+1
        while(not_finished and p<len(key_list)):
            ftr_list=sorted(feature_table[key_list[p]])
            j=0
            while(j<len(ftr_list) and len(ftr_list[j])==0):
                j=j+1
            if j==len(ftr_list):
                p=p+1
                continue
            if np.abs(round(ftr_list[j][0][0]-ftr_pred[0][0], new_mz_resolution))>tolerance:
                break   
            for j in range (0, len(ftr_list)):
                if len(ftr_list[j])==0:
                    continue
                if ftr_list[j][0][0]>round(ftr_pred[0][0]+tolerance, new_mz_resolution):
                    break   

                mono_mz_pred=ftr_pred[0][0]
#                    tolerance=round((mono_mz_pred*ppm)/10**6, new_mz_resolution)        
                ftr=ftr_list[j]
                z_ftr=ftr[len(ftr)-1][0]
                mono_mz=ftr[0][0]
                mono_mz_pred=ftr_pred[0][0]
                if z_pred==z_ftr and round(np.abs(mono_mz-mono_mz_pred), new_mz_resolution)<=tolerance:
                    a=ftr_pred[0][1][1]
                    b=ftr_pred[0][1][2]
                    c=ftr[0][1][1]
                    d=ftr[0][1][2]
                    if a<=d and b>=c:
                        RT_peak_pred=ftr_pred[0][1][0]
                        RT_peak=ftr[0][1][0]
                        if np.abs(RT_peak_pred-RT_peak)<=RT_tol: #4:
                        # merge 
                            min_isotope=min(len(ftr)-1,len(ftr_pred)-1)
                            new_ftr=[]
                            for iso_index in range (0, min_isotope):
                                a=min(ftr_pred[iso_index][1][1], ftr[iso_index][1][1]) #start
                                b=max(ftr_pred[iso_index][1][2], ftr[iso_index][1][2]) #end

                                if ftr_pred[iso_index][1][3]<ftr[iso_index][1][3]: 
                                    mz_new_ftr=ftr[iso_index][0]
                                    peak_rt=RT_peak
                                else:
                                    mz_new_ftr=ftr_pred[iso_index][0]
                                    peak_rt=RT_peak_pred



                                new_ftr.append([mz_new_ftr, [peak_rt, a, b, ftr_pred[iso_index][1][3]+ftr[iso_index][1][3], ftr_pred[iso_index][1][4]+ftr[iso_index][1][4]]])

                            if len(ftr)-1 >min_isotope:
                                for iso_idx in range (iso_index+1, len(ftr)-1):  
                                    new_ftr.append(ftr[iso_idx])
                            elif len(ftr_pred)-1 >min_isotope:
                                for iso_idx in range (iso_index+1, len(ftr_pred)-1):  
                                    new_ftr.append(ftr_pred[iso_idx])
                            new_ftr.append([z_ftr])
                            ftr_pred=copy.deepcopy(new_ftr)                            

                            # replace the
#                                if ftr_pred[0][1][3]<ftr[0][1][3]: 
#                                    ftr_pred=copy.deepcopy(ftr)
                            feature_table[key_list[p]][j]=[]
                            break
            p=p+1        

        new_feature_table[round(ftr_pred[0][0], 2)].append(ftr_pred)
        count=count+1

feature_table=copy.deepcopy(new_feature_table)

key_list=feature_table.keys()
count=0
for mz in key_list:
    ftr_list=sorted(feature_table[mz])
    for k in range (0, len(ftr_list)):
        ftr=ftr_list[k]
        print("monoisotope m/z=%g, RT=%g (%g to %g), charge=%d, number of isotopes=%d"%(ftr[0][0],ftr[0][1][0], ftr[0][1][1],ftr[0][1][2], int(ftr[len(ftr)-1][0]), len(ftr)-1))
        count=count+1
print("total features %d "%count)


f=open(resultpath+filename+'_featureTable','wb') 
pickle.dump(feature_table, f, protocol=2)
f.close() 

    

