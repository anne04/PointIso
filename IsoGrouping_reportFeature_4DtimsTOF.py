# nohup python -u  IsoGrouping_reportFeature_4DtimsTOF.py [recordpath] [sample_name] [modelpath] [gpu_index] [scanpath] > output.log &
''' nohup python -u IsoGrouping_reportFeature_4DtimsTOF.py '/data/anne/timsTOF/hash_records/' 'A1_1_2042' /data/anne/pointIso/4D_model/ 0 
/data/anne/timsTOF/scanned_result/ > output.log & '''


from __future__ import print_function, division
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import pickle
import math
import bisect
import gc
from collections import defaultdict
import os
from sklearn import metrics
import gzip


recordpath=sys.argv[1]
sample_name=sys.argv[2]
modelpath=sys.argv[3]
gpu_index=sys.argv[4]
scanpath=sys.argv[5]
gpu=gpu_index

os.environ["CUDA_VISIBLE_DEVICES"]=gpu

max_part=12
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
mz_resolution=5
mz_unit=0.0001
learning_rate= .07 #0.08 -- gave best so far
num_epochs= 200

log_no='deepIsoV2_isoGrouping_v3_timsTOF_r2'
#log_no='deepIsoV2_isoGrouping_auc_exact_v2_r6b' #r1 is w/o fp, r2 wat w/o TB 
batch_size=128
print('%s, learn rate %g'%(log_no,learning_rate))
#log_no_old='deepIsoV2_isoGrouping_auc_exact_v3_r1' #r1 is w/o fp, r2 wat w/o TB 

take_zero=1
#log_no_old='deepIsoV2_isoGrouping_auc_v2'
activation_func=2
#val_start=100
total_frames_hor=truncated_backprop_length
total_hops_horizontal= total_frames_hor//truncated_backprop_length 
drop_out_k=0.5

validation_index=9

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
    
    
    W_conv0=[]
    b_conv0=[]
    W_conv1=[]
    b_conv1=[]
    W_auc=[]
    b_auc=[]
    W_fc1=[]
    b_fc1=[]
    W_out=[]
    b_out=[]
    
    for i in range (0, truncated_backprop_length): 
        W_conv0.append(weight_variable([3, 2 , 1, 8], 'W_conv0'+str(i))) #v10: 
        b_conv0.append(bias_variable([8], 'b_conv0'+str(i))) #15-3+1=13,3-2+1=2
        # pool - 7, 1

        W_conv1.append(weight_variable([3, 1 , 8, 16], 'W_conv1'+str(i))) #v10: 7-3+1, 1-1+1= 5,1 
        b_conv1.append(bias_variable([16], 'b_conv1'+str(i))) #for each of feature maps
        # pool - 3, 1


        W_auc.append(weight_variable([1 , 8], 'W_auc'+str(i))) #
        b_auc.append(bias_variable([8], 'b_auc'+str(i)))


        W_fc1.append(weight_variable([3 * 1 * 16 + 8 , 16], 'W_fc1'+str(i))) # + 4
        b_fc1.append(bias_variable([16], 'b_fc1'+str(i)))

        W_out.append(weight_variable([16, fc_size], 'W_out'+str(i))) #8
        b_out.append(bias_variable([fc_size], 'b_out'+str(i))) 

#-------------------------------------------------------------------------------------------------
    W_seq_conv_0 = weight_variable([1, fc_size*2 ,  1, 8], 'W_seq_conv_0') #32 done
    b_seq_conv_0 = bias_variable([8], 'b_seq_conv_0')  #1,4

#    W_seq_conv_1 = weight_variable([1,  3, 8, 8], 'W_seq_conv_1') #32 done - 1,2
#    b_seq_conv_1 = bias_variable([8], 'b_seq_conv_1')
#


#    W_fc2 = weight_variable([fc_size*total_frames_hor, 64], 'W_fc2') #32 done
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
    batchY_placeholder = tf.placeholder(tf.int32, [None])
    batchZ_placeholder = tf.placeholder(tf.float32, [None, 1])
    
#    init_state = tf.placeholder(tf.float32, [None, state_size])

    # Forward pass
#    current_state = init_state
    states_series = []
    for j in range (0, truncated_backprop_length):
        ##############################
        x_image = tf.reshape(batchX_placeholder[:, : , mz_window*j : mz_window* (j+1)], [-1, RT_window, mz_window, 1]) #flatten to 2d: row: RT, column: mz

        h_conv0 = tf.tanh(conv2d(x_image, W_conv0[j]) + b_conv0[j]) # now the input is: (15-8+1) x (211-22+1) x 16 = 8 x 190 x 16           
        h_pool0 = max_pool_2x2(h_conv0)    

        h_conv1 = tf.tanh(conv2d(h_pool0, W_conv1[j]) + b_conv1[j]) # now the input is: (8-4+1) x (190-6+1) x 16 = 5 x 185 x 16
        h_pool1 = max_pool_2x2(h_conv1)

        h_conv2_flat = tf.reshape(h_pool1, [-1, 3 * 1  * 16])


        h_auc = tf.tanh(tf.matmul(batchAUC_placeholder[:, j:j+1], W_auc[j]) + b_auc[j])
        h_conv_auc=tf.concat([h_conv2_flat, h_auc], 1) 
                
        h_fc1 = tf.tanh(tf.matmul(h_conv_auc, W_fc1[j]) + b_fc1[j])

        h_fc1_dropout=tf.nn.dropout(h_fc1, keep_prob)
        
        frame_out= tf.tanh(tf.matmul(h_fc1_dropout, W_out[j]) + b_out[j]) # finally this will connect with RNN
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
    pred_score = tf.nn.softmax(logit) #tf.reduce_max(tf.nn.softmax(logit),axis=1)
    
    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=batchY_placeholder)
        
    total_loss = tf.reduce_mean(loss)

    train_step = tf.train.AdamOptimizer(learn_rate).minimize(total_loss) #tf.contrib.opt.NadamOptimizer(learn_rate).minimize(total_loss) # tf.train.AdagradOptimizer(learn_rate).minimize(total_loss) # # tf.contrib.opt.NadamOptimizer(learn_rate).minimize(total_loss) #tf.train.AdagradOptimizer(learn_rate).minimize(total_loss) #

config=tf.ConfigProto(allow_soft_placement=True) #, log_device_placement=True)
#config=tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver = tf.train.Saver(max_to_keep=7)
#sess.run(tf.global_variables_initializer())
saver.restore(sess, modelpath+'trained-model_'+log_no+'_best.ckpt')
#saver.restore(sess, modelpath+'trained-model_'+log_no+'_best_save_1.ckpt')


##########################

#################### report feature module#####################################

print('trying to load ms1 record')
RT_index=dict()
for part in range (1, max_part+1):
    print(part)
    f=gzip.open(recordpath+sample_name+'_RT_index_part'+str(part), 'rb')
    RT_index_temp=pickle.load(f)
    f.close()
    RT_index.update(RT_index_temp)
    print(gc.collect())
print('read done')

f=open(recordpath+'pointCloud_'+sample_name+'_maxI', 'rb')
maxI=pickle.load(f)
f.close()


print('data restore done')



RT_list=sorted(RT_index.keys())
max_RT=RT_list[len(RT_list)-1]
min_RT=RT_list[0]

RT_index_array=dict()
sorted_mz_list=[]
for i in range (0, len(RT_list)):
    RT_value=np.float32(round(RT_list[i], 2))
    RT_index_array[RT_value]=i
    sorted_mz_list.append(sorted(RT_index[RT_value].keys()))        


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
print("%g %g"%(max_mz,min_mz))

rt_search_index=0
while(RT_list[rt_search_index]<=min_RT):
    if RT_list[rt_search_index]==min_RT:
        break
    rt_search_index=rt_search_index+1 

########################################################
f=gzip.open(scanpath+sample_name+'_4DtimsTOF_clusters_mz3_v5c', 'rb')
isotope_cluster, max_num_iso,  total_clusters=pickle.load(f)
f.close()
print('making cluster list')
mz_list=sorted(isotope_cluster.keys())
cluster_list=[]
for i in range (0, len(mz_list)): #len(mz_list)
    ftr_list=isotope_cluster[mz_list[i]]
    for j in range (0, len(ftr_list)):
        ftr=ftr_list[j]        
        cluster_list.append(ftr)



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
    print('%d, %d'%(batch_idx, total_feature))
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
            RT_peak=np.float32(round(feature[int(current_iso[c])][1][0], 2))        
            # 7 step before, peak, 7 step after
#                            count=count+1                          
            RT_s=max(RT_index_array[RT_peak]-RT_window//2, 0)
            RT_e=min(RT_s+RT_window, len(RT_list)) #ex
            charge=int(feature[len(feature)-1][0])   
#                            num_isotopes=int(feature[len(feature)-1][1])
#                            feature_width=min(total_frames_hor, num_isotopes)
            mz_point=round(feature[int(current_iso[c])][1][5], mz_resolution)   #round(peptide_feature[ftr, 0], new_mz_resolution)
            mz_tolerance=round((mz_point*2.0)/10**6, mz_resolution)        

            auc_list=[]
            auc_cal=[]
            cut_block=np.zeros((total_frames_hor, RT_window, mz_window))
            for iso in range (0, total_frames_hor):
    #            mz_point=round(mz_point+isotope_gap[charge], new_mz_resolution)
                mz_s=round(mz_point-mz_tolerance,  mz_resolution)
                mz_e=round(mz_point+mz_tolerance, mz_resolution)
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
                    mz_value=sorted_mz_list[RT_idx][find_mz_idx_start]
    #                        print('rt %d, mz_start %g'%(RT_idx, mz_value))

                    datapoint=RT_index[round(RT_list[RT_idx], 2)][mz_value]    
                    intensity=((datapoint[6]-0)/(maxI-0))*255 #round(, 2) # scale it to the grey value
#                        if datapoint[0]<=10000:
#                            intensity=0
                    mz_dict[rt_row][round(mz_value, 2)].append(intensity)
                    auc_iso_list[rt_row].append(intensity)

                    find_mz_idx_start=find_mz_idx_start+1         
                    mz_value=sorted_mz_list[RT_idx][find_mz_idx_start]
                    while mz_value<=mz_e:                    
                        datapoint=RT_index[round(RT_list[RT_idx], 2)][mz_value]    
                        intensity=((datapoint[6]-0)/(maxI-0))*255 #round(, 2) # scale it to the grey value
#                            if datapoint[0]<=10000:
#                                intensity=0

                        mz_dict[rt_row][round(mz_value, 2)].append(intensity)
                        find_mz_idx_start=find_mz_idx_start+1         
                        mz_value=sorted_mz_list[RT_idx][find_mz_idx_start]
                        auc_iso_list[rt_row].append(intensity)
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
    #                            mz_poz=int(round((mz_value-cut_block_start)/0.01, new_mz_resolution))
    #                            if mz_poz<frame_width:
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
        _prediction_batch , score = sess.run(
            [prediction, pred_score],
            feed_dict={
                batchX_placeholder:batchX,
                batchZ_placeholder:ftr_z[0:current_batch_size] ,
                batchAUC_placeholder:batch_auc_val[0:current_batch_size, :],
#                                init_state:_current_state, 
                keep_prob:1.0, 
                keep_prob_seq:1.0
            })          
#                        for b in range (0, current_batch_size):
#                            pred_charge=int(_prediction_batch[b])
#                            if pred_charge==0:
#                                bt=np.round(cut_block[b], 2)
#                                bt=255-bt
#                                scipy.misc.imsave(datapath+'debug'+str(b)+'.jpg', bt) 
#                        break
        # now ck the prediction for each cluster and set the start iso for next run accordingly
        count=-1
        for c in range (start_cluster, end_cluster):
            if cluster_length[c]<=0:
                continue
            count=count+1            
            _prediction= _prediction_batch[count] 
#                if _prediction==1 and cluster_length[c]==1 and  abs(score[count][0]-np.max(score[count]))<=0.60: # and score[count][0]>=.20: #                    
#                    _prediction=0

            if _prediction>0 :
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

                    new_ftr.append([ftr[len(ftr)-1][0],score[count]]) # charge
                    #new_ftr.append(score[count]) #score
                    feature_table[round(new_ftr[0][0], 2)].append(new_ftr)
                    total_feature=total_feature+1                    

                    start_iso[c]=end_iso                        
                    current_iso[c]=start_iso[c]
                    cluster_length[c]=cluster_length[c]-_prediction-1
#                        if cluster_length[c]==1:
#                            cluster_length[c]=0
#                                    _current_state = np.zeros((batch_size_val, state_size))
            else:
                end_iso=int(current_iso[c]+_prediction+1)
                if current_iso[c]!=start_iso[c]: # it was continuing 
                    new_ftr=[]
                    ftr=cluster_list[c]
                    for isotope in range (int(start_iso[c]),end_iso):
                        new_ftr.append(ftr[isotope])

                    new_ftr.append([ftr[len(ftr)-1][0],score[count]]) # charge
                    #new_ftr.append(score[count]) #score
                    feature_table[round(new_ftr[0][0], 2)].append(new_ftr)
                    total_feature=total_feature+1                    

                start_iso[c]=end_iso                        
                current_iso[c]=start_iso[c]
                cluster_length[c]=cluster_length[c]-_prediction-1
#                    if cluster_length[c]==1:
#                        cluster_length[c]=0

#                                _current_state = np.zeros((batch_size_val, state_size))

############## match feature module ###########################################
print(total_feature)

f=gzip.open(scanpath+sample_name+'_featureTable_v2_timsTOF','wb')
pickle.dump(feature_table, f, protocol=3)
f.close() 
#    ##################
