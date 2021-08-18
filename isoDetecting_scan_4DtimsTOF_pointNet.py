# nohup python -u isoDetecting_scan_4DtimsTOF_pointNet.py [recordpath] [sample_name] [modelpath] [gpu_index] [segment] [scanpath] > output.log &
''' nohup python -u isoDetecting_scan_4DtimsTOF_pointNet.py '/data/anne/timsTOF/hash_records/' 'A1_1_2042' /data/anne/pointIso/3D_model/ 0 1 
/data/anne/timsTOF/scanned_result/ > output.log & '''

from __future__ import division
from __future__ import print_function
#import tensorflow as tf
#import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from time import time
import pickle
import numpy as np
from collections import deque
from collections import defaultdict
#import copy
#import scipy.misc
import sys
import bisect
import gc
import gzip
import os

recordpath=sys.argv[1]
sample_name=sys.argv[2]
modelpath=sys.argv[3]
gpu_index=sys.argv[4]
segment=sys.argv[5]
scanpath=sys.argv[6]


target_part= int(segment) 
gpu=gpu_index
os.environ["CUDA_VISIBLE_DEVICES"]=gpu


total_class=10
RT_unit=0.01
mz_resolution=5
max_part=12
min_part=1

#dataname=['A1_1_2042','A2_1_2043','A3_1_2044','A4_1_2045','A5_1_2046','A6_1_2047', 'A7_1_2048', 'A8_1_2049','A9_1_2050','A10_1_2051', 'A11_1_2052', 'A12_1_2053', 'B1_1_2054',  'B2_1_2055', 'B3_1_2056', 'B4_1_2057']
#data_suffix=['2042','2043','2044','2045','2046','2047', '2048','2049','2050','2051','2052','2053','2054','2055','2056','2057']

delim=','

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

rt_resolution=2
k0_resolution=4
mz_resolution=5

mz_unit=0.00001
k0_unit=0.0001
max_datapoints=0
min_datapoints=10000
RT_window=10
mz_window_unit=1.0

datapoints= 6000 #3000


activation_func=2
set_lr_rate= 0.001 #float(sys.argv[1]) #
log_no='deepIso_pointNet_isoDetect_mz5_4DtimsTOF_displacedData_r1'  
#log_no='deepIso_pointNet_isoDetect_mz5_4DtimsTOF_'+'lrp001r2' #sys.argv[2] #--> 
momentum_value=0.9
reg_weight=0.001
RT_window=10
mz_unit=0.00001
mz_window=int(round(mz_window_unit/mz_unit)) #200
######################################################################
num_class=10
state_size =10
block_size=5
mid_block=12
mid_block_input=4
def weight_variable(shape, variable_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=variable_name)

def bias_variable(shape, variable_name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=variable_name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def bias_variable_Tnet(shape, variable_name):
    initial = tf.constant(np.eye(shape).flatten(), dtype=tf.float32)
    return tf.Variable(initial, name=variable_name)


#################################################################

with tf.device('/device:GPU:'+ gpu): #):  # 
    is_train = tf.placeholder(tf.bool, name="is_train")
    batchX_placeholder = tf.placeholder(tf.float32, [None, 9, datapoints, 4]) #image block to consider for one run of training by back propagation
    sample_weight = tf.placeholder(tf.float32, [None, datapoints]) 
#    batchY_placeholder = tf.placeholder(tf.float32, [None, datapoints])
    batchY_placeholder = tf.placeholder(tf.int32, [None, datapoints])
    keep_probability = tf.placeholder(tf.float32)
    learn_rate=tf.placeholder(tf.float32) #set_lr_rate
    #    total_loss_place = tf.placeholder(tf.float32, [None, 9,datapoints,3])

    # T-Net
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------
    T_net_W_conv0 = weight_variable([1, 4 , 1, 16], 'W_conv0')# 64 kernels each having [1,3] sized filter.
    T_net_b_conv0 = bias_variable([16], 'b_conv0') #for each of feature maps

    T_net_W_conv1 = weight_variable([1, 1 , 16, 32], 'W_conv1')# #20x193
    T_net_b_conv1 = bias_variable([32], 'b_conv1') #for each of feature maps


    T_net_W_fc0 = weight_variable([32, 32], 'W_fc0')
    T_net_b_fc0 = bias_variable([32], 'b_fc0')

    T_net_W_fc1 = weight_variable([32, 16], 'W_fc1')
    T_net_b_fc1 = bias_variable([16], 'b_fc1')
    #
    T_net_W_out = weight_variable([16, 4*4], 'W_out')
    T_net_b_out = bias_variable_Tnet(4, 'b_out')

    # -------------------

    nw_W_conv0 = weight_variable([1, 4 , 1, 16], 'nw_W_conv0')# 64 kernels each having [1,3] sized filter.
    nw_b_conv0 = bias_variable([16], 'nw_b_conv0') #for each of feature maps

#    nw_W_conv0b = weight_variable([1, 1 , 16, 32], 'nw_W_conv0b')# 64 kernels each having [1,3] sized filter.
#    nw_b_conv0b = bias_variable([32], 'nw_b_conv0b') #for each of feature maps

    #--------------------

    F_net_W_conv0 = weight_variable([1, 16 , 1, 16], 'F_W_conv0')# 64 kernels each having [1,3] sized filter.
    F_net_b_conv0 = bias_variable([16], 'F_b_conv0') #for each of feature maps

    F_net_W_conv1 = weight_variable([1, 1 , 16, 32], 'F_W_conv1')# #20x193
    F_net_b_conv1 = bias_variable([32], 'F_b_conv1') #for each of feature maps

    F_net_W_fc0 = weight_variable([32, 32], 'F_W_fc0')
    F_net_b_fc0 = bias_variable([32], 'F_b_fc0')

    F_net_W_fc1 = weight_variable([32, 16], 'F_W_fc1')
    F_net_b_fc1 = bias_variable([16], 'F_b_fc1')

    F_net_W_out = weight_variable([16, 16*16], 'F_W_out')
    
    F_net_b_out = bias_variable_Tnet(16, 'F_b_out')

    #--------------------------
    nw_W_conv1 = weight_variable([1, 16 , 1, 16], 'nw_W_conv1')# 64 kernels each having [1,3] sized filter.
    nw_b_conv1 = bias_variable([16], 'nw_b_conv1') #for each of feature maps

    nw_W_conv2 = weight_variable([1, 1 , 16, 32], 'nw_W_conv2')# #20x193
    nw_b_conv2 = bias_variable([32], 'nw_b_conv2') #for each of feature maps

    #----------------------here you got the F_net+global feature+states-----

    pf_WL = weight_variable([1, 16+32, 1, 64], 'pf_WL')
    pf_bL = bias_variable([64], 'pf_bL')  

    pf_WR = weight_variable([1, 16+32, 1, 64], 'pf_WR')
    pf_bR = bias_variable([64], 'pf_bR')  

    pf_WU = weight_variable([1, 16+32, 1, 64], 'pf_WU')
    pf_bU = bias_variable([64], 'pf_bU')

    pf_WD = weight_variable([1, 16+32, 1, 64], 'pf_WD')
    pf_bD = bias_variable([64], 'pf_bD')  

    pf_WM = weight_variable([1, 16+32, 1, 64], 'pf_WM')
    pf_bM = bias_variable([64], 'pf_bM')  

    attention_weight_L=weight_variable([datapoints, 64], 'attention_weight_L')
    attention_weight_R=weight_variable([datapoints, 64], 'attention_weight_R')
    attention_weight_U=weight_variable([datapoints, 64], 'attention_weight_U')
    attention_weight_D=weight_variable([datapoints, 64], 'attention_weight_D')

    P_W_conv1 = weight_variable([1, 64 , 1, 256], 'P_W_conv1')# 64 kernels each having [1,3] sized filter.
    P_b_conv1 = bias_variable([256], 'P_b_conv1') #for each of feature maps

    P_W_conv2 = weight_variable([1, 1 , 256, 128], 'P_W_conv2')# #20x193
    P_b_conv2 = bias_variable([128], 'P_b_conv2') #for each of feature maps    

    P_W_conv3 = weight_variable([1, 1 , 128, 64], 'P_W_conv3')# #20x193
    P_b_conv3 = bias_variable([64], 'P_b_conv3') #for each of feature maps    

    #    P_W_conv4 = weight_variable([1, 1 , 32, 16], 'P_W_conv4')# #20x193
    #    P_b_conv4 = bias_variable([16], 'P_b_conv4') #for each of feature maps    


    nw_W_out = weight_variable([1, 1 , 64, num_class], 'nw_W_out')
    nw_b_out = bias_variable([num_class], 'nw_b_out')
    
#    var_dictionary={'W_conv0':T_net_W_conv0, 'b_conv0':T_net_b_conv0, 
#    'W_conv1':T_net_W_conv1, 'b_conv1':T_net_b_conv1,
#    'W_fc0':T_net_W_fc0 ,  'b_fc0':T_net_b_fc0,
#    'W_fc1':T_net_W_fc1 ,  'b_fc1':T_net_b_fc1, 
#    'W_out':T_net_W_out ,  'b_out':T_net_b_out,
#    'nw_W_conv0': nw_W_conv0, 'nw_b_conv0': nw_b_conv0, 
#    'F_W_conv0':F_net_W_conv0, 'F_b_conv0':F_net_b_conv0, 
#    'F_W_conv1':F_net_W_conv1, 'F_b_conv1':F_net_b_conv1, 
#    'F_W_fc0': F_net_W_fc0, 'F_b_fc0':F_net_b_fc0, 
#    'F_W_fc1':F_net_W_fc1, 'F_b_fc1':F_net_b_fc1, 
#    'F_W_out': F_net_W_out, 'F_b_out': F_net_b_out, 
#    'nw_W_conv1':nw_W_conv2,  'nw_b_conv2':nw_b_conv2, 
#    'pf_WL':pf_WL, 'pf_WR':pf_WR, 'pf_WU':pf_WU, 'pf_WD':pf_WD,  'pf_WM': pf_WM, 
#    'pf_bL':pf_bL, 'pf_bR':pf_bR, 'pf_bU':pf_bU, 'pf_bD':pf_bD, 'pf_bM':pf_bM, 
#    'attention_weight_L':attention_weight_L, 'attention_weight_R':attention_weight_R, 'attention_weight_U':attention_weight_U, 'attention_weight_D':attention_weight_D, 
#    'P_W_conv1':P_W_conv1, 'P_W_conv2':P_W_conv2, 
#    'P_b_conv1':P_b_conv1,  'P_b_conv2':P_b_conv2
#    }
    #-----------------------------

    global_feature=[]
    local_feature=[]
    ##############
    #2d for loop starts for fw pass
    global_position_idx=0 #for real 3x3 block to hold weighted input
    for row_idx in range (1, 4):
        for col_idx in range (1, 4):
            global_position_idx_state= row_idx*block_size+col_idx
            block_input=batchX_placeholder[:, global_position_idx, :, :]
            T_net_mlp_0 = tf.nn.relu(conv2d(tf.reshape(block_input[:, :, :], [-1, datapoints, 4, 1]) , T_net_W_conv0) + T_net_b_conv0) # now the layer is : b x n x 64  
            T_net_mlp_1 = tf.nn.relu(conv2d(T_net_mlp_0, T_net_W_conv1) + T_net_b_conv1) # now layer is : b x n x 128
        #    T_net_mlp_2 = tf.tanh(conv2d(T_net_mlp_1, T_net_W_conv2) + T_net_b_conv2) # now the layer is : b x n x 1024
            T_net_maxpool_points=tf.nn.max_pool(T_net_mlp_1, ksize=[1, datapoints, 1, 1], strides=[1, datapoints, 1, 1], padding='SAME')
            T_net_maxpool_points = tf.reshape(T_net_maxpool_points, [-1, 32])
            T_net_fc0 = tf.nn.relu(tf.matmul(T_net_maxpool_points, T_net_W_fc0) + T_net_b_fc0) # finally giving the output
            T_net_fc1 = tf.nn.relu(tf.matmul(T_net_fc0, T_net_W_fc1) + T_net_b_fc1) # finally giving the output
            T_net_point_transformation_matrix = tf.nn.relu(tf.matmul(T_net_fc1, T_net_W_out) + T_net_b_out) # finally giving the output [b x 3 x 3]
            ##############################

            T_net_point_transformation_matrix = tf.reshape(T_net_point_transformation_matrix, [-1, 4, 4])
            T_net_point_transformation = tf.matmul(block_input, T_net_point_transformation_matrix) # dimension [n x 3]
            # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            nw_mlp_0 = tf.nn.relu(conv2d(tf.reshape(T_net_point_transformation[:, :, :], [-1, datapoints, 4, 1]), nw_W_conv0) + nw_b_conv0) # now the layer is : b x n x 64  
#            nw_mlp_0b = tf.nn.relu(conv2d(nw_mlp_0, nw_W_conv0b) + nw_b_conv0b) # now the layer is : b x n x 64 
            nw_mlp_0b = tf.reshape(nw_mlp_0,[-1,datapoints,16])        
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            F_net_mlp_0 = tf.nn.relu(conv2d(tf.reshape(nw_mlp_0b[:, :, :], [-1, datapoints, 16, 1]), F_net_W_conv0) + F_net_b_conv0) # now the layer is : b x n x 64  
            F_net_mlp_1 = tf.nn.relu(conv2d(F_net_mlp_0, F_net_W_conv1) + F_net_b_conv1) # now layer is : b x n x 128
        #    F_net_mlp_2 = tf.tanh(conv2d(F_net_mlp_1, F_net_W_conv2) + F_net_b_conv2) # now the layer is : b x n x 1024
            F_net_maxpool_points=tf.nn.max_pool(F_net_mlp_1, ksize=[1, datapoints, 1, 1], strides=[1, datapoints, 1, 1], padding='SAME')
            F_net_maxpool_points = tf.reshape(F_net_maxpool_points, [-1, 32])
            F_net_fc0 = tf.nn.relu(tf.matmul(F_net_maxpool_points, F_net_W_fc0) + F_net_b_fc0) # finally giving the output
            F_net_fc1 = tf.nn.relu(tf.matmul(F_net_fc0, F_net_W_fc1) + F_net_b_fc1) # finally giving the output
            F_net_point_transformation_matrix = tf.nn.relu(tf.matmul(F_net_fc1, F_net_W_out) + F_net_b_out) # finally giving the output [b x 3 x 3]
            ##############################

            F_net_point_transformation_matrix = tf.reshape(F_net_point_transformation_matrix, [-1, 16, 16])
            #if row_idx==2 and col_idx==2:
            #    transformation_matrix_holder=F_net_point_transformation_matrix
            F_net_point_transformation = tf.matmul(nw_mlp_0b , F_net_point_transformation_matrix) # dimension [n x 64]
            local_feature.append(F_net_point_transformation)
            # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            nw_mlp_1 = tf.nn.relu(conv2d(tf.reshape(F_net_point_transformation, [-1, datapoints, 16, 1]), nw_W_conv1) + nw_b_conv1) # now the layer is : b x n x 64  
            nw_mlp_2 = tf.nn.relu(conv2d(nw_mlp_1, nw_W_conv2) + nw_b_conv2) # now layer is : b x n x 128


            nw_maxpool_points=tf.nn.max_pool(nw_mlp_2, ksize=[1, datapoints, 1, 1], strides=[1, datapoints, 1, 1], padding='SAME')
            global_feature.append(tf.reshape(nw_maxpool_points, [-1, 32]) ) # this is the global feature
            global_position_idx=global_position_idx+1


    #####-------------following block will be executed only for mid block------------------------------------------------  
    feature_l=tf.reshape( global_feature[3], [-1, 1,  global_feature[3].shape[1].value])
    feature_l= tf.tile(feature_l,  [1, datapoints, 1])
    feature_l=tf.concat((local_feature[3],feature_l), axis=2) 
    point_feature_l=tf.nn.relu(conv2d(tf.reshape(feature_l, [-1, datapoints, 16+32, 1]), pf_WL) + pf_bL) # B
    point_feature_l=tf.reshape(point_feature_l, [-1, datapoints,point_feature_l.shape[3].value]) # [?, 3000, 16] = point_feature


    feature_r=tf.reshape(global_feature[5], [-1, 1,  global_feature[5].shape[1].value])
    feature_r= tf.tile(feature_r,  [1, datapoints, 1])
    feature_r=tf.concat((local_feature[5],feature_r), axis=2)
    point_feature_r=tf.nn.relu(conv2d(tf.reshape(feature_r, [-1, datapoints, 16+32, 1]), pf_WR) + pf_bR) 
    point_feature_r=tf.reshape(point_feature_r, [-1, datapoints,point_feature_r.shape[3].value]) # [?, 3000, 16] = point_feature

    feature_u=tf.reshape(global_feature[7], [-1, 1,  global_feature[1].shape[1].value])
    feature_u= tf.tile(feature_u, [1, datapoints, 1])
    feature_u=tf.concat((local_feature[7],feature_u), axis=2)
    point_feature_u=tf.nn.relu(conv2d(tf.reshape(feature_u, [-1, datapoints, 16+32, 1]), pf_WU) + pf_bU) 
    point_feature_u=tf.reshape(point_feature_u, [-1, datapoints,point_feature_u.shape[3].value]) # [?, 3000, 16] = point_feature

    feature_d=tf.reshape(global_feature[1], [-1, 1,  global_feature[1].shape[1].value])
    feature_d= tf.tile(feature_d,  [1, datapoints, 1])
    feature_d=tf.concat((local_feature[1],feature_d), axis=2)
    point_feature_d=tf.nn.relu(conv2d(tf.reshape(feature_d, [-1, datapoints, 16+32, 1]), pf_WD) + pf_bD) 
    point_feature_d=tf.reshape(point_feature_d, [-1, datapoints,point_feature_d.shape[3].value]) # [?, 3000, 16] = point_feature

    global_feature_expand= tf.tile(tf.reshape(global_feature[4], [-1, 1,  global_feature[4].shape[1].value]), [1, datapoints, 1]) # [?, 3000, 32]
    concat_local_global=tf.concat((local_feature[4],global_feature_expand), axis=2) # [?, 3000, 32+16] 
    # pass it through another weight???
    point_feature_m=tf.nn.relu(conv2d(tf.reshape(concat_local_global, [-1, datapoints, 16+32, 1]), pf_WM) + pf_bM)    # C
    point_feature_m=tf.reshape(point_feature_m, [-1, datapoints,point_feature_m.shape[3].value]) # [?, 3000, 16]  = point_feature    

    filtered_left=tf.matmul(tf.nn.softmax(tf.matmul(point_feature_m,  tf.transpose(point_feature_l, perm=[0,2,1])),  axis=2) , point_feature_l ) # [?, 3000, 16]  = point_feature    
    attention_left=tf.multiply(filtered_left, attention_weight_L) # [?, 3000, 16]  = point_feature    

    filtered_right=tf.matmul(tf.nn.softmax(tf.matmul(point_feature_m,  tf.transpose(point_feature_r, perm=[0,2,1])),  axis=2) , point_feature_r )
    attention_right=tf.multiply(filtered_right, attention_weight_R) # [?, 3000, 16]  = point_feature    

    filtered_up=tf.matmul(tf.nn.softmax(tf.matmul(point_feature_m,  tf.transpose(point_feature_u, perm=[0,2,1])),  axis=2) , point_feature_u )
    attention_up=tf.multiply(filtered_up, attention_weight_U) # [?, 3000, 16]  = point_feature    

    filtered_down=tf.matmul(tf.nn.softmax(tf.matmul(point_feature_m,  tf.transpose(point_feature_d, perm=[0,2,1])),  axis=2), point_feature_d)  
    attention_down=tf.multiply(filtered_down, attention_weight_D) # [?, 3000, 16]  = point_feature    

    surrounding_feature=tf.add(attention_left,attention_right)
    surrounding_feature= tf.add(surrounding_feature,attention_up)
    surrounding_feature= tf.add(surrounding_feature,attention_down) #[?, dataoints, 16]
    # surrounding_feature holds all useful point features from up, down, right, left for each datapoints in current window

    concat_local_global_surround=tf.add(point_feature_m, surrounding_feature)  #[?, dataoints, 16]

    p_mlp_1 = tf.nn.relu(conv2d(tf.reshape(concat_local_global_surround[:, :, :], [-1, datapoints, concat_local_global_surround.shape[2].value, 1]), P_W_conv1) + P_b_conv1) # finally giving the output
    p_mlp_2 = tf.nn.relu(conv2d(tf.nn.dropout(p_mlp_1, keep_probability), P_W_conv2) + P_b_conv2)
    p_mlp_3 = tf.nn.relu(conv2d(tf.nn.dropout(p_mlp_2, keep_probability), P_W_conv3) + P_b_conv3)
    #    p_mlp_4 = tf.nn.relu(conv2d(tf.nn.dropout(p_mlp_3, keep_probability), P_W_conv4) + P_b_conv4)

    nw_out =  conv2d(tf.nn.dropout(p_mlp_3, keep_probability), nw_W_out) +  nw_b_out
    #    nw_out =  conv2d(p_mlp_3, nw_W_out) +  nw_b_out
    nw_out=tf.reshape(nw_out, [-1, datapoints, num_class])
    prediction = tf.argmax(tf.nn.softmax(nw_out), 2) # it should be [n x num_class]
    #    prediction_score = np.max(tf.nn.softmax(nw_out), 2)
    
    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=nw_out, labels=batchY_placeholder) # batchY_placeholder = [n x num_class] -- one hot vector for each n
    considered_loss=tf.multiply(sample_weight, loss)  
    total_loss=tf.reduce_mean(tf.reduce_mean(considered_loss, axis=1))

#    train_step = tf.contrib.opt.NadamOptimizer(learn_rate).minimize(total_loss)
    
config=tf.ConfigProto(allow_soft_placement=True) #, log_device_placement=True)
#config=tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver = tf.train.Saver()
#saver.restore(sess, modelpath+log_no+'_best_model.ckpt')
saver.restore(sess, modelpath+log_no+'_best_model_zero.ckpt')
print('~ Model built ~ ')
####################################################################

print('trying to load ms1 record')

part=target_part
f=gzip.open(recordpath+sample_name+'_RT_index_part'+str(part), mode='rb')
RT_index=pickle.load(f)
f.close()

if part+1<=max_part:
    f=gzip.open(recordpath+sample_name+'_RT_index_part'+str(part+1), mode='rb')
    RT_index_temp=pickle.load(f)
    f.close()
    RT_list=sorted(RT_index_temp.keys(), reverse=False)
    for rt_idx in range (0, min(RT_window,len(RT_list))):
        print('rt indx %d'%rt_idx)
        rt_value=np.float32(RT_list[rt_idx])
        RT_index[rt_value]=RT_index_temp[rt_value]


if part-1>=min_part:
    f=gzip.open(recordpath+sample_name+'_RT_index_part'+str(part-1), mode='rb')
    RT_index_temp=pickle.load(f)
    f.close()
    RT_list=sorted(RT_index_temp.keys(), reverse=True)
    for rt_idx in range (0, RT_window):
        rt_value=np.float32(RT_list[rt_idx])
        RT_index[rt_value]=RT_index_temp[rt_value]

RT_index_temp=0

f=open(recordpath+'pointCloud_'+sample_name+'_maxI_k0', 'rb')
maxI=pickle.load(f)
f.close()

sorted_mz_list=[] 
RT_index_array=dict()
RT_list=sorted(RT_index.keys())
i=0
while (i<len(RT_list)):
    RT_value=np.float32(round(RT_list[i], 2))
    RT_index_array[RT_value]=i
    #making sorted_mz_list
    sorted_mz_list.append(sorted(RT_index[RT_value].keys()))
    i=i+1

print('RT_list len %d'%len(RT_list))
#    copy_RT_index=copy.deepcopy(RT_index)

###########################
#scan ms1_block and record the cnn outputs in list_dict[z]: hash table based on m/z
#for each m/z
max_RT=RT_list[len(RT_list)-1]
min_RT=RT_list[0]

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
while(RT_list[rt_search_index]<=min_RT):
    if RT_list[rt_search_index]==min_RT:
        break
    rt_search_index=rt_search_index+1 

total_RT=len(RT_list)-rt_search_index

##################1/k0 list###########################
f=gzip.open(recordpath+'pointCloud_'+sample_name+'_k0_dict', 'rb')
k0_dict=pickle.load(f)
f.close()       
################label########################    

mz_used_before=np.zeros((total_class))    
pred_RT=np.zeros((total_class))
pred_start=np.zeros((total_class))

total_intensity=[]
list_dict=[]
list_dict_RT=[]
for i in range (0, total_class):
    list_dict.append(dict())
    total_intensity.append(dict())
    list_dict_RT.append(dict())
batch_size=1

current_mz=min_mz
current_RT=min_RT

total_time=time()
while current_mz<max_mz:
    start_time=time()
    print('##### mz:%g #######'%current_mz)
    output_list=dict()
#        output_list=np.zeros((batch_size, total_RT, mz_window)) #
    real_RT_index=rt_search_index
    real_img_row=RT_window #check
    while real_img_row<= total_RT-RT_window-RT_window:
        RT_index_start=real_RT_index
#            print('RT for output:%d, mz for output: %g'%(real_img_row, current_mz))
        batch_ms1=np.zeros((batch_size,9, datapoints, 4))
        batch_points=np.zeros((batch_size, 9))

        count=0
        #####
        #prepare it
        RT_inc=RT_window
        for row_idx in range (0, 3):
            mz_start=np.float32(round(current_mz-mz_window*mz_unit, mz_resolution))
            RT_index_end=min(RT_index_start+RT_window-1, len(RT_list)-1) #inc
            rt_idx_s=RT_index_start
            rt_idx_e=RT_index_end
            if row_idx==0:
                rt_idx_s=rt_idx_s+RT_window//2
            elif row_idx==2:
                rt_idx_e=rt_idx_e-RT_window//2                
            for col_idx in range (0, 3):
#                    print('mz_start %g'%(mz_start))
                flat_index=row_idx*3+col_idx
                point_index=0
                mz_end=np.float32(round(mz_start+mz_window_unit-mz_unit, mz_resolution)) #inc, we don't allow overlap, like 700 to 701.99 --> 200 pixels

                rt_row=0
                break_flag=-1
                for RT_idx in range (rt_idx_s, rt_idx_e+1):
                    if RT_idx<0 or RT_idx>(len(RT_list)-1):
                        rt_row=rt_row+1
                        continue
                    rt_value=RT_list[RT_idx]                    
                    mz_value=mz_start
                    find_mz_idx_start= bisect.bisect_left(sorted_mz_list[RT_idx], mz_value)
                    if len(sorted_mz_list[RT_idx])==find_mz_idx_start or sorted_mz_list[RT_idx][find_mz_idx_start]>mz_end:
                        rt_row=rt_row+1
                        continue
                    mz_value=sorted_mz_list[RT_idx][find_mz_idx_start]    

                    mz_row=round(mz_value-mz_start,mz_resolution) #scale it to 0 to 1.99 
                    k0_list=list(set(sorted(RT_index[rt_value][mz_value].keys()))-{6})


                    for k0_value in k0_list:
                        if k0_value not in k0_dict:
                            continue

                        if point_index<datapoints:
                            #k0_value can be directly inserted w/o scaling since whole range is accepted
                            intensity=round(((RT_index[rt_value][mz_value][k0_value][0]-0)/(maxI-0))*255, 2) # scale it to the grey value                              
                            batch_ms1[count, flat_index, point_index, 0]=mz_row
                            batch_ms1[count, flat_index, point_index, 1]=rt_row
                            batch_ms1[count, flat_index, point_index, 2]=k0_value
                            batch_ms1[count, flat_index, point_index, 3]=intensity
                            point_index=point_index+1
                        else:
                            break_flag=1
                            break

                    find_mz_idx_start=find_mz_idx_start+1         
                    if len(sorted_mz_list[RT_idx])>find_mz_idx_start:
                            mz_value= sorted_mz_list[RT_idx][find_mz_idx_start]
                    else:
                        rt_row=rt_row+1
                        continue

                    while mz_value<=mz_end:
                        mz_row=round(mz_value-mz_start,mz_resolution) # 0 to 1.99999
                        k0_list=list(set(sorted(RT_index[rt_value][mz_value].keys()))-{6})
                        mz_row=round(mz_value-mz_start, mz_resolution)  #(()/(mz_end-mz_start))*mz_difference, mz_resolution) # scale it to 0 to 1.99 
                        for k0_value in k0_list:
                            if k0_value not in k0_dict:
                                continue

                            if point_index<datapoints:
                                #k0_value can be directly inserted w/o scaling since whole range is accepted
                                intensity=round(((RT_index[rt_value][mz_value][k0_value][0]-0)/(maxI-0))*255, 2) # scale it to the grey value                              
                                batch_ms1[count, flat_index, point_index, 0]=mz_row
                                batch_ms1[count, flat_index, point_index, 1]=rt_row
                                batch_ms1[count, flat_index, point_index, 2]=k0_value
                                batch_ms1[count, flat_index, point_index, 3]=intensity   
                                point_index=point_index+1
                            else:
                                break_flag=1
                                break

                        if break_flag==1:
                            break
                        find_mz_idx_start=find_mz_idx_start+1
                        if len(sorted_mz_list[RT_idx])>find_mz_idx_start:
                            mz_value= sorted_mz_list[RT_idx][find_mz_idx_start]
                        else:
                            break

                    if break_flag==1:
                        break
                    rt_row=rt_row+1

                mz_start=np.float32(round(mz_end+mz_unit, mz_resolution)) # go to next right window
                if col_idx==1 and row_idx==1: # mid block
                    if RT_idx<rt_idx_e: # 0 to 14 = 15. if stopped at 12, then next scan starts at 13, instead of 15. 12-0+1=13. otherwise, 14-0+1=15. 
                        RT_inc=RT_idx-rt_idx_s #+1
                batch_points[count, flat_index]=point_index

            RT_index_start=min(RT_index_start+min(RT_window, RT_inc), len(RT_list)-1) # go to next up window
        #####
    # one 3x3 seq is formed
#            print('block is formed')
        if int(batch_points[count, 4])!=0 and np.max(batch_ms1[count, 4, :, 3])>0:                
            batchX = batch_ms1 #[:,row_idx,:,:]                             
            _prediction = sess.run(
                prediction,
                feed_dict={
                    batchX_placeholder:batchX,
                    keep_probability:1.0, 
                    learn_rate:set_lr_rate,
                    is_train:False
                })                                        

            #one batch is done
            count=0

            for point_idx in range (0, int(batch_points[count, 4])):
                mz_index=int(round(batch_ms1[count, 4, point_idx, 0]/mz_unit, mz_resolution)) # 0 to 199999
                rt_index=int(batch_ms1[count, 4, point_idx, 1])  
                k0_index=round(batch_ms1[count, 4, point_idx, 2], k0_resolution) 
                if mz_index not in output_list:
                    output_list[mz_index]=defaultdict(list)

                output_list[mz_index][real_img_row+rt_index].append([_prediction[count][point_idx], k0_index, batch_ms1[count, 4, point_idx, 3]])
        else:
            print('blank!')

        real_img_row=real_img_row+min(RT_window, RT_inc)
        real_RT_index=real_RT_index+min(RT_window, RT_inc)


    for batch_index in range (0, batch_size):
        mz_keys=sorted(output_list.keys())
        for j in mz_keys: #range (0, int(mz_window)): ## <---
            mz_poz=round(current_mz+j*mz_unit, mz_resolution)
#                if mz_poz==707.36: 
#                    break                
            mz_used_before[:]=0
            pred_RT[:]=0
            pred_start[:]=0
            not_exist=1
            RT_keys=sorted(output_list[j].keys())
            for count_rt_index in range (0,  len(RT_keys)): #range (RT_window, total_RT):
                i=RT_keys[count_rt_index]
                RT_poz=round(RT_list[rt_search_index+i], 2) 
                current_intensity=np.zeros((num_class))
                z_exist=np.zeros((num_class))
                for z_idx in output_list[j][i]: 
                    z=int(z_idx[0])
                    k0_poz=z_idx[1]
                    k0_intensity=z_idx[2]
                    if z!=0:
                        z_exist[z]=1                            
                        if RT_poz not in list_dict_RT[z]:
                            list_dict_RT[z][RT_poz]=defaultdict(list)
                        list_dict_RT[z][RT_poz][mz_poz].append([k0_poz, k0_intensity])
                        current_intensity[z]=current_intensity[z]+(k0_intensity/255)*maxI

                for z in range (0, num_class):
                    if z_exist[z]==1:
                        # add (m/z,RT) to the dict
                        if mz_used_before[z]==1:  #list_dict[p_ion[i]].has_key(mz_poz):
                            # append the new number to the existing array at this slot
                            if RT_index_array[np.float32(RT_poz)]-RT_index_array[np.float32(pred_RT[z])]==1: #continuation of same isotope
                                pred_RT[z]=RT_poz
                                #merge: dict() already exist
                                total_intensity[z][RT_poz]=current_intensity[z]                                    

                            elif pred_start[z]==pred_RT[z]: # the RT span of this isotope is only one scan  
                                list_dict[z][mz_poz].append([[pred_start[z]], total_intensity[z]])
                                pred_start[z]=pred_RT[z]=RT_poz
                                total_intensity[z]=dict()
                                total_intensity[z][RT_poz]=current_intensity[z]


                            else: #if the RT span of current isotope is => 2 scan then keep it
                                list_dict[z][mz_poz].append([[pred_start[z], pred_RT[z]], total_intensity[z]])
                                pred_start[z]=pred_RT[z]=RT_poz
                                total_intensity[z]=dict()
                                total_intensity[z][RT_poz]=current_intensity[z]

                        else: #this mz does not exist
                            # create a new array in this slot
                            list_dict[z][mz_poz] = deque()#[RT_poz]
                            mz_used_before[z]=1
                            pred_start[z]=pred_RT[z]=RT_poz
                            total_intensity[z]=dict()
                            total_intensity[z][RT_poz]=current_intensity[z]



            for z in range (1, total_class):
                if mz_used_before[z]==1: 
                    if pred_start[z]==pred_RT[z]: # the RT span of this isotope is only .01 minute (one step)
                        list_dict[z][mz_poz].append([[pred_start[z]],total_intensity[z]])
                    else: #if the RT span of current isotope is => 2 step (.02 minute) then keep it
                        list_dict[z][mz_poz].append([[pred_start[z], pred_RT[z]],total_intensity[z]])

            # all rt done for one mz
    #one stripe done
    current_mz=round(current_mz+(mz_window)*mz_unit, mz_resolution) ## <------
# followings are tabbed back
    time_elapsed=time()-start_time 
    print(time_elapsed)



print('total time taken %g'%(time()-total_time))     
print('writing')

f=gzip.open(scanpath+sample_name+'_timsTOF_list_dict_'+str(part), 'wb') #v3r2
pickle.dump([list_dict,part], f, protocol=3) #all mz_done
f.close()

f=gzip.open(scanpath+sample_name+'_timsTOF_list_dict_RT_'+str(part), 'wb') #v3r2
pickle.dump([list_dict_RT,part], f, protocol=3) #all mz_done
f.close()
print('done')












