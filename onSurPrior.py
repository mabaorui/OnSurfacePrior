# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:44:22 2020

@author: Administrator
"""
import numpy as np
#import tensorflow as tf 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os 
import shutil
import random
import math
import scipy.io as sio
import time
import argparse
import trimesh
import re
from scipy.spatial import cKDTree
from plyfile import PlyData
from plyfile import PlyElement
#import mcubes
from skimage.measure import marching_cubes_lewiner


parser = argparse.ArgumentParser()
parser.add_argument('--dis',action='store_true', default=False)
parser.add_argument('--train',action='store_true', default=False)
parser.add_argument('--test',action='store_true', default=False)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--class_idx', type=str, default="026911156")
parser.add_argument('--save_idx', type=int, default=-1)
parser.add_argument('--CUDA', type=int, default=0)
parser.add_argument('--dataset', type=str, default="other")
parser.add_argument('--finetune_dir', type=str, default="no_finetune")
parser.add_argument('--INPUT_NUM', type=int, default=0)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--input_ply_file', type=str, default="test.ply")
a = parser.parse_args()

cuda_idx = str(a.CUDA)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_idx



BS = 1
knn = 50
POINT_NUM_SPARSE = 500
POINT_NUM = 4096
POINT_NUM_GT = 20000
INPUT_DIR = a.data_dir
#INPUT_DIR = '/home/mabaorui/AtlasNetOwn/data/sphere/'
OUTPUT_DIR = a.out_dir

#GT_DIR = '/data/mabaorui/common_data/ShapeNetCore.v1/' + a.class_idx + '/'
#GT_DIR = '/data1/mabaorui/nerualpull_gan/data/scene_data/cvpr/water_' + a.class_idx + '/'
#GT_DIR = '/data/mabaorui/nerualpull_gan/data/scene_data/cvpr/' + a.class_idx + '/'
GT_DIR = '/data1/mabaorui/nerualpull_gan/data/scene_data/eccv/'
TRAIN = a.train
bd = 0.6

# fileAll = os.listdir(a.test_dir)
# for file in fileAll:
#     if(re.findall(r'.*.npz', file, flags=0)):
#         test_num  = test_num + 1

test_num = 2000

if(TRAIN or a.dis):
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print ('test_res_dir: deleted and then created!')
    os.makedirs(OUTPUT_DIR)



def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
  """ Fully connected layer with non-linear operation.
  
  Args:
    inputs: 2-D tensor BxN
    num_outputs: int
  
  Returns:
    Variable tensor of size B x num_outputs.
  """
  with tf.variable_scope(scope) as sc:
    num_input_units = inputs.get_shape()[-1].value
    weights = _variable_with_weight_decay('weights',
                                          shape=[num_input_units, num_outputs],
                                          use_xavier=use_xavier,
                                          stddev=stddev,
                                          wd=weight_decay)
    outputs = tf.matmul(inputs, weights)
    biases = _variable_on_cpu('biases', [num_outputs],
                             tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
     

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs
def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D max pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs
def _variable_on_cpu(name, shape, initializer, use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)


      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs



def safe_norm_np(x, epsilon=1e-12, axis=1):
    return np.sqrt(np.sum(x*x, axis=axis) + epsilon)

def safe_norm(x, epsilon=1e-12, axis=None):
  return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)

def boundingbox(x,y,z):
    return min(x),max(x),min(y),max(y),min(z),max(z)

    

def chamfer_distance_tf_None(array1, array2):
    array1 = tf.reshape(array1,[-1,3])
    array2 = tf.reshape(array2,[-1,3])
    av_dist1 = av_dist_None(array1, array2)
    av_dist2 = av_dist_None(array2, array1)
    return av_dist1+av_dist2

def distance_matrix_None(array1, array2, num_point, num_features = 3):
    """
    arguments: 
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1
            , it's size: (num_point, num_point)
    """
    expanded_array1 = tf.tile(array1, (num_point, 1))
    expanded_array2 = tf.reshape(
            tf.tile(tf.expand_dims(array2, 1), 
                    (1, num_point, 1)),
            (-1, num_features))
    distances = tf.norm(expanded_array1-expanded_array2, axis=1)
    distances = tf.reshape(distances, (num_point, num_point))
    return distances

def av_dist_None(array1, array2):
    """
    arguments:
        array1, array2: both size: (num_points, num_feature)
    returns:
        distances: size: (1,)
    """
    distances = distance_matrix_None(array1, array2,points_input_num[0,0])
    distances = tf.reduce_min(distances, axis=1)
    distances = tf.reduce_mean(distances)
    return distances


def vis_single_points_with_color(points, colors, plyname): 
    
    
    header = "ply\n" \
             "format ascii 1.0\n" \
             "element vertex {}\n" \
             "property double x\n" \
             "property double y\n" \
             "property double z\n" \
             "property uchar red\n" \
             "property uchar green\n" \
             "property uchar blue\n" \
             "end_header\n".format(points.shape[0])
    with open(plyname, 'w') as f:
        f.write(header)
        for i in range(int(points.shape[0])):
            f.write('{} {} {} {} {} {}\n'.format(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]))

def vis_single_points(points, plyname): 
    
    
    header = "ply\n" \
             "format ascii 1.0\n" \
             "element vertex {}\n" \
             "property double x\n" \
             "property double y\n" \
             "property double z\n" \
             "property uchar red\n" \
             "property uchar green\n" \
             "property uchar blue\n" \
             "end_header\n".format(points.shape[0])
    with open(plyname, 'w') as f:
        f.write(header)
        for i in range(int(points.shape[0])):
            f.write('{} {} {} {} {} {}\n'.format(points[i,0], points[i,1], points[i,2], 255, 0, 0))

def knn_extractor(adj_matrix, k=5):
    """Get KNN based on the pairwise distance.
    Args:
      pairwise distance: (batch_size, num_points, num_points)
      k: int
    Returns:
      nearest neighbors: (batch_size, num_points, k)
    """
    neg_adj = -adj_matrix
    dis, nn_idx = tf.nn.top_k(neg_adj, k=k)
    return nn_idx, dis


def pairwise_distance_gt(point_cloud_src, point_cloud_target):
    """Compute pairwise distance of a point cloud.
    Args:
      point_cloud_src: tensor (batch_size, num_points, num_dims)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    point_cloud_transpose = tf.transpose(point_cloud_target, perm=[0, 2, 1])
    point_cloud_inner = tf.matmul(point_cloud_src, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud_src), axis=-1, keep_dims=True)
    point_cloud_square_tranpose = tf.transpose(tf.reduce_sum(tf.square(point_cloud_target), axis=-1, keep_dims=True), perm=[0, 2, 1])

    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

def get_neighbors(point_cloud, nn_idx):
    """Construct edge feature for each point
    Args:
      point_cloud: (batch_size, num_points, 1, num_dims)
      nn_idx: (batch_size, num_points, k)
      k: int
    Returns:
      edge features: (batch_size, num_points, k, num_dims)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)

    return point_cloud_neighbors
def sample_query_points(input_ply_file):
    data = PlyData.read(a.data_dir + input_ply_file)
    v = data['vertex'].data
    v = np.asarray(v)
    print(v.shape)

    #rt = np.random.choice(v.shape, 50000, replace = False)

    points = []
    for i in range(v.shape[0]):
        points.append(np.array([v[i][0],v[i][1],v[i][2]]))
    points = np.asarray(points)
    pointcloud_s =points.astype(np.float32)
    print('pointcloud sparse:',pointcloud_s.shape[0])
    
    pointcloud_s_t = pointcloud_s - np.array([np.min(pointcloud_s[:,0]),np.min(pointcloud_s[:,1]),np.min(pointcloud_s[:,2])])
    pointcloud_s_t = pointcloud_s_t / (np.array([np.max(pointcloud_s[:,0]) - np.min(pointcloud_s[:,0]), np.max(pointcloud_s[:,0]) - np.min(pointcloud_s[:,0]), np.max(pointcloud_s[:,0]) - np.min(pointcloud_s[:,0])]))
    trans = np.array([np.min(pointcloud_s[:,0]),np.min(pointcloud_s[:,1]),np.min(pointcloud_s[:,2])])
    scal = np.array([np.max(pointcloud_s[:,0]) - np.min(pointcloud_s[:,0]), np.max(pointcloud_s[:,0]) - np.min(pointcloud_s[:,0]), np.max(pointcloud_s[:,0]) - np.min(pointcloud_s[:,0])])
    pointcloud_s = pointcloud_s_t
    
    print(np.min(pointcloud_s[:,0]), np.max(pointcloud_s[:,0]))
    print(np.min(pointcloud_s[:,1]), np.max(pointcloud_s[:,1]))
    print(np.min(pointcloud_s[:,2]), np.max(pointcloud_s[:,2]))
    
    sample = []
    sample_near = []
    sample_near_o = []
    sample_dis = []
    sample_vec = []
    
    for i in range(int(1000000/pointcloud_s.shape[0])):
        
        pnts = pointcloud_s
        ptree = cKDTree(pnts)
        i = 0
        sigmas = []
        for p in np.array_split(pnts,100,axis=0):
            d = ptree.query(p,51)
            sigmas.append(d[0][:,-1])
        
            i = i+1
        
        sigmas = np.concatenate(sigmas)
        sigmas_big = 0.2 * np.ones_like(sigmas)
        sigmas = sigmas
        
        #tt = pnts + 0.5*0.25*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pnts.shape)
        tt = pnts + 0.25*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pnts.shape)
        #tt = pnts + 1*np.expand_dims(sigmas_big,-1) * np.random.normal(0.0, 1.0, size=pnts.shape)
        sample.append(tt)
        
    """ for i in range(int(1000000/pointcloud_s.shape[0])):
        
        pnts = pointcloud_s
        ptree = cKDTree(pnts)
        i = 0
        sigmas = []
        for p in np.array_split(pnts,100,axis=0):
            d = ptree.query(p,51)
            sigmas.append(d[0][:,-1])
        
            i = i+1
        
        sigmas = np.concatenate(sigmas)
        sigmas_big = 0.2 * np.ones_like(sigmas)
        sigmas = sigmas
        
        #tt = pnts + 0.25*0.25*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pnts.shape)
        tt = pnts + 0.5*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pnts.shape)
        #tt = pnts + 1*np.expand_dims(sigmas_big,-1) * np.random.normal(0.0, 1.0, size=pnts.shape)
        sample.append(tt) """
        



    
    sample = np.asarray(sample).reshape(-1,3)
    np.savez_compressed(a.data_dir + input_ply_file , sample = sample, pointcloud_s = pointcloud_s, trans = trans, scal = scal)

#if(TRAIN or a.dis):
sample_query_points(a.input_ply_file)
files = []
files_path = []


fileAll = os.listdir(INPUT_DIR)
for file in fileAll:
    if(re.findall(r'.*.npz', file, flags=0)):
        #print(file.strip().split('.')[0])
        files.append(file.strip().split('.')[0])

for file in files:
    files_path.append(INPUT_DIR + file + '.npz')
  
    
def get_data_from_filename(filename):
    load_data = np.load(filename)
    
    sample = np.asarray(load_data['sample']).reshape(-1,3)
    lable = np.asarray(load_data['sample_dis']).reshape(-1,1)
    point = np.asarray(load_data['sample_vec']).reshape(-1,knn,3)
    return sample.astype(np.float32), lable.astype(np.float32), point.astype(np.float32)
    # rt = np.random.choice(sample.shape[0], POINT_NUM, replace = False)
    
    # return sample[rt,:].astype(np.float32), lable[rt,:].astype(np.float32), point[rt,:,:].astype(np.float32)

filelist = tf.placeholder(tf.string, shape=[None])
ds = tf.data.Dataset.from_tensor_slices((filelist))

ds = ds.map(
    lambda item: tuple(tf.py_func(get_data_from_filename, [item], (tf.float32, tf.float32, tf.float32))),num_parallel_calls = 32)
ds = ds.repeat()  # Repeat the input indefinitely.
ds = ds.batch(1)
ds = ds.prefetch(buffer_size = 200)
iterator = ds.make_initializable_iterator()
next_element = iterator.get_next()

SHAPE_NUM = len(files_path)
print('SHAPE_NUM:',SHAPE_NUM)


pointclouds = []
samples = []
lables = []
mm = 0
#if(a.train or a.test):
# if(a.train):
#     for file in files_path:
#         if(mm>5):
#             break
#         mm = mm + 1
#         #print('load:',file)
#         load_data = np.load(file)
        
#         sample = np.asarray(load_data['sample']).reshape(-1,3)
#         #lable = np.asarray(load_data['sample_dis']).reshape(-1,1)
#         point = np.asarray(load_data['pointcloud_s']).reshape(1,POINT_NUM_SPARSE,3)
#         #if(a.train):
#         #    point = np.asarray(load_data['pointcloud_s']).reshape(1,POINT_NUM_SPARSE,3)
#         #else:
#         #    point = np.asarray(load_data['sample_vec']).reshape(-1,knn,3)

#         #lables.append(lable)
#         pointclouds.append(point)
#         samples.append(sample)
# lables = np.asarray(lables)
# pointclouds = np.asarray(pointclouds)
# samples = np.asarray(samples)
# print('data shape:',pointclouds.shape,samples.shape,lables.shape)

# ply_vertex = np.zeros(pointclouds.shape[2], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
# for i in range(pointclouds.shape[2]):
#     ply_vertex[i] = (pointclouds[0,0,i,0], pointclouds[0,0,i,1], pointclouds[0,0,i,2])
# el = PlyElement.describe(ply_vertex, 'vertex')
# PlyData([el], text=True).write('./vis.ply')

#feature_object = tf.placeholder(tf.float32, shape=[None,SHAPE_NUM])
feature_object = tf.placeholder(tf.float32, shape=[POINT_NUM,test_num])
input_points_3d = tf.placeholder(tf.float32, shape=[POINT_NUM,3])
points_target_num = tf.placeholder(tf.int32, shape=[1,1])
points_input_num = tf.placeholder(tf.int32, shape=[1,1])
dis_points_lable = tf.reshape(next_element[1],[POINT_NUM,1])
dis_points_3d = tf.reshape(next_element[0],[POINT_NUM,3])
dis_knn_3d = tf.reshape(next_element[2],[POINT_NUM,knn,3])
#dis_knn_3d = tf.placeholder(tf.float32, shape=[POINT_NUM,knn,3])
points_target_sparse = tf.placeholder(tf.float32, shape=[1,a.INPUT_NUM,3])

    
def pointnet(point_set):
    with tf.variable_scope('pointnet', reuse=tf.AUTO_REUSE):
        point_set = tf.reshape(point_set,[-1,knn*3])
        feature_f = tf.nn.relu(tf.layers.dense(point_set,512))
        print('feature_f:',feature_f)
        net = feature_f
        with tf.variable_scope('point_decoder', reuse=tf.AUTO_REUSE):
            for i in range(8):
                with tf.variable_scope("resnetBlockFC_%d" % i ):
                    net = tf.layers.dense(tf.nn.relu(net),512)
                    
        feature = tf.layers.dense(tf.nn.relu(net),512)
        print('pointnet:',feature)
        return feature
    
def local_decoder(query_3d):
    with tf.variable_scope('dis', reuse=tf.AUTO_REUSE):
        feature_f = tf.nn.relu(tf.layers.dense(query_3d,512))
        net = feature_f
        with tf.variable_scope('dis_decoder', reuse=tf.AUTO_REUSE):
            for i in range(8):
                with tf.variable_scope("resnetBlockFC_%d" % i ):
                    net = tf.layers.dense(tf.nn.relu(net),512)
                    
        dis_udf = tf.nn.relu(tf.layers.dense(tf.nn.relu(net),1))
        return dis_udf

def g_decoder(feature_g,input_points_3d_g): 
    with tf.variable_scope('global', reuse=tf.AUTO_REUSE):

        feature_f = tf.nn.relu(tf.layers.dense(feature_g,128))
        net = tf.nn.relu(tf.layers.dense(input_points_3d_g, 512))
        net = tf.concat([net,feature_f],1)
        print('net:',net)
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            for i in range(8):
                with tf.variable_scope("resnetBlockFC_%d" % i ):
                    b_initializer=tf.constant_initializer(0.0)
                    w_initializer = tf.random_normal_initializer(mean=0.0,stddev=np.sqrt(2) / np.sqrt(512))
                    net = tf.layers.dense(tf.nn.relu(net),512,kernel_initializer=w_initializer,bias_initializer=b_initializer)
                    
        b_initializer=tf.constant_initializer(-0.5)
        w_initializer = tf.random_normal_initializer(mean=2*np.sqrt(np.pi) / np.sqrt(512), stddev = 0.000001)
        print('net:',net)
        sdf = tf.layers.dense(tf.nn.relu(net),1,kernel_initializer=w_initializer,bias_initializer=b_initializer)
        grad = tf.gradients(ys=sdf, xs=input_points_3d_g) 
        print('grad',grad)
        print(grad[0])
        normal_p_lenght = tf.expand_dims(safe_norm(grad[0],axis = -1),-1)
        print('normal_p_lenght',normal_p_lenght)
        grad_norm = grad[0]/(normal_p_lenght + 1e-12)
        print('grad_norm',grad_norm)
        
        g_points = input_points_3d_g - sdf * grad_norm
        return sdf, g_points


sdf, g_points = g_decoder(feature_object,input_points_3d)
print('g_points:',g_points)

g_points_batch = tf.reshape(g_points,[1,-1,3])
dis_m = pairwise_distance_gt(g_points_batch,points_target_sparse)
near_idx, _ = knn_extractor(dis_m, knn)
g_points_knn = get_neighbors(points_target_sparse, near_idx)
print('g_points_knn:',g_points_knn)
g_points_knn = tf.reshape(g_points_knn,[-1,knn,3])
print('g_points_knn:',g_points_knn)
rotate_p = tf.tile(tf.reshape(g_points,[POINT_NUM,1,3]),(1,knn,1))
print('rotate_p:',rotate_p)
rotate_p = rotate_p - g_points_knn





rotated = tf.reshape(rotate_p,[POINT_NUM,knn,3])

# gen_knn_vec = tf.tile(tf.expand_dims(g_points, 1), (1, knn, 1)) - gen_knn_3d
# print('gen_knn_vec:',gen_knn_vec)
# gen_knn_vec = tf.reshape(gen_knn_vec,[-1,knn,3])
# print('gen_knn_vec:',gen_knn_vec)
feature_knn_np = pointnet(rotated)
query_moved_dis = local_decoder(feature_knn_np)
#query_moved_dis = tf.abs(query_moved_dis - 0.005)
loss_move = tf.reduce_mean(tf.abs(sdf))
loss_sdf = tf.reduce_mean(query_moved_dis) 
loss = loss_sdf + 0.2*loss_move
#loss = tf.reduce_mean(1/(1+tf.exp(-query_moved_dis))-0.5)

g_points_batch_i = tf.reshape(input_points_3d,[1,-1,3])
dis_m = pairwise_distance_gt(g_points_batch_i,points_target_sparse)
near_idx, _ = knn_extractor(dis_m, knn)
g_points_knn_i = get_neighbors(points_target_sparse, near_idx)
print('g_points_knn:',g_points_knn_i)
g_points_knn_i = tf.reshape(g_points_knn_i,[-1,knn,3])
print('g_points_knn:',g_points_knn_i)
rotate_p_i = tf.tile(tf.reshape(input_points_3d,[POINT_NUM,1,3]),(1,knn,1))
print('rotate_p:',rotate_p_i)
rotate_p_i = rotate_p_i - g_points_knn_i
rotated_i = tf.reshape(rotate_p_i,[POINT_NUM,knn,3])
feature_knn_np_i = pointnet(rotated_i)
query_moved_dis_i = local_decoder(feature_knn_np_i)

g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global')
optim = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9)
loss_grads_and_vars = optim.compute_gradients(loss, var_list=g_vars)
loss_optim = optim.apply_gradients(loss_grads_and_vars)

feature_knn = pointnet(dis_knn_3d)
udf = local_decoder(feature_knn)
loss_dis = tf.losses.mean_squared_error(dis_points_lable, udf)
#loss_dis = tf.losses.absolute_difference(dis_points_lable, udf)

t_vars = tf.trainable_variables()
optim_dis = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9)
loss_grads_and_vars_dis = optim_dis.compute_gradients(loss_dis, var_list=t_vars)
loss_optim_dis = optim.apply_gradients(loss_grads_and_vars_dis)

config = tf.ConfigProto(allow_soft_placement=False) 

# config = tf.ConfigProto(allow_soft_placement=True) 

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)  

# config=tf.ConfigProto(gpu_options=gpu_options)

saver_restore = tf.train.Saver(var_list=t_vars)
saver = tf.train.Saver(max_to_keep=2000000)



with tf.Session(config=config) as sess:
    feature_bs = []
    
    
    for i in range(test_num):
        t = np.zeros(test_num)
        t[i] = 1
        feature_bs.append(t)
        
    feature_bs = np.asarray(feature_bs)
    print('feature_bs:',feature_bs.shape)
    
    if(a.dis):
        sess.run(tf.global_variables_initializer())
        if(a.finetune_dir != 'no_finetune'):
            print('finetune')
            saver.restore(sess, a.finetune_dir)
        print('dis train',SHAPE_NUM)
        epoch_batch = 1000
        #epoch_batch = 40
        #SHAPE_NUM = 4
        for i in range(15000):
            epoch_index = np.random.choice(SHAPE_NUM, epoch_batch, replace = False)
            ini_data_path_epoch = []
            for fi in range(epoch_batch):
                ini_data_path_epoch.append(files_path[epoch_index[fi]])
            sess.run(iterator.initializer, feed_dict={filelist: ini_data_path_epoch})
            loss_i = 0
            for epoch in epoch_index:
                # rt = np.random.choice(samples.shape[1], POINT_NUM, replace = False)
                # input_points_2d_bs = samples[epoch,rt,:].reshape(POINT_NUM, 3)
                # lable_bs = lables[epoch,rt].reshape(POINT_NUM,1)
                # knn_bs = pointclouds[epoch,rt,:,:].reshape(POINT_NUM,knn,3)
                #_,loss_c = sess.run([loss_optim_dis,loss_dis],feed_dict={dis_points_3d:input_points_2d_bs,dis_points_lable:lable_bs,dis_knn_3d:knn_bs})
                _,loss_c,dis_points_lable_c, udf_c = sess.run([loss_optim_dis,loss_dis,dis_points_lable, udf])
                loss_i = loss_i + loss_c
            loss_i = loss_i / epoch_batch
            if(i%5 == 0):
                print('epoch:', i, 'epoch loss:', loss_i)
                #print('dis_points_lable:',dis_points_lable_c[0:3])
                #print('udf:', udf_c[0:3])
            if(i%50 == 0):
                dis_points_lable_c = np.asarray(dis_points_lable_c)
                udf_c = np.asarray(udf_c)
                print(dis_points_lable_c[0],udf_c[0])
                print(dis_points_lable_c[1],udf_c[1])
                print(dis_points_lable_c[2],udf_c[2])
            if(i%200 == 0):
                print('save model')
                saver.save(sess, os.path.join(OUTPUT_DIR, "model"), global_step=i+1)
                
    if(TRAIN):
        print('train start')
        start_time = time.time()

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './pre_train_model/model-10201') 
        load_data = np.load(a.data_dir + a.input_ply_file + '.npz')
    
        samples = np.asarray(load_data['sample']).reshape(1,-1,3)
        pointclouds = np.asarray(load_data['pointcloud_s']).reshape(1,1,-1,3)
        SP_NUM = samples.shape[1]
        feature_bs_t =  np.tile(feature_bs[0,:],[POINT_NUM]).reshape(-1,test_num)
        for i in range(a.epoch):
            #rt = np.random.choice(SP_NUM, POINT_NUM, replace = False)
            index_coarse = np.random.choice(10, 1)
            index_fine = np.random.choice(SP_NUM//10, POINT_NUM, replace = False)
            rt = index_fine * 10 + index_coarse
            input_points_2d_bs = samples[0,rt,:].reshape(POINT_NUM, 3)
            knn_bs = pointclouds[0,0,:,:].reshape(1,-1,3)
            sess.run([loss_optim],feed_dict={input_points_3d:input_points_2d_bs,feature_object:feature_bs_t,points_target_sparse:knn_bs})
            if(i%500 == 0):
                _,loss_c,g_points_c,loss_move_c,loss_sdf_c = sess.run([loss_optim,loss,g_points,loss_move,loss_sdf],feed_dict={input_points_3d:input_points_2d_bs,feature_object:feature_bs_t,
                                                                                                                            points_target_sparse:knn_bs})
                print('epoch:', i, 'epoch loss:', loss_c,'loss_sdf:',loss_sdf_c, 'move loss:',loss_move_c)

        print('save model')
        saver.save(sess, os.path.join(OUTPUT_DIR, "model"), global_step=0)
        end_time = time.time()
        print('run_time:',end_time-start_time)
    if(a.test):
        print('test start')
       
        
        s = np.arange(-bd,bd, (2*bd)/128)
            
        print(s.shape[0])
        vox_size = s.shape[0]
        POINT_NUM_GT_bs = np.array(vox_size).reshape(1,1)
        points_input_num_bs = np.array(POINT_NUM).reshape(1,1)
        
        
        
        
        POINT_NUM_GT_bs = np.array(vox_size*vox_size).reshape(1,1)

        sess.run(tf.global_variables_initializer())

        saver.restore(sess, a.out_dir + 'model-0')
        
        point_sparse = np.load(a.data_dir + a.input_ply_file + '.npz')['pointcloud_s']
        

        
        input_points_2d_bs = []

        bd_max = [np.max(point_sparse[:,0]), np.max(point_sparse[:,1]), np.max(point_sparse[:,2])] 
        bd_min = [np.min(point_sparse[:,0]), np.min(point_sparse[:,1]),np.min(point_sparse[:,2])] 
        bd_max  = np.asarray(bd_max) + 0.05
        bd_min = np.asarray(bd_min) - 0.05
        sx = np.arange(bd_min[0], bd_max[0], (bd_max[0] - bd_min[0])/vox_size)
        sy = np.arange(bd_min[1], bd_max[1], (bd_max[1] - bd_min[1])/vox_size)
        sz = np.arange(bd_min[2], bd_max[2], (bd_max[2] - bd_min[2])/vox_size)
        print(bd_max)
        print(bd_min)
        for i in sx:
            for j in sy:
                for k in sz:
                    input_points_2d_bs.append(np.asarray([i,j,k]))
        input_points_2d_bs = np.asarray(input_points_2d_bs)
        input_points_2d_bs = input_points_2d_bs.reshape((-1,POINT_NUM,3))
                    
        vox = []
        feature_bs = []
        moved_points = []
        for j in range(POINT_NUM):
            t = np.zeros(test_num)
            t[0] = 1
            feature_bs.append(t)
        feature_bs = np.asarray(feature_bs)
        for i in range(input_points_2d_bs.shape[0]):
            
            input_points_2d_bs_t = input_points_2d_bs[i,:,:].reshape(POINT_NUM, 3)
            feature_bs_t = feature_bs.reshape(POINT_NUM,test_num)
            sdf_c = sess.run([sdf],feed_dict={input_points_3d:input_points_2d_bs_t,feature_object:feature_bs_t})
            #sdf_c = sess.run([sdf],feed_dict={input_points_3d:input_points_2d_bs_t,feature_object:feature_bs_t,points_target_num:POINT_NUM_GT_bs,points_input_num:points_input_num_bs})
            vox.append(sdf_c)

            
        vox = np.asarray(vox)
        #vis_single_points(moved_points, 'moved_points.ply')
        #print('vox',np.min(vox),np.max(vox),np.mean(vox))
        vox = vox.reshape((vox_size,vox_size,vox_size))
        vox_max = np.max(vox.reshape((-1)))
        vox_min = np.min(vox.reshape((-1)))
        print('max_min:',vox_max,vox_min,np.mean(vox))
        
        #threshs = [0.001,0.0015,0.002,0.0025,0.005]
        threshs = [0.005]
        for thresh in threshs:
            print(np.sum(vox>thresh),np.sum(vox<thresh))
            
            if(np.sum(vox>0.0)<np.sum(vox<0.0)):
                thresh = -thresh
            #vertices, triangles = libmcubes.marching_cubes(vox, thresh)
            #vertices, triangles = mcubes.marching_cubes(vox, thresh)
            vertices, triangles, _, _ = marching_cubes_lewiner(vox, thresh)
            if(vertices.shape[0]<10 or triangles.shape[0]<10):
                print('no sur---------------------------------------------')
                continue
            if(np.sum(vox>0.0)>np.sum(vox<0.0)):
                triangles_t = []
                for it in range(triangles.shape[0]):
                    tt = np.array([triangles[it,2],triangles[it,1],triangles[it,0]])
                    triangles_t.append(tt)
                triangles_t = np.asarray(triangles_t)
            else:
                triangles_t = triangles
                triangles_t = np.asarray(triangles_t)

            vertices -= 0.5
            # Undo padding
            vertices -= 1
            # Normalize to bounding box
            vertices /= np.array([vox_size-1, vox_size-1, vox_size-1])
            vertices = (bd_max-bd_min) * vertices + bd_min
            mesh = trimesh.Trimesh(vertices, triangles_t,
                            vertex_normals=None,
                            process=False)
            
            
            loc_data = np.load(a.data_dir + a.input_ply_file + '.npz')
            vertices = vertices * loc_data['scal'] + loc_data['trans']
            mesh = trimesh.Trimesh(vertices, triangles_t,
                                vertex_normals=None,
                                process=False)
            mesh.export(OUTPUT_DIR +  '/OSP_' + a.input_ply_file + '_'+ str(thresh) + '.off')
                

                    
    
    