import time
import argparse

import numpy as np
import tensorflow as tf

from model import *
from dataset import *

parser = argparse.ArgumentParser()

parser.add_argument('--datafile',
                    type=str,
                    default='../data/gridworld_8x8.npz',
                    help='Path to data file')
parser.add_argument('--imsize', 
                    type=int, 
                    default=8, 
                    help='Size of image')
parser.add_argument('--lr', 
                    type=float, 
                    default=0.002, 
                    help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
parser.add_argument('--epochs', 
                    type=int, 
                    default=30, 
                    help='Number of epochs to train')
parser.add_argument('--k', 
                    type=int, 
                    default=10, 
                    help='Number of Value Iterations')
parser.add_argument('--ch_i', 
                    type=int, 
                    default=2, 
                    help='Number of channels in input layer')
parser.add_argument('--ch_h', 
                    type=int, 
                    default=150, 
                    help='Number of channels in first hidden layer')
parser.add_argument('--ch_q',
                    type=int,
                    default=10, 
                    help='Number of channels in q layer (~actions) in VI-module')
parser.add_argument('--batch_size',
                    type=int, 
                    default=128, 
                    help='Batch size')
parser.add_argument('--use_log',
                    type=bool, 
                    default=False, 
                    help='True to enable TensorBoard summary')
parser.add_argument('--logdir',
                    type=str, 
                    default='.log/', 
                    help='Directory to store TensorBoard summary')

args = parser.parse_args()

# Input tensor: Stack obstacle image and goal image, i.e. ch_i = 2
X = tf.keras.Input(shape=(args.imsize, args.imsize, args.ch_i, ), name='X', dtype=tf.float32, )
# Input batches of vertical positions
S1 = tf.keras.Input(shape=(1, ), name='S1', dtype=tf.int32)
# Input batches of horizontal positions
S2 = tf.keras.Input(shape=(1, ), name='S2', dtype=tf.int32)
# Labels: actions {0,...,7}
y = tf.keras.Input(shape=(1, ), name='y', dtype=tf.int32)

# call the VIN model
logits, prob_actions = VIN(X, S1, S2, args)

# compute the loss
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name='cross_entropy')
loss = tf.math.reduce_sum(cross_entropy, name='cross_entropy_sum')

# optimizer
optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=args.lr, epsilon=1e-6, centered=True)

train_step = optimizer.minimize(loss)

# use the probability to select actions
actions = tf.math.argmax(prob_actions, axis=1)

# number of error predictions
num_err = tf.math.reduce_sum(tf.cast(tf.not_equal(y, actions), tf.float32))

# load the datasets
trainset = Dataset(args.datafile, mode='train', imsize=args.imsize)
testset = Dataset(args.datafile, mode='test', imsize=args.imsize)

