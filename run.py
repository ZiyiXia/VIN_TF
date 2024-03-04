import time
import argparse

import numpy as np
import tensorflow as tf

from model import *
from dataset import *

# @tf.function
def train_or_eval(model, dataset, args, train_mode):

    num_batches = dataset.num_examples // args.batch_size
    total_examples = num_batches * args.batch_size

    total_err = 0.0
    total_loss = 0.0
    
    for _ in range(num_batches):
        # X, S1, S2, y = feed_ops
        X_batch, S1_batch, S2_batch, y_batch = dataset.next_batch(args.batch_size)

        optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=args.lr, epsilon=1e-6, centered=True)

        with tf.GradientTape() as tape:
            logits = model(X_batch, S1_batch, S2_batch, training=train_mode)  # Logits for this minibatch

            # compute the probability and then select actions
            prob_actions = tf.nn.softmax(logits, name='probability_actions')
            actions = tf.math.argmax(prob_actions, axis=1)

            # number of error predictions
            err = tf.math.reduce_sum(tf.cast(tf.not_equal(y_batch, actions), tf.float32))

            y = tf.cast(y_batch, dtype=tf.int32)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name='cross_entropy')
            loss = tf.math.reduce_sum(cross_entropy, name='cross_entropy_sum')

        # grads = tape.gradient(cross_entropy, model.trainable_weights)
        # optimizer.apply_gradients(zip(grads, model.trainable_weights))
        optimizer.minimize(cross_entropy, model.trainable_weights, tape=tape)

        total_err += err
        total_loss += loss

    return total_err/total_examples, total_loss/total_examples


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

# load the datasets
trainset = Dataset(args.datafile, mode='train', imsize=args.imsize)
testset = Dataset(args.datafile, mode='test', imsize=args.imsize)

model = VIN(args)

# start training
for epoch in range(args.epochs):
    print("==============================")

    start_time = time.time()

    mean_err, mean_loss = train_or_eval(model,
                                        trainset, 
                                        args,
                                        train_mode=True)
    
    time_duration = time.time() - start_time
    out_str = 'Epoch: {:3d} ({:.1f} s): \n\t Train Loss: {:.5f} \t Train Err: {:.5f}'
    print(out_str.format(epoch, time_duration, mean_loss, mean_err))

    print('\n Finished training...\n ')
    
    # Testing
    print('\n Testing...\n')
    
    mean_err, mean_loss = train_or_eval(model,
                                        testset, 
                                        args, 
                                        train_mode=False)
    
    print('Test Accuracy: {:.2f}%\n'.format(100*(1 - mean_err)))





    # # @tf.function
# def process(model, X, S1, S2, y, args, train_mode):
#     logits = model(X, S1, S2)

#     # compute the loss
#     y = tf.cast(y, dtype=tf.int32)
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name='cross_entropy')
#     loss = tf.math.reduce_sum(cross_entropy, name='cross_entropy_sum')

#     # optimizer
#     optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=args.lr, epsilon=1e-6, centered=True)

#     if train_mode:
#         optimizer.minimize(loss)

#     # compute the probability and then select actions
#     prob_actions = tf.nn.softmax(logits, name='probability_actions')
#     actions = tf.math.argmax(prob_actions, axis=1)

#     # number of error predictions
#     num_err = tf.math.reduce_sum(tf.cast(tf.not_equal(y, actions), tf.float32))

#     return num_err, loss