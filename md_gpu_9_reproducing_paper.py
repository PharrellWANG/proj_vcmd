# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import math
# from tensorflow.contrib.learn.python.learn.datasets.mode_decision import read_data_sets
from a_vcmd_read_data_multiple_test_set import read_data_sets
from datetime import datetime

tf.set_random_seed(0.0)

# Download images and labels into mode_decision.test (10K images+labels) and mode_decision.train (60K images+labels)
# mode_decision = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
mode_decision = read_data_sets(reshape=False, validation_size=0)
# number of categories
cats = 37
block_width = 16
# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                    X [batch, 16, 16, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer +BN 4x4x1=>96 stride 1      W1 [4, 4, 1, 96 ]        B1 [96]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                              Y1 [batch, 16, 16, 6]
#   @ @ @ @ @ @ @ @     -- conv. layer +BN 5x5x6=>256 stride 1      W2 [5, 5, 6, 256]        B2 [48]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                                Y2 [batch, 14, 14, 12]
#     @ @ @ @ @ @       -- conv. layer +BN 4x4x12=>256 stride 2     W3 [4, 4, 12, 256]       B3 [64]
#     ∶∶∶∶∶∶∶∶∶∶∶                                                  Y3 [batch, 7, 7, 24] => reshaped to YY [batch, 7*7*24]
#      \x/x\x\x/ ✞      -- fully connected layer (relu+dropout+BN) W4 [7*7*24, 200]       B4 [200]
#       · · · ·                                                    Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)         W5 [200, 10]           B5 [10]
#        · · ·                                                     Y [batch, 20]


def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,
                                                       iteration)  # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_everages


def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()


def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])
    return noiseshape


def training_step(i, update_test_data, update_train_data, loop_start_stamp):
    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mode_decision.train.next_batch(100)

    # learning rate decay
    max_learning_rate = 0.05
    min_learning_rate = 0.01
    decay_speed = 1500
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)

    if update_train_data:
        a, c = sess.run([accuracy, cross_entropy], {X: batch_X, Y_: batch_Y, tst: False, pkeep: 1.0, pkeep_conv: 1.0})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")

    if update_test_data:
        save_path = saver.save(sess,
                               "/Users/Pharrell_WANG/PycharmProjects/proj_vcmd/ckpt_9/"
                               "md_9.ckpt")
        print("Model saved in file: %s" % save_path)

        stamp = datetime.now()

        time_passed_since_loop_start = stamp - loop_start_stamp
        print("")
        print("--------->> *** *** -------->>                    loop started at  : " + str(loop_start_stamp))
        print('--------->> *** *** -------->>                    now the time is  : ' + str(stamp))
        print(
            "--------->> *** *** -------->>        time passed since beginning  : " + str(time_passed_since_loop_start))
        print("")
        # test_size = 74000
        sum_a = 0
        sum_c = 0

        for step in range(740):
            batch_test_x, batch_test_y = mode_decision.test.next_batch(100)
            a, c = sess.run([accuracy, cross_entropy],
                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,
                                       pkeep_conv: 1.0})
            sum_a += a
            sum_c += c

        a = sum_a / 740
        c = sum_c / 740
        print(str(i) + ": ********* epoch " + str(
            i * 100 // mode_decision.train.images.shape[0] + 1) +
              " ********* test accuracy for ->    ALL 37 modes   <-: " + str(a) + " test loss: " + str(
            c))

        sum_a = 0
        sum_c = 0
        for step in range(740):
            batch_test_x, batch_test_y = mode_decision.test0.next_batch(100)
            a, c = sess.run([accuracy, cross_entropy],
                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,
                                       pkeep_conv: 1.0})
            sum_a += a
            sum_c += c

        a = sum_a / 740
        c = sum_c / 740
        print(str(i) + ": ********* epoch " + str(
            i * 100 // mode_decision.train.images.shape[0] + 1) +
              " ********* test accuracy for ---->>   mode 0    <---: " + str(a) + " test loss: " + str(
            c))

        sum_a = 0
        sum_c = 0
        for step in range(740):
            batch_test_x, batch_test_y = mode_decision.test1.next_batch(100)
            a, c = sess.run([accuracy, cross_entropy],
                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,
                                       pkeep_conv: 1.0})
            sum_a += a
            sum_c += c

        a = sum_a / 740
        c = sum_c / 740
        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +
              " ********* test accuracy for ---->>   mode 1    <---: " + str(a) + " test loss: " + str(c))

        sum_a = 0
        sum_c = 0
        for step in range(740):
            batch_test_x, batch_test_y = mode_decision.test2.next_batch(100)
            a, c = sess.run([accuracy, cross_entropy],
                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,
                                       pkeep_conv: 1.0})
            sum_a += a
            sum_c += c

        a = sum_a / 740
        c = sum_c / 740
        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +
              " ********* test accuracy for ---->>   mode 2    <---: " + str(a) + " test loss: " + str(c))

        #######################
        sum_a = 0
        sum_c = 0
        for step in range(740):
            batch_test_x, batch_test_y = mode_decision.test24.next_batch(100)
            a, c = sess.run([accuracy, cross_entropy],
                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,
                                       pkeep_conv: 1.0})
            sum_a += a
            sum_c += c

        a = sum_a / 740
        c = sum_c / 740
        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +
              " ********* test accuracy for ---->>   mode 24   <---: " + str(a) + " test loss: " + str(c))

        #######################
        sum_a = 0
        sum_c = 0
        for step in range(740):
            batch_test_x, batch_test_y = mode_decision.test25.next_batch(100)
            a, c = sess.run([accuracy, cross_entropy],
                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,
                                       pkeep_conv: 1.0})
            sum_a += a
            sum_c += c

        a = sum_a / 740
        c = sum_c / 740
        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +
              " ********* test accuracy for ---->>   mode 25   <---: " + str(a) + " test loss: " + str(c))

        #######################
        sum_a = 0
        sum_c = 0
        for step in range(740):
            batch_test_x, batch_test_y = mode_decision.test26.next_batch(100)
            a, c = sess.run([accuracy, cross_entropy],
                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,
                                       pkeep_conv: 1.0})
            sum_a += a
            sum_c += c

        a = sum_a / 740
        c = sum_c / 740
        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +
              " ********* test accuracy for ---->>   mode 26   <---: " + str(a) + " test loss: " + str(c))

        #######################
        sum_a = 0
        sum_c = 0
        for step in range(740):
            batch_test_x, batch_test_y = mode_decision.test27.next_batch(100)
            a, c = sess.run([accuracy, cross_entropy],
                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,
                                       pkeep_conv: 1.0})
            sum_a += a
            sum_c += c

        a = sum_a / 740
        c = sum_c / 740
        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +
              " ********* test accuracy for ---->>   mode 27   <---: " + str(a) + " test loss: " + str(c))

        #######################
        sum_a = 0
        sum_c = 0
        for step in range(740):
            batch_test_x, batch_test_y = mode_decision.test28.next_batch(100)
            a, c = sess.run([accuracy, cross_entropy],
                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,
                                       pkeep_conv: 1.0})
            sum_a += a
            sum_c += c

        a = sum_a / 740
        c = sum_c / 740
        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +
              " ********* test accuracy for ---->>   mode 28   <---: " + str(a) + " test loss: " + str(c))

        #######################
        sum_a = 0
        sum_c = 0
        for step in range(740):
            batch_test_x, batch_test_y = mode_decision.test29.next_batch(100)
            a, c = sess.run([accuracy, cross_entropy],
                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,
                                       pkeep_conv: 1.0})
            sum_a += a
            sum_c += c

        a = sum_a / 740
        c = sum_c / 740
        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +
              " ********* test accuracy for ---->>   mode 29   <---: " + str(a) + " test loss: " + str(c))

        #######################
        sum_a = 0
        sum_c = 0
        for step in range(740):
            batch_test_x, batch_test_y = mode_decision.test30.next_batch(100)
            a, c = sess.run([accuracy, cross_entropy],
                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,
                                       pkeep_conv: 1.0})
            sum_a += a
            sum_c += c

        a = sum_a / 740
        c = sum_c / 740
        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +
              " ********* test accuracy for ---->>   mode 30   <---: " + str(a) + " test loss: " + str(c))

        #######################
        sum_a = 0
        sum_c = 0
        for step in range(740):
            batch_test_x, batch_test_y = mode_decision.test31.next_batch(100)
            a, c = sess.run([accuracy, cross_entropy],
                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,
                                       pkeep_conv: 1.0})
            sum_a += a
            sum_c += c

        a = sum_a / 740
        c = sum_c / 740
        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +
              " ********* test accuracy for ---->>   mode 31   <---: " + str(a) + " test loss: " + str(c))

        #######################
        sum_a = 0
        sum_c = 0
        for step in range(740):
            batch_test_x, batch_test_y = mode_decision.test32.next_batch(100)
            a, c = sess.run([accuracy, cross_entropy],
                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,
                                       pkeep_conv: 1.0})
            sum_a += a
            sum_c += c

        a = sum_a / 740
        c = sum_c / 740
        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +
              " ********* test accuracy for ---->>   mode 32   <---: " + str(a) + " test loss: " + str(c))

        #######################
        sum_a = 0
        sum_c = 0
        for step in range(740):
            batch_test_x, batch_test_y = mode_decision.test33.next_batch(100)
            a, c = sess.run([accuracy, cross_entropy],
                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,
                                       pkeep_conv: 1.0})
            sum_a += a
            sum_c += c

        a = sum_a / 740
        c = sum_c / 740
        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +
              " ********* test accuracy for ---->>   mode 33   <---: " + str(a) + " test loss: " + str(c))

        #######################
        sum_a = 0
        sum_c = 0
        for step in range(740):
            batch_test_x, batch_test_y = mode_decision.test34.next_batch(100)
            a, c = sess.run([accuracy, cross_entropy],
                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,
                                       pkeep_conv: 1.0})
            sum_a += a
            sum_c += c

        a = sum_a / 740
        c = sum_c / 740
        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +
              " ********* test accuracy for ---->>   mode 34   <---: " + str(a) + " test loss: " + str(c))

        #######################
        sum_a = 0
        sum_c = 0
        for step in range(740):
            batch_test_x, batch_test_y = mode_decision.test35.next_batch(100)
            a, c = sess.run([accuracy, cross_entropy],
                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,
                                       pkeep_conv: 1.0})
            sum_a += a
            sum_c += c

        a = sum_a / 740
        c = sum_c / 740
        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +
              " ********* test accuracy for ---->>   mode 35   <---: " + str(a) + " test loss: " + str(c))

        #######################
        sum_a = 0
        sum_c = 0
        for step in range(740):
            batch_test_x, batch_test_y = mode_decision.test36.next_batch(100)
            a, c = sess.run([accuracy, cross_entropy],
                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,
                                       pkeep_conv: 1.0})
            sum_a += a
            sum_c += c

        a = sum_a / 740
        c = sum_c / 740
        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +
              " ********* test accuracy for ---->>   mode 36   <---: " + str(a) + " test loss: " + str(c))

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False, pkeep: 0.75, pkeep_conv: 1.0})
    sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i, pkeep: 1.0, pkeep_conv: 1.0})


# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
with tf.device("/gpu:0"):
    X = tf.placeholder(tf.float32, [None, block_width, block_width, 1])
    # correct answers will go here
    Y_ = tf.placeholder(tf.float32, [None, cats])
    # variable learning rate
    lr = tf.placeholder(tf.float32)
    # test flag for batch norm
    tst = tf.placeholder(tf.bool)
    iter = tf.placeholder(tf.int32)
    # dropout probability
    pkeep = tf.placeholder(tf.float32)
    pkeep_conv = tf.placeholder(tf.float32)

    # three convolutional layers with their channel counts, and a
    # fully connected layer (tha last layer has 37 softmax neurons)
    K = 96  # first convolutional layer output depth
    L = 256  # second convolutional layer output depth
    M = 256  # third convolutional layer
    N = 1024  # fully connected layer

    W1 = tf.Variable(tf.truncated_normal([4, 4, 1, K], stddev=0.01))  # 6x6 patch, 1 input channel, K output channels
    B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
    W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.01))
    B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
    W3 = tf.Variable(tf.truncated_normal([3, 3, L, M], stddev=0.01))
    B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

    W4 = tf.Variable(tf.truncated_normal([8 * 8 * M, N], stddev=0.005))
    B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
    W5 = tf.Variable(tf.truncated_normal([N, cats], stddev=0.01))
    B5 = tf.Variable(tf.constant(0.1, tf.float32, [cats]))

    # The model
    # batch norm scaling is not useful with relus
    # batch norm offsets are used instead of biases
    stride = 1  # output is 16*16
    Y1l = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')
    Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1, convolutional=True)
    Y1r = tf.nn.relu(Y1bn)
    Y1 = tf.nn.dropout(Y1r, pkeep_conv, compatible_convolutional_noise_shape(Y1r))
    stride = 1  # output is 16*16
    Y2l = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
    Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2, convolutional=True)
    Y2r = tf.nn.relu(Y2bn)
    Y2 = tf.nn.dropout(Y2r, pkeep_conv, compatible_convolutional_noise_shape(Y2r))
    stride = 2  # output is 8*8
    Y3l = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
    Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3, convolutional=True)
    Y3r = tf.nn.relu(Y3bn)
    Y3 = tf.nn.dropout(Y3r, pkeep_conv, compatible_convolutional_noise_shape(Y3r))

    # reshape the output from the third convolution for the fully connected layer
    YY = tf.reshape(Y3, shape=[-1, 8 * 8 * M])

    Y4l = tf.matmul(YY, W4)
    Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4)
    Y4r = tf.nn.relu(Y4bn)
    Y4 = tf.nn.dropout(Y4r, pkeep)
    Ylogits = tf.matmul(Y4, W5) + B5
    Y = tf.nn.softmax(Ylogits)

    update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)

    # cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
    # TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
    # problems with log(0) which is NaN
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy) * 100

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # matplotlib visualisation
    allweights = tf.concat(
        [tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])],
        0)
    allbiases = tf.concat(
        [tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])],
        0)
    conv_activations = tf.concat([tf.reshape(tf.reduce_max(Y1r, [0]), [-1]), tf.reshape(tf.reduce_max(Y2r, [0]), [-1]),
                                  tf.reshape(tf.reduce_max(Y3r, [0]), [-1])], 0)
    dense_activations = tf.reduce_max(Y4r, [0])

    # training step, the learning rate is a placeholder
    # global_step = tf.Variable(0, name='global_step', trainable=False)
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    # init
    init = tf.global_variables_initializer()

saver = tf.train.Saver()
# define the session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# run init operation
sess.run(init)
print("======------------>>>>>>>    " + str(mode_decision.train.images.shape[0]))
print("initialized successfully! -------------------------------- ")
loop_start = datetime.now()

for i in range(10000000 + 1):
    training_step(i, i % 10000 == 0, i % 20 == 0, loop_start_stamp=loop_start)
