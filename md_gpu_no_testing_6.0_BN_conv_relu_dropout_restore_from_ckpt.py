# encoding: UTF-8import tensorflow as tfimport mathfrom a_vcmd_read_data import read_data_setstf.set_random_seed(0)mode_decision = read_data_sets(reshape=False, validation_size=0)image_width = 16def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,                                                       iteration)  # adding the iteration prevents from averaging across non-existing iterations    bnepsilon = 1e-5    if convolutional:        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])    else:        mean, variance = tf.nn.moments(Ylogits, [0])    update_moving_everages = exp_moving_avg.apply([mean, variance])    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)    return Ybn, update_moving_everagesdef no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):    return Ylogits, tf.no_op()def compatible_convolutional_noise_shape(Y):    noiseshape = tf.shape(Y)    noiseshape = noiseshape * tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])    return noiseshapewith tf.device("/gpu:0"):    X = tf.placeholder(tf.float32, [None, image_width, image_width, 1])    # correct answers go here    Y_ = tf.placeholder(tf.float32, [None, 37])    # learning rate    lr = tf.placeholder(tf.float32)    # test flag for batch norm    tst = tf.placeholder(tf.bool)    iter = tf.placeholder(tf.int32)    # dropout probability    pkeep = tf.placeholder(tf.float32)    pkeep_conv = tf.placeholder(tf.float32)    CONV_K = 60  # first conv layer output depth    CONV_L = 120  # second conv layer output depth    CONV_M = 240  # third conv layer output depth    L = 1000  # 1st FC    M = 800  # 2nd FC    N = 600  # 3rd FC    O = 300  # 4th FC    P = 150  # 5th FC    Q = 60  # 6th FC    R = 37  # last layer which is softmax for 37 modes classification    # 5x5 patch, 1 input channel, conv_K output channels    CONV_W1 = tf.Variable(tf.truncated_normal([5, 5, 1, CONV_K], stddev=0.1))    CONV_B1 = tf.Variable(tf.constant(0.1, tf.float32, [CONV_K]))    CONV_W2 = tf.Variable(tf.truncated_normal([3, 3, CONV_K, CONV_L], stddev=0.1))    CONV_B2 = tf.Variable(tf.constant(0.1, tf.float32, [CONV_L]))    CONV_W3 = tf.Variable(tf.truncated_normal([3, 3, CONV_L, CONV_M], stddev=0.1))    CONV_B3 = tf.Variable(tf.constant(0.1, tf.float32, [CONV_M]))    # truncated normal    W1 = tf.Variable(tf.truncated_normal([4 * 4 * CONV_M, L], stddev=0.1))    B1 = tf.Variable(tf.constant(0.1, tf.float32, [L]))    W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))    B2 = tf.Variable(tf.constant(0.1, tf.float32, [M]))    W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))    B3 = tf.Variable(tf.constant(0.1, tf.float32, [N]))    W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))    B4 = tf.Variable(tf.constant(0.1, tf.float32, [O]))    W5 = tf.Variable(tf.truncated_normal([O, P], stddev=0.1))    B5 = tf.Variable(tf.constant(0.1, tf.float32, [P]))    W6 = tf.Variable(tf.truncated_normal([P, Q], stddev=0.1))    B6 = tf.Variable(tf.constant(0.1, tf.float32, [Q]))    W7 = tf.Variable(tf.truncated_normal([Q, R], stddev=0.1))    B7 = tf.Variable(tf.constant(0.1, tf.float32, [R]))    # The model    # batch norm scaling is not useful with relus    # batch norm offsets are used instead of biases    stride = 1  # output is 16x16    CONV_Y1l = tf.nn.conv2d(X, CONV_W1, strides=[1, stride, stride, 1], padding='SAME')    CONV_Y1bn, CONV_update_ema1 = batchnorm(CONV_Y1l, tst, iter, CONV_B1, convolutional=True)    CONV_Y1r = tf.nn.relu(CONV_Y1bn)    CONV_Y1 = tf.nn.dropout(CONV_Y1r, pkeep_conv, compatible_convolutional_noise_shape(CONV_Y1r))    # Original we used the line below for the version of without batch normalization    # CONV_Y1 = tf.nn.relu(tf.nn.conv2d(X, CONV_W1, strides=[1, stride, stride, 1], padding='SAME') + CONV_B1)    stride = 2  # output is 8x8    CONV_Y2l = tf.nn.conv2d(CONV_Y1, CONV_W2, strides=[1, stride, stride, 1], padding='SAME')    CONV_Y2bn, CONV_update_ema2 = batchnorm(CONV_Y2l, tst, iter, CONV_B2, convolutional=True)    CONV_Y2r = tf.nn.relu(CONV_Y2bn)    CONV_Y2 = tf.nn.dropout(CONV_Y2r, pkeep_conv, compatible_convolutional_noise_shape(CONV_Y2r))    # CONV_Y2 = tf.nn.relu(tf.nn.conv2d(CONV_Y1, CONV_W2, strides=[1, stride, stride, 1], padding='SAME') + CONV_B2)    stride = 2  # output is 4x4    CONV_Y3l = tf.nn.conv2d(CONV_Y2, CONV_W3, strides=[1, stride, stride, 1], padding='SAME')    CONV_Y3bn, CONV_update_ema3 = batchnorm(CONV_Y3l, tst, iter, CONV_B3, convolutional=True)    CONV_Y3r = tf.nn.relu(CONV_Y3bn)    CONV_Y3 = tf.nn.dropout(CONV_Y3r, pkeep_conv, compatible_convolutional_noise_shape(CONV_Y3r))    # CONV_Y3 = tf.nn.relu(tf.nn.conv2d(CONV_Y2, CONV_W3, strides=[1, stride, stride, 1], padding='SAME') + CONV_B3)    # reshape the output from the third convolution for the fully connected layer    YY = tf.reshape(CONV_Y3, shape=[-1, 4 * 4 * CONV_M])    # relu, relu    Y1l = tf.matmul(YY, W1)    Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1)    Y1r = tf.nn.relu(Y1bn)    Y1 = tf.nn.dropout(Y1r, pkeep)    # Y1 = tf.nn.relu(tf.matmul(YY, W1) + B1)    # Y1d = tf.nn.dropout(Y1, pkeep)    Y2l = tf.matmul(Y1, W2)    Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2)    Y2r = tf.nn.relu(Y2bn)    Y2 = tf.nn.dropout(Y2r, pkeep)    # Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)    # Y2d = tf.nn.dropout(Y2, pkeep)    Y3l = tf.matmul(Y2, W3)    Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3)    Y3r = tf.nn.relu(Y3bn)    Y3 = tf.nn.dropout(Y3r, pkeep)    # Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)    # Y3d = tf.nn.dropout(Y3, pkeep)    Y4l = tf.matmul(Y3, W4)    Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4)    Y4r = tf.nn.relu(Y4bn)    Y4 = tf.nn.dropout(Y4r, pkeep)    # Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)    # Y4d = tf.nn.dropout(Y4, pkeep)    Y5l = tf.matmul(Y4, W5)    Y5bn, update_ema5 = batchnorm(Y5l, tst, iter, B5)    Y5r = tf.nn.relu(Y5bn)    Y5 = tf.nn.dropout(Y5r, pkeep)    # Y5 = tf.nn.relu(tf.matmul(Y4d, W5) + B5)    # Y5d = tf.nn.dropout(Y5, pkeep)    Y6l = tf.matmul(Y5, W6)    Y6bn, update_ema6 = batchnorm(Y6l, tst, iter, B6)    Y6r = tf.nn.relu(Y6bn)    Y6 = tf.nn.dropout(Y6r, pkeep)    # Y6 = tf.nn.relu(tf.matmul(Y5d, W6) + B6)    # Y6d = tf.nn.dropout(Y6, pkeep)    # Y = tf.nn.softmax(tf.matmul(Y6, W7) + B7)    Ylogits = tf.matmul(Y6, W7) + B7    Y = tf.nn.softmax(Ylogits)    update_ema = tf.group(CONV_update_ema1, CONV_update_ema2, CONV_update_ema3,                          update_ema1, update_ema2, update_ema3,                          update_ema4, update_ema5, update_ema6)    # cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images    # TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability    # problems with log(0) which is NaN    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)    cross_entropy = tf.reduce_mean(cross_entropy) * 100    # accuracy of the trained model, between 0 (worst) and 1 (best)    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    # learning_rate = 0.005    # optimizer = tf.train.AdamOptimizer(0.005)    optimizer = tf.train.AdamOptimizer(lr)    train_step = optimizer.minimize(cross_entropy)    # init = tf.global_variables_initializer()saver = tf.train.Saver()# define the sessionsess = tf.Session(config=tf.ConfigProto(log_device_placement=True))# run init operation# sess.run(init)saver.restore(sess, "/Users/Pharrell_WANG/PycharmProjects/proj_vcmd/checkpoint/md_gpu_no_test_6_BN_conv_dropout_model.ckpt")print("Model restored.")print(" ")print(" ")print(" ")print("======------------>>>>>>>    " + str(mode_decision.train.images.shape[0]))# now after the init op, we can use the model now. (train the model)def training_step(i, update_train_data, update_test_data):    # load batch of images and correct answers    batch_X, batch_Y = mode_decision.train.next_batch(100)    lrmax = 0.05    lrmin = 0.0001    decay_speed = 1000.0    learning_rate = lrmin + (lrmax - lrmin) * math.exp(-i / decay_speed)    if update_train_data:        a, c = sess.run([accuracy, cross_entropy],                        feed_dict={X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False, pkeep: 1.0, pkeep_conv: 1.0})        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")    if update_test_data:        save_path = saver.save(sess,                               "/Users/Pharrell_WANG/PycharmProjects/proj_vcmd/checkpoint/"                               "md_gpu_no_test_6_BN_conv_dropout_model_restore_from_ckpt.ckpt")        print("Model saved in file: %s" % save_path)        # test_data = {X: mode_decision.test.images, Y_: mode_decision.test.labels}        # a, c = sess.run([accuracy, cross_entropy],        #                 feed_dict={X: mode_decision.test.images, Y_: mode_decision.test.labels, tst: True, pkeep: 1.0,        #                            pkeep_conv: 1.0})        # print(str(i) + ": ********* epoch " + str(        #     i * 100 // mode_decision.train.images.shape[0] + 1) +        #       " ********* test accuracy for all:" + str(a) + " test loss: " + str(        #     c))    #     # test_data0 = {X: mode_decision.test0.images, Y_: mode_decision.test0.labels}    #     a, c = sess.run([accuracy, cross_entropy],    #                     feed_dict={X: mode_decision.test0.images, Y_: mode_decision.test0.labels, tst: True, pkeep: 1.0,    #                                pkeep_conv: 1.0})    #     print(str(i) + ": ********* epoch " + str(    #         i * 100 // mode_decision.train.images.shape[0] + 1) +    #           " ********* test accuracy for mode 0:" + str(a) + " test loss: " + str(    #         c))    #    #     a, c = sess.run([accuracy, cross_entropy],    #                     feed_dict={X: mode_decision.test1.images, Y_: mode_decision.test1.labels, tst: True, pkeep: 1.0,    #                                pkeep_conv: 1.0})    #     print(str(i) + ": ********* epoch " + str(    #         i * 100 // mode_decision.train.images.shape[0] + 1) +    #           " ********* test accuracy for mode 1:" + str(a) + " test loss: " + str(    #         c))    #    #     a, c = sess.run([accuracy, cross_entropy],    #                     feed_dict={X: mode_decision.test2.images, Y_: mode_decision.test2.labels, tst: True, pkeep: 1.0,    #                                pkeep_conv: 1.0})    #     print(str(i) + ": ********* epoch " + str(    #         i * 100 // mode_decision.train.images.shape[0] + 1) +    #           " ********* test accuracy for mode 2:" + str(a) + " test loss: " + str(    #         c))    #    #     a, c = sess.run([accuracy, cross_entropy],    #                     feed_dict={X: mode_decision.test24.images, Y_: mode_decision.test24.labels, tst: True,    #                                pkeep: 1.0, pkeep_conv: 1.0})    #     print(str(i) + ": ********* epoch " + str(    #         i * 100 // mode_decision.train.images.shape[0] + 1) +    #           " ********* test accuracy for mode 24:" + str(a) + " test loss: " + str(    #         c))    #    #     a, c = sess.run([accuracy, cross_entropy],    #                     feed_dict={X: mode_decision.test25.images, Y_: mode_decision.test25.labels, tst: True,    #                                pkeep: 1.0, pkeep_conv: 1.0})    #     print(str(i) + ": ********* epoch " + str(    #         i * 100 // mode_decision.train.images.shape[0] + 1) +    #           " ********* test accuracy for mode 25:" + str(a) + " test loss: " + str(    #         c))    #     # ----    #     a, c = sess.run([accuracy, cross_entropy],    #                     feed_dict={X: mode_decision.test26.images, Y_: mode_decision.test26.labels, tst: True,    #                                pkeep: 1.0, pkeep_conv: 1.0})    #     print(str(i) + ": ********* epoch " + str(    #         i * 100 // mode_decision.train.images.shape[0] + 1) +    #           " ********* test accuracy for mode 26:" + str(a) + " test loss: " + str(    #         c))    #    #     a, c = sess.run([accuracy, cross_entropy],    #                     feed_dict={X: mode_decision.test27.images, Y_: mode_decision.test27.labels, tst: True,    #                                pkeep: 1.0, pkeep_conv: 1.0})    #     print(str(i) + ": ********* epoch " + str(    #         i * 100 // mode_decision.train.images.shape[0] + 1) +    #           " ********* test accuracy for mode 27:" + str(a) + " test loss: " + str(    #         c))    #    #     a, c = sess.run([accuracy, cross_entropy],    #                     feed_dict={X: mode_decision.test28.images, Y_: mode_decision.test28.labels, tst: True,    #                                pkeep: 1.0, pkeep_conv: 1.0})    #     print(str(i) + ": ********* epoch " + str(    #         i * 100 // mode_decision.train.images.shape[0] + 1) +    #           " ********* test accuracy for mode 28:" + str(a) + " test loss: " + str(    #         c))    #    #     a, c = sess.run([accuracy, cross_entropy],    #                     feed_dict={X: mode_decision.test29.images, Y_: mode_decision.test29.labels, tst: True,    #                                pkeep: 1.0, pkeep_conv: 1.0})    #     print(str(i) + ": ********* epoch " + str(    #         i * 100 // mode_decision.train.images.shape[0] + 1) +    #           " ********* test accuracy for mode 29:" + str(a) + " test loss: " + str(    #         c))    #    #     a, c = sess.run([accuracy, cross_entropy],    #                     feed_dict={X: mode_decision.test30.images, Y_: mode_decision.test30.labels, tst: True,    #                                pkeep: 1.0, pkeep_conv: 1.0})    #     print(str(i) + ": ********* epoch " + str(    #         i * 100 // mode_decision.train.images.shape[0] + 1) +    #           " ********* test accuracy for mode 30:" + str(a) + " test loss: " + str(    #         c))    #    #     a, c = sess.run([accuracy, cross_entropy],    #                     feed_dict={X: mode_decision.test31.images, Y_: mode_decision.test31.labels, tst: True,    #                                pkeep: 1.0, pkeep_conv: 1.0})    #     print(str(i) + ": ********* epoch " + str(    #         i * 100 // mode_decision.train.images.shape[0] + 1) +    #           " ********* test accuracy for mode 31:" + str(a) + " test loss: " + str(    #         c))    #    #     a, c = sess.run([accuracy, cross_entropy],    #                     feed_dict={X: mode_decision.test32.images, Y_: mode_decision.test32.labels, tst: True,    #                                pkeep: 1.0, pkeep_conv: 1.0})    #     print(str(i) + ": ********* epoch " + str(    #         i * 100 // mode_decision.train.images.shape[0] + 1) +    #           " ********* test accuracy for mode 32:" + str(a) + " test loss: " + str(    #         c))    #    #     a, c = sess.run([accuracy, cross_entropy],    #                     feed_dict={X: mode_decision.test33.images, Y_: mode_decision.test33.labels, tst: True,    #                                pkeep: 1.0, pkeep_conv: 1.0})    #     print(str(i) + ": ********* epoch " + str(    #         i * 100 // mode_decision.train.images.shape[0] + 1) +    #           " ********* test accuracy for mode 33:" + str(a) + " test loss: " + str(    #         c))    #    #     a, c = sess.run([accuracy, cross_entropy],    #                     feed_dict={X: mode_decision.test34.images, Y_: mode_decision.test34.labels, tst: True,    #                                pkeep: 1.0, pkeep_conv: 1.0})    #     print(str(i) + ": ********* epoch " + str(    #         i * 100 // mode_decision.train.images.shape[0] + 1) +    #           " ********* test accuracy for mode 34:" + str(a) + " test loss: " + str(    #         c))    #    #     a, c = sess.run([accuracy, cross_entropy],    #                     feed_dict={X: mode_decision.test35.images, Y_: mode_decision.test35.labels, tst: True,    #                                pkeep: 1.0, pkeep_conv: 1.0})    #     print(str(i) + ": ********* epoch " + str(    #         i * 100 // mode_decision.train.images.shape[0] + 1) +    #           " ********* test accuracy for mode 35:" + str(a) + " test loss: " + str(    #         c))    #    #     a, c = sess.run([accuracy, cross_entropy],    #                     feed_dict={X: mode_decision.test36.images, Y_: mode_decision.test36.labels, tst: True,    #                                pkeep: 1.0, pkeep_conv: 1.0})    #     print(str(i) + ": ********* epoch " + str(    #         i * 100 // mode_decision.train.images.shape[0] + 1) +    #           " ********* test accuracy for mode 36:" + str(a) + " test loss: " + str(    #         c))    # the backpropagation training step    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False, pkeep: 0.75, pkeep_conv: 1.0})    sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i, pkeep: 1.0, pkeep_conv: 1.0})for i in range(100000000 + 1):    training_step(i, i % 20 == 0, i % 1000 == 0)