# encoding: UTF-8import tensorflow as tfimport mathfrom a_vcmd_read_data_multiple_test_set import read_data_setsfrom datetime import datetimetf.set_random_seed(0)mode_decision = read_data_sets(reshape=False, validation_size=0)image_width = 16BATCH_SIZE = 100AFTER_CONV_DIM = 4def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,                                                       iteration)  # adding the iteration prevents from averaging across non-existing iterations    bnepsilon = 1e-5    if convolutional:        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])    else:        mean, variance = tf.nn.moments(Ylogits, [0])    update_moving_everages = exp_moving_avg.apply([mean, variance])    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)    return Ybn, update_moving_everagesdef no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):    return Ylogits, tf.no_op()def compatible_convolutional_noise_shape(Y):    noiseshape = tf.shape(Y)    noiseshape = noiseshape * tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])    return noiseshapedef variable_summaries(var):    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""    with tf.name_scope('summaries'):        mean = tf.reduce_mean(var)        tf.summary.scalar('mean', mean)        with tf.name_scope('stddev'):            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))        tf.summary.scalar('stddev', stddev)        tf.summary.scalar('max', tf.reduce_max(var))        tf.summary.scalar('min', tf.reduce_min(var))        tf.summary.histogram('histogram', var)# 5 conv, 6 relu, 1 softmaxwith tf.Graph().as_default():    with tf.device("/gpu:0"):        with tf.name_scope('input'):            X = tf.placeholder(tf.float32, [None, image_width, image_width, 1], name='X-input')            # correct answers go here            Y_ = tf.placeholder(tf.float32, [None, 37], name='Y_-input')        with tf.name_scope('input_reshape'):            image_shaped_input = tf.reshape(X, [-1, image_width, image_width, 1])            tf.summary.image('input', image_shaped_input, 37)        # learning rate        lr = tf.placeholder(tf.float32)        # test flag for batch norm        tst = tf.placeholder(tf.bool)        iter = tf.placeholder(tf.int32)        # dropout probability        pkeep = tf.placeholder(tf.float32)        pkeep_conv = tf.placeholder(tf.float32)        CONV_K = 64  # first conv layer output depth        CONV_L = 64  # second conv layer output depth        CONV_M = 128  # third conv layer output depth        CONV_N = 128  #        CONV_O = 256        L = 1000  # 1st FC        M = 800  # 2nd FC        N = 600  # 3rd FC        O = 400  # 4th FC        P = 200  # 5th FC        Q = 100  # 6th FC        R = 37  # last layer which is softmax for 37 modes classification        # 5x5 patch, 1 input channel, conv_K output channels        with tf.name_scope('weights'):            CONV_W1 = tf.Variable(tf.truncated_normal([3, 3, 1, CONV_K], stddev=0.5))            CONV_W2 = tf.Variable(tf.truncated_normal([3, 3, CONV_K, CONV_L], stddev=0.5))            CONV_W3 = tf.Variable(tf.truncated_normal([3, 3, CONV_L, CONV_M], stddev=0.5))            CONV_W4 = tf.Variable(tf.truncated_normal([3, 3, CONV_M, CONV_N], stddev=0.5))            CONV_W5 = tf.Variable(tf.truncated_normal([3, 3, CONV_N, CONV_O], stddev=0.5))            W1 = tf.Variable(tf.truncated_normal([AFTER_CONV_DIM * AFTER_CONV_DIM * CONV_O, L], stddev=0.5))            W7 = tf.Variable(tf.truncated_normal([L, R], stddev=0.5))            variable_summaries(CONV_W1)            variable_summaries(CONV_W2)            variable_summaries(CONV_W3)            variable_summaries(CONV_W4)            variable_summaries(CONV_W5)            variable_summaries(W1)            variable_summaries(W7)        with tf.name_scope('biases'):            CONV_B1 = tf.Variable(tf.constant(0.5, tf.float32, [CONV_K]))            CONV_B2 = tf.Variable(tf.constant(0.5, tf.float32, [CONV_L]))            CONV_B3 = tf.Variable(tf.constant(0.5, tf.float32, [CONV_M]))            CONV_B4 = tf.Variable(tf.constant(0.5, tf.float32, [CONV_N]))            CONV_B5 = tf.Variable(tf.constant(0.5, tf.float32, [CONV_O]))            B1 = tf.Variable(tf.constant(0.5, tf.float32, [L]))            B7 = tf.Variable(tf.constant(0.5, tf.float32, [R]))            variable_summaries(CONV_B1)            variable_summaries(CONV_B2)            variable_summaries(CONV_B3)            variable_summaries(CONV_B4)            variable_summaries(CONV_B5)            variable_summaries(B1)            variable_summaries(B7)        # truncated normal        # W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))        # B2 = tf.Variable(tf.constant(0.1, tf.float32, [M]))        #        # W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))        # B3 = tf.Variable(tf.constant(0.1, tf.float32, [N]))        #        # W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))        # B4 = tf.Variable(tf.constant(0.1, tf.float32, [O]))        #        # W5 = tf.Variable(tf.truncated_normal([O, P], stddev=0.1))        # B5 = tf.Variable(tf.constant(0.1, tf.float32, [P]))        #        # W6 = tf.Variable(tf.truncated_normal([P, Q], stddev=0.1))        # B6 = tf.Variable(tf.constant(0.1, tf.float32, [Q]))        # The model        # batch norm scaling is not useful with relus        # batch norm offsets are used instead of biases        stride = 1  # output is 16x16        CONV_Y1l = tf.nn.conv2d(X, CONV_W1, strides=[1, stride, stride, 1], padding='SAME')        CONV_Y1bn, CONV_update_ema1 = batchnorm(CONV_Y1l, tst, iter, CONV_B1, convolutional=True)        CONV_Y1r = tf.nn.relu(CONV_Y1bn)        CONV_Y1 = tf.nn.dropout(CONV_Y1r, pkeep_conv, compatible_convolutional_noise_shape(CONV_Y1r))        # Original we used the line below for the version of without batch normalization        # CONV_Y1 = tf.nn.relu(tf.nn.conv2d(X, CONV_W1, strides=[1, stride, stride, 1], padding='SAME') + CONV_B1)        stride = 1  # output is 16x16        CONV_Y2l = tf.nn.conv2d(CONV_Y1, CONV_W2, strides=[1, stride, stride, 1], padding='SAME')        CONV_Y2bn, CONV_update_ema2 = batchnorm(CONV_Y2l, tst, iter, CONV_B2, convolutional=True)        CONV_Y2r = tf.nn.relu(CONV_Y2bn)        CONV_Y2 = tf.nn.dropout(CONV_Y2r, pkeep_conv, compatible_convolutional_noise_shape(CONV_Y2r))        # CONV_Y2 = tf.nn.relu(tf.nn.conv2d(CONV_Y1, CONV_W2, strides=[1, stride, stride, 1], padding='SAME') + CONV_B2)        stride = 2  # output is 8x8        CONV_Y3l = tf.nn.conv2d(CONV_Y2, CONV_W3, strides=[1, stride, stride, 1], padding='SAME')        CONV_Y3bn, CONV_update_ema3 = batchnorm(CONV_Y3l, tst, iter, CONV_B3, convolutional=True)        CONV_Y3r = tf.nn.relu(CONV_Y3bn)        CONV_Y3 = tf.nn.dropout(CONV_Y3r, pkeep_conv, compatible_convolutional_noise_shape(CONV_Y3r))        # CONV_Y3 = tf.nn.relu(tf.nn.conv2d(CONV_Y2, CONV_W3, strides=[1, stride, stride, 1], padding='SAME') + CONV_B3)        stride = 1  # output is 8x8        CONV_Y4l = tf.nn.conv2d(CONV_Y3, CONV_W4, strides=[1, stride, stride, 1], padding='SAME')        CONV_Y4bn, CONV_update_ema4 = batchnorm(CONV_Y4l, tst, iter, CONV_B4, convolutional=True)        CONV_Y4r = tf.nn.relu(CONV_Y4bn)        CONV_Y4 = tf.nn.dropout(CONV_Y4r, pkeep_conv, compatible_convolutional_noise_shape(CONV_Y4r))        stride = 2  # output is 4x4        CONV_Y5l = tf.nn.conv2d(CONV_Y4, CONV_W5, strides=[1, stride, stride, 1], padding='SAME')        CONV_Y5bn, CONV_update_ema5 = batchnorm(CONV_Y5l, tst, iter, CONV_B5, convolutional=True)        CONV_Y5r = tf.nn.relu(CONV_Y5bn)        CONV_Y5 = tf.nn.dropout(CONV_Y5r, pkeep_conv, compatible_convolutional_noise_shape(CONV_Y5r))        # reshape the output from the third convolution for the fully connected layer        YY = tf.reshape(CONV_Y5, shape=[-1, AFTER_CONV_DIM * AFTER_CONV_DIM * CONV_O])        # relu, relu        Y1l = tf.matmul(YY, W1)        Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1)        Y1r = tf.nn.relu(Y1bn)        Y1 = tf.nn.dropout(Y1r, pkeep)        # # Y1 = tf.nn.relu(tf.matmul(YY, W1) + B1)        # # Y1d = tf.nn.dropout(Y1, pkeep)        # Y2l = tf.matmul(Y1, W2)        # Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2)        # Y2r = tf.nn.relu(Y2bn)        # Y2 = tf.nn.dropout(Y2r, pkeep)        #        # # Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)        # # Y2d = tf.nn.dropout(Y2, pkeep)        #        # Y3l = tf.matmul(Y2, W3)        # Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3)        # Y3r = tf.nn.relu(Y3bn)        # Y3 = tf.nn.dropout(Y3r, pkeep)        #        # # Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)        # # Y3d = tf.nn.dropout(Y3, pkeep)        #        # Y4l = tf.matmul(Y3, W4)        # Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4)        # Y4r = tf.nn.relu(Y4bn)        # Y4 = tf.nn.dropout(Y4r, pkeep)        #        # # Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)        # # Y4d = tf.nn.dropout(Y4, pkeep)        #        # Y5l = tf.matmul(Y4, W5)        # Y5bn, update_ema5 = batchnorm(Y5l, tst, iter, B5)        # Y5r = tf.nn.relu(Y5bn)        # Y5 = tf.nn.dropout(Y5r, pkeep)        #        # # Y5 = tf.nn.relu(tf.matmul(Y4d, W5) + B5)        # # Y5d = tf.nn.dropout(Y5, pkeep)        # Y6l = tf.matmul(Y5, W6)        # Y6bn, update_ema6 = batchnorm(Y6l, tst, iter, B6)        # Y6r = tf.nn.relu(Y6bn)        # Y6 = tf.nn.dropout(Y6r, pkeep)        #        # # Y6 = tf.nn.relu(tf.matmul(Y5d, W6) + B6)        # # Y6d = tf.nn.dropout(Y6, pkeep)        # # Y = tf.nn.softmax(tf.matmul(Y6, W7) + B7)        Ylogits = tf.matmul(Y1, W7) + B7        Y = tf.nn.softmax(Ylogits)        update_ema = tf.group(CONV_update_ema1, CONV_update_ema2, CONV_update_ema3,                              update_ema1)        with tf.name_scope('cross_entropy'):            # The raw formulation of cross-entropy,            #            # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),            #                               reduction_indices=[1]))            #            # can be numerically unstable.            #            # So here we use tf.nn.softmax_cross_entropy_with_logits on the            # raw outputs of the nn_layer above, and then average across            # the batch.            diff = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)            with tf.name_scope('total'):                cross_entropy = tf.reduce_mean(diff) * 100        tf.summary.scalar('cross_entropy', cross_entropy)        # cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images        # TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability        # problems with log(0) which is NaN        # accuracy of the trained model, between 0 (worst) and 1 (best)        with tf.name_scope('accuracy'):            with tf.name_scope('correct_prediction'):                correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))            with tf.name_scope('accuracy'):                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))        tf.summary.scalar('accuracy', accuracy)        global_step = tf.Variable(0, name='global_step', trainable=False)        # training step, the learning rate is a placeholder        with tf.name_scope('train'):            train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy, global_step=global_step)        # Build the summary Tensor based on the TF collection of Summaries.        summary = tf.summary.merge_all()        # Add the variable initializer Op.        init = tf.global_variables_initializer()        # Create a saver for writing training checkpoints.        saver = tf.train.Saver()        # Create a session for running Ops on the Graph.        config = tf.ConfigProto(allow_soft_placement=True)        sess = tf.Session(config=config)        # Instantiate a SummaryWriter to output summaries and the Graph.        summary_writer = tf.summary.FileWriter('/Users/Pharrell_WANG/PycharmProjects/proj_vcmd/ckpt_6_3_1', sess.graph)        test_writer = tf.summary.FileWriter('/Users/Pharrell_WANG/PycharmProjects/proj_vcmd/ckpt_6_3_1/test')        # And then after everything is built:        # Run the Op to initialize the variables.        sess.run(init)        print("**************************---- " + str(            mode_decision.train.images.shape[0]) + " -----************************** ")# now after the init op, we can use the model now. (train the model)def training_step(i, update_train_data, update_test_data, save_ckpt, loop_start_stamp):    # load batch of images and correct answers    batch_X, batch_Y = mode_decision.train.next_batch(100)    lrmax = 0.01    lrmin = 0.00005    decay_speed = 1800.0    learning_rate = lrmin + (lrmax - lrmin) * math.exp(-i / decay_speed)    if update_train_data:        if i % 100 == 99:  # Record execution stats            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)            run_metadata = tf.RunMetadata()            Summary, a, c = sess.run([summary, accuracy, cross_entropy],                                        feed_dict={X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False, pkeep: 0.75, pkeep_conv: 0.75},                                        options=run_options,                                        run_metadata=run_metadata)            summary_writer.add_run_metadata(run_metadata, 'step%03d' % i)            summary_writer.add_summary(Summary, i)            print('Adding run metadata for', i)        else:  # Record a summary            Summary, a, c = sess.run([summary, accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False, pkeep: 1.0, pkeep_conv: 1.0},)            summary_writer.add_summary(Summary, i)        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)        run_metadata = tf.RunMetadata()        Summary, a, c = sess.run([summary, accuracy, cross_entropy],                                 feed_dict={X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False, pkeep: 0.75, pkeep_conv: 0.75},                                 options=run_options,                                 run_metadata=run_metadata)        summary_writer.add_run_metadata(run_metadata, 'step%03d' % i)        summary_writer.add_summary(Summary, i)        # print('Adding run metadata for', i)        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")    if save_ckpt:        checkpoint_file = '/Users/Pharrell_WANG/PycharmProjects/proj_vcmd/ckpt_6_3_1/model_6_3_1.ckpt'        saver.save(sess, checkpoint_file, global_step=i)    if update_test_data:        stamp = datetime.now()        time_passed_since_loop_start = stamp - loop_start_stamp        print("")        print("--------->> *** *** -------->>                    loop started at  : " + str(loop_start_stamp))        print('--------->> *** *** -------->>                    now the time is  : ' + str(stamp))        print(            "--------->> *** *** -------->>        time passed since beginning  : " + str(time_passed_since_loop_start))        print("")        # test_size = 74000        sum_a = 0        sum_c = 0        for step in range(740):            batch_test_x, batch_test_y = mode_decision.test.next_batch(100)            # a, c = sess.run([accuracy, cross_entropy],            #                 feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0, pkeep_conv: 1.0})            Summary, a, c = sess.run([summary, accuracy, cross_entropy],                                     feed_dict={X: batch_test_x, Y_: batch_test_y,                                                tst: True, pkeep: 1.0, pkeep_conv: 1.0})            test_writer.add_summary(Summary, i)            sum_a += a            sum_c += c        a = sum_a / 740        c = sum_c / 740        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for ->    ALL 37 modes   <-: " + str(a) + " test loss: " + str(            c))        sum_a = 0        sum_c = 0        for step in range(740):            batch_test_x, batch_test_y = mode_decision.test0.next_batch(100)            a, c = sess.run([accuracy, cross_entropy],                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,                                       pkeep_conv: 1.0})            sum_a += a            sum_c += c        a = sum_a / 740        c = sum_c / 740        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for ---->>   mode 0    <---: " + str(a) + " test loss: " + str(            c))        sum_a = 0        sum_c = 0        for step in range(740):            batch_test_x, batch_test_y = mode_decision.test1.next_batch(100)            a, c = sess.run([accuracy, cross_entropy],                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,                                       pkeep_conv: 1.0})            sum_a += a            sum_c += c        a = sum_a / 740        c = sum_c / 740        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for ---->>   mode 1    <---: " + str(a) + " test loss: " + str(c))        sum_a = 0        sum_c = 0        for step in range(740):            batch_test_x, batch_test_y = mode_decision.test2.next_batch(100)            a, c = sess.run([accuracy, cross_entropy],                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,                                       pkeep_conv: 1.0})            sum_a += a            sum_c += c        a = sum_a / 740        c = sum_c / 740        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for ---->>   mode 2    <---: " + str(a) + " test loss: " + str(c))        #######################        sum_a = 0        sum_c = 0        for step in range(740):            batch_test_x, batch_test_y = mode_decision.test24.next_batch(100)            a, c = sess.run([accuracy, cross_entropy],                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,                                       pkeep_conv: 1.0})            sum_a += a            sum_c += c        a = sum_a / 740        c = sum_c / 740        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for ---->>   mode 24   <---: " + str(a) + " test loss: " + str(c))        #######################        sum_a = 0        sum_c = 0        for step in range(740):            batch_test_x, batch_test_y = mode_decision.test25.next_batch(100)            a, c = sess.run([accuracy, cross_entropy],                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,                                       pkeep_conv: 1.0})            sum_a += a            sum_c += c        a = sum_a / 740        c = sum_c / 740        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for ---->>   mode 25   <---: " + str(a) + " test loss: " + str(c))        #######################        sum_a = 0        sum_c = 0        for step in range(740):            batch_test_x, batch_test_y = mode_decision.test26.next_batch(100)            a, c = sess.run([accuracy, cross_entropy],                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,                                       pkeep_conv: 1.0})            sum_a += a            sum_c += c        a = sum_a / 740        c = sum_c / 740        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for ---->>   mode 26   <---: " + str(a) + " test loss: " + str(c))        #######################        sum_a = 0        sum_c = 0        for step in range(740):            batch_test_x, batch_test_y = mode_decision.test27.next_batch(100)            a, c = sess.run([accuracy, cross_entropy],                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,                                       pkeep_conv: 1.0})            sum_a += a            sum_c += c        a = sum_a / 740        c = sum_c / 740        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for ---->>   mode 27   <---: " + str(a) + " test loss: " + str(c))        #######################        sum_a = 0        sum_c = 0        for step in range(740):            batch_test_x, batch_test_y = mode_decision.test28.next_batch(100)            a, c = sess.run([accuracy, cross_entropy],                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,                                       pkeep_conv: 1.0})            sum_a += a            sum_c += c        a = sum_a / 740        c = sum_c / 740        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for ---->>   mode 28   <---: " + str(a) + " test loss: " + str(c))        #######################        sum_a = 0        sum_c = 0        for step in range(740):            batch_test_x, batch_test_y = mode_decision.test29.next_batch(100)            a, c = sess.run([accuracy, cross_entropy],                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,                                       pkeep_conv: 1.0})            sum_a += a            sum_c += c        a = sum_a / 740        c = sum_c / 740        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for ---->>   mode 29   <---: " + str(a) + " test loss: " + str(c))        #######################        sum_a = 0        sum_c = 0        for step in range(740):            batch_test_x, batch_test_y = mode_decision.test30.next_batch(100)            a, c = sess.run([accuracy, cross_entropy],                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,                                       pkeep_conv: 1.0})            sum_a += a            sum_c += c        a = sum_a / 740        c = sum_c / 740        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for ---->>   mode 30   <---: " + str(a) + " test loss: " + str(c))        #######################        sum_a = 0        sum_c = 0        for step in range(740):            batch_test_x, batch_test_y = mode_decision.test31.next_batch(100)            a, c = sess.run([accuracy, cross_entropy],                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,                                       pkeep_conv: 1.0})            sum_a += a            sum_c += c        a = sum_a / 740        c = sum_c / 740        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for ---->>   mode 31   <---: " + str(a) + " test loss: " + str(c))        #######################        sum_a = 0        sum_c = 0        for step in range(740):            batch_test_x, batch_test_y = mode_decision.test32.next_batch(100)            a, c = sess.run([accuracy, cross_entropy],                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,                                       pkeep_conv: 1.0})            sum_a += a            sum_c += c        a = sum_a / 740        c = sum_c / 740        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for ---->>   mode 32   <---: " + str(a) + " test loss: " + str(c))        #######################        sum_a = 0        sum_c = 0        for step in range(740):            batch_test_x, batch_test_y = mode_decision.test33.next_batch(100)            a, c = sess.run([accuracy, cross_entropy],                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,                                       pkeep_conv: 1.0})            sum_a += a            sum_c += c        a = sum_a / 740        c = sum_c / 740        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for ---->>   mode 33   <---: " + str(a) + " test loss: " + str(c))        #######################        sum_a = 0        sum_c = 0        for step in range(740):            batch_test_x, batch_test_y = mode_decision.test34.next_batch(100)            a, c = sess.run([accuracy, cross_entropy],                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,                                       pkeep_conv: 1.0})            sum_a += a            sum_c += c        a = sum_a / 740        c = sum_c / 740        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for ---->>   mode 34   <---: " + str(a) + " test loss: " + str(c))        #######################        sum_a = 0        sum_c = 0        for step in range(740):            batch_test_x, batch_test_y = mode_decision.test35.next_batch(100)            a, c = sess.run([accuracy, cross_entropy],                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,                                       pkeep_conv: 1.0})            sum_a += a            sum_c += c        a = sum_a / 740        c = sum_c / 740        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for ---->>   mode 35   <---: " + str(a) + " test loss: " + str(c))        #######################        sum_a = 0        sum_c = 0        for step in range(740):            batch_test_x, batch_test_y = mode_decision.test36.next_batch(100)            a, c = sess.run([accuracy, cross_entropy],                            feed_dict={X: batch_test_x, Y_: batch_test_y, tst: True, pkeep: 1.0,                                       pkeep_conv: 1.0})            sum_a += a            sum_c += c        a = sum_a / 740        c = sum_c / 740        print(str(i) + ": ********* epoch " + str(i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for ---->>   mode 36   <---: " + str(a) + " test loss: " + str(c))    # the backpropagation training step    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False, pkeep: 0.75, pkeep_conv: 1.0})    sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i, pkeep: 1.0, pkeep_conv: 1.0})loop_start = datetime.now()# if tf.gfile.Exists('/Users/Pharrell_WANG/PycharmProjects/proj_vcmd/ckpt_6_3_1'):#     tf.gfile.DeleteRecursively('/Users/Pharrell_WANG/PycharmProjects/proj_vcmd/ckpt_6_3_1')# tf.gfile.MakeDirs('/Users/Pharrell_WANG/PycharmProjects/proj_vcmd/ckpt_6_3_1')for i in range(1000000):    # Update train, update test, update save ckpt    training_step(i, i % 10 == 0, i % 5000 == 0, i % 1000 == 0, loop_start)