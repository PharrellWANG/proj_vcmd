# encoding: UTF-8import tensorflow as tfimport math# import numpy as np# import from_mnist.tensorflowvisufrom a_vcmd_read_data_multiple_test_set import read_data_setstf.set_random_seed(0)mode_decision = read_data_sets(reshape=False, validation_size=0)image_width = 16X = tf.placeholder(tf.float32, [None, image_width, image_width, 1])lr = tf.placeholder(tf.float32)pkeep = tf.placeholder(tf.float32)# =============# =============# =============# =============# =============# =============# =============K = 40  # first convolutional layer output depthL = 80  # second convolutional layer output depthM = 100N = 120  # fully connected layerP = 200W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channelsB1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))W4 = tf.Variable(tf.truncated_normal([2, 2, M, N], stddev=0.1))B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))# -------------# -------------# -------------# -------------# -------------W5 = tf.Variable(tf.truncated_normal([2 * 2 * N, P], stddev=0.1))B5 = tf.Variable(tf.constant(0.1, tf.float32, [P]))W6 = tf.Variable(tf.truncated_normal([P, 37], stddev=0.1))B6 = tf.Variable(tf.constant(0.1, tf.float32, [37]))# The modelstride = 1  # output isY1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)stride = 2  # output isY2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)stride = 2  # output isY3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)stride = 2Y4 = tf.nn.relu(tf.nn.conv2d(Y3, W4, strides=[1, stride, stride, 1], padding='SAME') + B4)# reshape the output from the third convolution for the fully connected layerYY = tf.reshape(Y4, shape=[-1, 2 * 2 * N])Y5 = tf.nn.relu(tf.matmul(YY, W5) + B5)YY5 = tf.nn.dropout(Y5, pkeep)Ylogits = tf.matmul(YY5, W6) + B6Y = tf.nn.softmax(Ylogits)# =============# =============# =============# =============# =============# =============# =============# Variables are all the parameters that you want the training algorithm to determine for you.# in our case, our weights and biases.Y_ = tf.placeholder(tf.float32, [None, 37])# Loss func [we use the model prediction and the correct labels for computing# the loss function, in our case: the cross entropy function]cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)cross_entropy = tf.reduce_mean(cross_entropy) * 100# cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))# % of correct answers found in batch (the percentage of correctly recognized digits)is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))optimizer = tf.train.AdamOptimizer(lr)train_step = optimizer.minimize(cross_entropy)# in Tensorflow, the computation requires actual data to be# fed into the placeholders you have defined in your Tensorflow code# This is supplied in the form of a Python dictionary where the keys are names# of the placeholders.# add an op to initialize the variables# (As the op added by tf.global_variables_initializer() initializes all variables in parallel# you have to be careful when this is needed.)init = tf.global_variables_initializer()saver = tf.train.Saver()# define the sessionsess = tf.Session()# run init operationsess.run(init)print("======------------>>>>>>>    "+str(mode_decision.train.images.shape[0]))# now after the init op, we can use the model now. (train the model)for i in range(10000001):    # for i in range(5000000):    # load batch of images and correct answers    batch_X, batch_Y = mode_decision.train.next_batch(100)    # learning rate decay    max_learning_rate = 0.005    min_learning_rate = 0.0001    decay_speed = 1000.0  # 0.003-0.0001-2000=>0.9826 done in 5000 iterations    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)    # train_data = {X: batch_X, Y_: batch_Y, pkeep: 0.75}    # sess.run(train_step, feed_dict=train_data)    sess.run(train_step, {X: batch_X, Y_: batch_Y, pkeep: 0.75, lr: learning_rate})    # the backpropagation training step    # sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate})    if i % 10 == 0:        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1})        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")    if i % 500 == 0:        save_path = saver.save(sess, "/Users/Pharrell_WANG/PycharmProjects/proj_vcmd/checkpoint/1_3_model.ckpt")        print("Model saved in file: %s" % save_path)        # test_data = {X: mode_decision.test.images, Y_: mode_decision.test.labels}        a, c = sess.run([accuracy, cross_entropy],                        feed_dict={X: mode_decision.test.images, Y_: mode_decision.test.labels, pkeep: 1})        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for all:" + str(a) + " test loss: " + str(            c))        # test_data0 = {X: mode_decision.test0.images, Y_: mode_decision.test0.labels}        a, c = sess.run([accuracy, cross_entropy],                        feed_dict={X: mode_decision.test0.images, Y_: mode_decision.test0.labels, pkeep: 1})        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for mode 0:" + str(a) + " test loss: " + str(            c))        a, c = sess.run([accuracy, cross_entropy],                        feed_dict={X: mode_decision.test1.images, Y_: mode_decision.test1.labels, pkeep: 1})        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for mode 1:" + str(a) + " test loss: " + str(            c))        a, c = sess.run([accuracy, cross_entropy],                        feed_dict={X: mode_decision.test2.images, Y_: mode_decision.test2.labels, pkeep: 1})        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for mode 2:" + str(a) + " test loss: " + str(            c))        a, c = sess.run([accuracy, cross_entropy],                        feed_dict={X: mode_decision.test24.images, Y_: mode_decision.test24.labels, pkeep: 1})        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for mode 24:" + str(a) + " test loss: " + str(            c))        a, c = sess.run([accuracy, cross_entropy],                        feed_dict={X: mode_decision.test25.images, Y_: mode_decision.test25.labels, pkeep: 1})        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for mode 25:" + str(a) + " test loss: " + str(            c))# ----        a, c = sess.run([accuracy, cross_entropy],                        feed_dict={X: mode_decision.test26.images, Y_: mode_decision.test26.labels, pkeep: 1})        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for mode 26:" + str(a) + " test loss: " + str(            c))        a, c = sess.run([accuracy, cross_entropy],                        feed_dict={X: mode_decision.test27.images, Y_: mode_decision.test27.labels, pkeep: 1})        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for mode 27:" + str(a) + " test loss: " + str(            c))        a, c = sess.run([accuracy, cross_entropy],                        feed_dict={X: mode_decision.test28.images, Y_: mode_decision.test28.labels, pkeep: 1})        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for mode 28:" + str(a) + " test loss: " + str(            c))        a, c = sess.run([accuracy, cross_entropy],                        feed_dict={X: mode_decision.test29.images, Y_: mode_decision.test29.labels, pkeep: 1})        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for mode 29:" + str(a) + " test loss: " + str(            c))        a, c = sess.run([accuracy, cross_entropy],                        feed_dict={X: mode_decision.test30.images, Y_: mode_decision.test30.labels, pkeep: 1})        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for mode 30:" + str(a) + " test loss: " + str(            c))        a, c = sess.run([accuracy, cross_entropy],                        feed_dict={X: mode_decision.test31.images, Y_: mode_decision.test31.labels, pkeep: 1})        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for mode 31:" + str(a) + " test loss: " + str(            c))        a, c = sess.run([accuracy, cross_entropy],                        feed_dict={X: mode_decision.test32.images, Y_: mode_decision.test32.labels, pkeep: 1})        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for mode 32:" + str(a) + " test loss: " + str(            c))        a, c = sess.run([accuracy, cross_entropy],                        feed_dict={X: mode_decision.test33.images, Y_: mode_decision.test33.labels, pkeep: 1})        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for mode 33:" + str(a) + " test loss: " + str(            c))        a, c = sess.run([accuracy, cross_entropy],                        feed_dict={X: mode_decision.test34.images, Y_: mode_decision.test34.labels, pkeep: 1})        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for mode 34:" + str(a) + " test loss: " + str(            c))        a, c = sess.run([accuracy, cross_entropy],                        feed_dict={X: mode_decision.test35.images, Y_: mode_decision.test35.labels, pkeep: 1})        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for mode 35:" + str(a) + " test loss: " + str(            c))        a, c = sess.run([accuracy, cross_entropy],                        feed_dict={X: mode_decision.test36.images, Y_: mode_decision.test36.labels, pkeep: 1})        print(str(i) + ": ********* epoch " + str(            i * 100 // mode_decision.train.images.shape[0] + 1) +              " ********* test accuracy for mode 36:" + str(a) + " test loss: " + str(            c))