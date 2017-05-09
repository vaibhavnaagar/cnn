import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.python.client import timeline
from tsne_img_plot import *
import numpy as np

tf.set_random_seed(0.0)

def plots(x, y, z, steps):
        try:
            plt.figure(1)
            plt.plot(x, '-bo', label="Loss")
            plt.xlabel('Number of iterations', fontsize=18)
            plt.ylabel('Loss', fontsize=18)
            plt.title('Training Error rate vs Number of iterations')
            plt.savefig("Loss_function_vs_iter_b_128.jpeg")
        except:
            pass

        try:
            plt.figure(2)
            plt.plot(steps, y, '-bo', label="Training Loss")
            plt.plot(steps, z, '-ro', label="Validation Loss")
            plt.xlabel('Number of iterations', fontsize=18)
            plt.ylabel('Loss Value', fontsize=18)
            plt.title('Training and Validation error rates vs number of iterations')
            plt.legend(loc='upper right')
            plt.savefig("error_rates_b_128.jpeg")
        except:
            pass
        pass


mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=10000)

print(mnist.train.images.shape)

def get_accuracy(pred_output, true_output):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred_output, 1), tf.argmax(true_output, 1)), tf.float32)).eval() * 100

lenet5_graph = tf.Graph()
batch_size = 128
t_label = mnist.train.labels[:]

with lenet5_graph.as_default():
    ### Training Dataset ###
    X_train_img = tf.placeholder(tf.float32, [batch_size, 28, 28, 1])
    Y_train_lbl = tf.placeholder(tf.float32, [batch_size, 10])

    ### Test Dataset ###
    X_train_img_full = tf.constant(mnist.train.images)
    X_test_img = tf.constant(mnist.test.images)

    ## Validation dataset
    X_valid = tf.constant(mnist.validation.images)
    Y_valid = tf.constant(mnist.validation.labels)

    ###  Hyper-parameters ###
    # learning rate
    #alpha = tf.placeholder(tf.float32)
    #alpha = tf.Variable(tf.constant(0.001, tf.float32))
    # regularization parameter
    #beta = 0.001

    ### LENET-5 Model ###
    ## Channels in layers ##
    C_conv1 = 6
    C_conv2 = 16
    N_fc1   = 120
    N_fc2   = 84
    N_fc3   = 10

    ## Weights and biases of layers ##
    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, C_conv1], stddev=0.1))
    B_conv1 = tf.Variable(tf.constant(0.1, tf.float32, [C_conv1]))
    W_conv2 = tf.Variable(tf.truncated_normal([5, 5, C_conv1, C_conv2], stddev=0.1))
    B_conv2 = tf.Variable(tf.constant(0.1, tf.float32, [C_conv2]))

    W_fc1 = tf.Variable(tf.truncated_normal([400, N_fc1], stddev=0.1))
    B_fc1 = tf.Variable(tf.constant(0.1, tf.float32, [N_fc1]))
    W_fc2 = tf.Variable(tf.truncated_normal([N_fc1 , N_fc2], stddev=0.1))
    B_fc2 = tf.Variable(tf.constant(0.1, tf.float32, [N_fc2]))
    W_fc3 = tf.Variable(tf.truncated_normal([N_fc2 , N_fc3], stddev=0.1))
    B_fc3 = tf.Variable(tf.constant(0.1, tf.float32, [N_fc3]))

    def lenet5_model(input_imgs):
        ## Layers ##
        # conv1
        conv1 = tf.nn.relu(tf.nn.conv2d(input_imgs, W_conv1, strides=[1, 1, 1, 1], padding="SAME", name="conv1") + B_conv1, name="relu1")

        # max-pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")

        # conv2
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W_conv2, strides=[1, 1, 1, 1], padding="VALID", name="conv2") + B_conv2, name="relu2")

        # max-pool2
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")

        # fully-connected1
        fmap_shp = pool2.get_shape().as_list()
        fmap_reshp = tf.reshape(pool2, [fmap_shp[0], fmap_shp[1]*fmap_shp[2]*fmap_shp[3]], name="reshape")
        fc1 = tf.nn.sigmoid(tf.matmul(fmap_reshp, W_fc1) + B_fc1, name="fc1")

        # fully-connected2
        fc2 = tf.nn.sigmoid(tf.matmul(fc1, W_fc2) + B_fc2, name="fc2")

        # fully-connected3 with softmax
        output = tf.matmul(fc2, W_fc3) + B_fc3

        return output, fc2

    ### Loss ###
    logits, dummy = lenet5_model(X_train_img)
    #regularizers = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_fc3)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_train_lbl)) # + beta*regularizers)

    ### Gradient Optimizer (Adagrad) ###
    grad_optimizer = tf.train.AdamOptimizer().minimize(loss)

    #wsum = tf.reduce_sum(tf.square(W_conv1)) + tf.reduce_sum(tf.square(W_conv2)) + tf.reduce_sum(tf.square(W_fc1)) + tf.reduce_sum(tf.square(W_fc2)) + tf.reduce_sum(tf.square(W_fc3))

    valid_logits, dummy = lenet5_model(X_valid)
    valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits, labels=Y_valid))

    ### Predictions ###
    predict_train = tf.nn.softmax(logits)
    predict_train_full = tf.nn.softmax(lenet5_model(X_train_img_full)[0])
    final_output, final_actiavtions = lenet5_model(X_test_img)
    predict_test  = tf.nn.softmax(final_output)

epochs = 4
iterations = int(np.ceil(50000/ batch_size))
print("Train dataset: ", mnist.train.images.shape, mnist.train.labels.shape)
print(" Validation dataset", mnist.validation.images.shape, mnist.validation.labels.shape)
print(" Test dataset", mnist.test.images.shape, mnist.test.labels.shape)

with tf.Session(graph=lenet5_graph) as sess:
    """config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    sess = tf.Session(config=config)"""
    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #run_metadata = tf.RunMetadata()
    tf.global_variables_initializer().run()
    print("Initialization Done !!")
    cost_history = []
    steps = []
    valid_history = []
    cost_step = []
    step = 0
    for ep in range(epochs):
        for it in range(iterations):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            #print(X_batch.shape, Y_batch.shape)
            feed = {X_train_img : X_batch, Y_train_lbl : Y_batch}
            _, cost, train_predictions = sess.run([grad_optimizer, loss, predict_train], feed_dict=feed)
            cost_history += [cost]
            print("Iteration: ", it, " Cost: ", cost, " Minibatch accuracy: ", get_accuracy(train_predictions, Y_batch))
            if step%100 == 0:
                valid_cost = valid_loss.eval(session=sess)
                print("Validation Cost:", valid_cost)
                valid_history += [valid_cost]
                cost_step += [cost]
                steps += [step]
            step += 1

        print("=======================================")
        test_output = predict_test.eval(session=sess)
        print("Epoch: ", ep, " Test Accuracy: ", get_accuracy(test_output, mnist.test.labels))
        #print("Train Accuracy: ", get_accuracy(predict_train_full.eval(session=sess), t_label))
        print("=======================================")
    test_output = predict_test.eval(session=sess)
    #test_output = sess.run(predict_test)
    test_y = np.argmax(mnist.test.labels, axis=1)
    #print("Final Test Accuracy: ", get_accuracy(test_output, mnist.test.labels))
    print("plotting Error rates...")
    plots(cost_history, cost_step, valid_history, steps)
    #print(test_output[1].shape, test_y.shape)
    a = final_actiavtions.eval(session=sess)
    print("Plotting tsne...")
    plot_tsne(a, test_y, "tf_tsne_b_128.jpeg")


    ### On one 1 image ###
    #p = sess.run([predict_test], options=run_options, run_metadata=run_metadata)
    # Create the Timeline object, and write it to a json
    #tl = timeline.Timeline(run_metadata.step_stats)
    #ctf = tl.generate_chrome_trace_format()
    #with open('timeline.json', 'w') as f:
    #    f.write(ctf)
    pass
