import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

#helper.maybe_download_pretrained_vgg('./data')

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    # print("input_image shape ", input_image)
    # print("keep_prob ", keep_prob)
    # print("vgg_layer3_out ", vgg_layer3_out)
    # print("vgg_layer4_out ", vgg_layer4_out)
    # print("vgg_layer7_out ", vgg_layer7_out)
    return input_image, keep_prob, vgg_layer3_out, vgg_layer4_out,\
        vgg_layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # custom init with the seed set to 0 by default
    def custom_init(shape, dtype=tf.float32, partition_info=None, seed=0):
        return tf.random_normal(shape, dtype=dtype, seed=seed)

    print("vgg_layer_7_out ", vgg_layer7_out.get_shape())
    print("vgg_layer7_out ", vgg_layer7_out)

    # 1x1 convolution
    dense_out = tf.layers.dense(vgg_layer7_out, num_classes,
                                kernel_initializer=custom_init,
                                name='dense_out')
    conv_1x1 = tf.layers.conv2d(dense_out, num_classes, 1, strides=(1, 1),
                                kernel_initializer=custom_init,
                                name='conv_1x1', padding='SAME')
    conv_1x1 = tf.reshape(conv_1x1, [-1, -1, -1, 512])

    print("conv_1x1 ", conv_1x1)

    # encoder
    skip1 = tf.add(conv_1x1, vgg_layer4_out, name='skip1')
    print(skip1)
    trans1 = tf.layers.conv2d_transpose(skip1, num_classes, 4, strides=(2, 2),
                                        name='trans1')
    trans1 = tf.reshape(trans1, [-1, -1, -1, 256])
    print(trans1)

    skip2 = tf.add(trans1, vgg_layer3_out, name='skip2')
    print(skip2)
    trans2 = tf.layers.conv2d_transpose(skip2, num_classes, 16, strides=(8, 8),
                                        name='trans2')
    print(trans2)

    output = tf.layers.conv2d_transpose(trans1, num_classes, 4, strides=(2, 2),
                                        name='output')
    print(output)
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # print("nn_last_layer ", nn_last_layer)
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    # print("logits ", logits)
    # print("correct_label ", correct_label)

    with tf.name_scope('cross_entropy_loss'):
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                    labels=correct_label))
        with tf.name_scope('total'):
            cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)


    return logits, optimizer, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    log_dir = '/tmp/tf/adl/logs'
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    learn_rate = .001
    dropout = 0.25

    with sess.as_default():

        # Merge all the summaries and write them out to /tmp/tf/adl/logs
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)

        # test_writer = tf.summary.FileWriter(log_dir + '/test')

        # saver = tf.train.Saver([])

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # train model
        step = 0
        for i in range(epochs):
            for batch_image, batch_label in get_batches_fn(batch_size):
                step += 1

                if step % 1000 == 0:  # record a summary with trace
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged,  train_op],
                                          feed_dict={input_image: batch_image,
                                                     correct_label: batch_label,
                                                     learning_rate: learn_rate,
                                                     keep_prob: dropout},
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata,
                                                  'epoch%03d-step%04d' % (i,step))
                    train_writer.add_summary(summary, step)
                    print('Adding run metadata for ', step)

                elif step % 100 == 0:  # record a summary
                    summary, _ = sess.run([merged,  train_op],
                                          feed_dict={input_image: batch_image,
                                                     correct_label: batch_label,
                                                     learning_rate: learn_rate,
                                                     keep_prob: dropout})
                    train_writer.add_summary(summary, step)
                else:
                    _ = sess.run(train_op,
                                 feed_dict={input_image: batch_image,
                                            correct_label: batch_label,
                                            learning_rate: learn_rate,
                                            keep_prob: dropout})



            # save a model checkpoint for tensorboard
            # saver.save(sess, log_dir+'/model.ckpt', step)

        # saver.save(sess. os.getcwd() + "/fcn-run")
        train_writer.close()

        # test_writer.close()

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        training_dir = os.path.join(data_dir, 'data_road/training')
        get_batches_fn = helper.gen_batch_function(training_dir, image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out\
            = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out,
                               vgg_layer7_out, num_classes)
        learning_rate = tf.placeholder(tf.float32, name='learning-rate')
        correct_label = tf.placeholder(tf.float32,
                                       (None, None, None, num_classes),
                                       name='correct-label')
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer,
                                                        correct_label,
                                                        learning_rate,
                                                        num_classes)

        # TODO: Train NN using the train_nn function
        epochs = 5
        batch_size = 5
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
                 cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)
        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
