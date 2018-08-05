
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import tag_constants
import matplotlib.pyplot as plt
import time

import traceback
import logging


from trainer.params_extractor import extract_parameters
from trainer.tf_methods import *
from trainer.graph_final_layer import append_final_layer_to_graph
import pickle
import winsound
import datetime

rng = np.random

# CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'
CHECKPOINT_NAME = 'trainer/trained_model3/'
FLAGS = None

# Parameters
learning_rate = 0.001
display_step = 20

batch_size = 16
bottleneck_tensor_size = 2048

cost_y = []

is_training = 1


def init_data():
    global data_X, data_Y
    data_X = np.arange(200, step=0.2)
    data_Y = data_X + 20 * np.sin(data_X / 10)
    return shuffle_data(data_X, data_Y)


def shuffle_data(X, Y):
    data_X_Y = list(zip(X, Y))
    rng.shuffle(data_X_Y)
    return zip(*data_X_Y)


def split_data_to_train_and_test():
    global train_X, train_Y, test_X, test_Y
    data_size = len(data_X)
    train_X = np.asarray(data_X[:int(data_size * 0.9)])
    train_Y = np.asarray(data_Y[:int(data_size * 0.9)])
    test_X = np.asarray(data_X[int(data_size * 0.9):])
    test_Y = np.asarray(data_Y[int(data_size * 0.9):])
    return (train_X, train_Y, test_X, test_Y)


if __name__ == '__main__':
    name = '12'
    # image_dir_folder = 'three_classes'
    image_dir_folder = 'M_labeled_train_validate'
    #image_dir_folder = 'X'

    epochs = 2
    create_bottlenecks = 1

    params = (name, image_dir_folder, epochs, create_bottlenecks)
    FLAGS, unparsed = extract_parameters(argparse.ArgumentParser(), params)
    initFlags(FLAGS)
    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    tf.logging.set_verbosity(tf.logging.INFO)
    prepare_file_system()

    winsound.Beep(440, 500)
    winsound.Beep(540, 100)
    winsound.Beep(440, 100)
    winsound.Beep(340, 1000)

    # Gather information about the model architecture we'll be using.
    model_info = create_model_info(FLAGS.architecture)
    if not model_info:
        tf.logging.error('Did not recognize architecture flag')
        sys.exit("finishing.")

    tf.logging.info("elo")
    winsound.Beep(440, 100)
    winsound.Beep(540, 100)
    winsound.Beep(640, 200)

    # Look at the folder structure, and create lists of all the images.
    image_lists = get_or_create_image_lists()

    winsound.Beep(640, 100)
    winsound.Beep(540, 100)
    winsound.Beep(440, 200)

    tf.logging.info("elo. image lists created.")


try:
    # Set up the pre-trained graph.
    maybe_download_and_extract(model_info['data_url'])
    graph, bottleneck_tensor, resized_image_tensor = (
        create_model_graph(model_info))

    # Add final layers to graph description
    (bottleneck_input, ground_truth_input,
     final_tensor, MAE, train_step) = append_final_layer_to_graph(graph, bottleneck_tensor, bottleneck_tensor_size,
                                                                  learning_rate)
    tf.logging.info("Added final layer.")

    ##########################

    img_dir = "imgs/3121.jpg"  # age 125 zwraca  2.33032274  61.44803619
    # img_dir = "imgs/10434.jpg" # age 126 zwraca  2.33032894 61.44826126
    # img_dir = "imgs/2024.jpg" # age 126 zwraca              61.44814682
    # img_dir = "imgs/_801_4808766.jpeg" # age 126 zwraca     61.44825745
    img_dir = "imgs/_5186_8315726.jpeg"  # age 204 zwraca    61.44829178
    img_dir = "imgs/_5185_9483.jpeg"  # age 204 zwraca

    img_dir = 'trainer/tester/' + img_dir
    gt = np.array(0.0).reshape(1, )

    image_data = gfile.FastGFile(img_dir, 'rb').read()

    ##########################



    # Start training
    with tf.Session(graph=graph) as sess:
        # Set up the image decoding sub-graph.
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
            model_info['input_width'], model_info['input_height'],
            model_info['input_depth'], model_info['input_mean'],
            model_info['input_std'])

        # We'll make sure we've calculated the 'bottleneck' image summaries and
        # cached them on disk.
        if FLAGS.create_bottlenecks:
            cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                              FLAGS.bottleneck_dir, jpeg_data_tensor,
                              decoded_image_tensor, resized_image_tensor,
                              bottleneck_tensor, FLAGS.architecture)

        # Create the operations we need to evaluate the accuracy of our new layer.
        # evaluation_step, _ = add_evaluation_step(final_tensor, ground_truth_input)
        # evaluation_MAE, _ = add_evaluation_step_MAE(final_tensor, ground_truth_input)

        tf.logging.info("elo. evaluations added")

        # Merge all the summaries and write them out to the summaries_dir

        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                             sess.graph)

        validation_writer = tf.summary.FileWriter(
            FLAGS.summaries_dir + '/validation')

        # Create a train saver that is used to restore values into an eval graph
        # when exporting models.
        # train_saver = tf.train.Saver()
        saver = tf.train.Saver()

        # Set up all our weights to their initial default values.
        init = tf.global_variables_initializer()
        sess.run(init)

        tf.logging.info("elo. session prepared")

        ################################


        training_data = 0
        for k in list(image_lists.keys()):
            if image_lists[k]['training']:
                training_data += len(image_lists[k]['training'])
        print("imagelists", image_lists)
        tf.logging.info("all training data: %d" % (training_data))

        steps_in_epoch = int(training_data / FLAGS.train_batch_size) - 1

        i = 0
        for j in range(FLAGS.how_many_epochs):

            winsound.Beep(440, 100)
            winsound.Beep(540, 200)
            winsound.Beep(640, 300)
            print("Start of epoch " + str(j + 1))
            print(time.time())
            print(datetime.datetime.now())

            epoch_image_lists = []
            whole_image_lists = []
            # epoch_image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
            #                              FLAGS.validation_percentage)
            with open(FLAGS.bottleneck_dir + '/image_list_division.pkl', 'rb') as input:
                epoch_image_lists = pickle.load(input)
                tf.logging.info("loaded")
            with open(FLAGS.bottleneck_dir + '/image_list_division.pkl', 'rb') as input:
                whole_image_lists = pickle.load(input)
                tf.logging.info("loaded")

            bottleneck_rnd_test = BottlenecksRandomizer('validation', whole_image_lists)
            (all_test_bottlenecks,
             all_test_ground_truth, _, _) = get_random_cached_bottlenecks(
                sess, bottleneck_rnd_test, -1,
                FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                FLAGS.architecture)

            bottleneck_rnd_train = BottlenecksRandomizer('training', epoch_image_lists)

            tf.logging.info("image list labels:")
            tf.logging.info(len(list(epoch_image_lists.keys())))

            tf.logging.info("starting epoch %d" % j)

            for s in range(steps_in_epoch):
                if len(epoch_image_lists.keys()) is 0:
                    tf.logging.info("too many steps")
                    continue
                tf.logging.info("epoch %d, batch %d, overal iteration %d" % (j, s, i))
                #
                # train_bottlenecks = all_train_bottlenecks[
                #                   j * FLAGS.train_batch_size: j * FLAGS.train_batch_size + FLAGS.train_batch_size]
                # train_ground_truth = all_train_ground_truth[
                #                   j * FLAGS.train_batch_size: j * FLAGS.train_batch_size + FLAGS.train_batch_size]
                merged = tf.summary.merge_all()

                (train_bottlenecks,
                 train_ground_truth, _, epoch_image_lists) = get_random_cached_bottlenecks(
                    sess, bottleneck_rnd_train, FLAGS.train_batch_size,
                    FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                    decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                    FLAGS.architecture)
                if epoch_image_lists is -1:
                    break

                # Feed the bottlenecks and ground truth into the graph, and run a training
                # step. Capture training summaries for TensorBoard with the `merged` op.
                # train_summary, _ = sess.run(
                #     [merged, train_step],
                #     feed_dict={bottleneck_input: train_bottlenecks,
                #                ground_truth_input: train_ground_truth})
                # train_writer.add_summary(train_summary, i)

                x = train_bottlenecks
                y = train_ground_truth

                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                results, _, _ = sess.run([final_tensor, train_step, extra_update_ops],
                                         feed_dict={bottleneck_input: x, ground_truth_input: y,
                                                    'dense_1/InTrainingMode:0': True})
                # train_writer.add_summary(summary, i)
                # sess.run(train_step, feed_dict={bottleneck_input: np.array(x).reshape(batch_size, bottleneck_tensor_size), ground_truth_input: np.array(y).reshape(batch_size)})



                if (i) % display_step == 0:
                    print("results for labels from training set:")
                    print(zip(y, results))
                    # print("results", str(results))
                    # print("for:", str(y))
                    # print("():", str(x[0]))

                    # results = sess.run([final_tensor],
                    #                    feed_dict={bottleneck_input: x, ground_truth_input: y})
                    # print("resultsZ1", str(results))
                    # print("for:", str(y))
                    # print("():", str(x[0]))
                    #
                    # results = sess.run([final_tensor],
                    #                    feed_dict={bottleneck_input: x, ground_truth_input: y})
                    # print("resultsZ1bis", str(results))
                    # print("for:", str(y))
                    # print("():", str(x[0]))
                    #

                    #
                    # btln_list2 = []
                    # btln_gt = []
                    # with open("trainer/tester/imgs/_5185_9483.jpeg_inception_v3.txt", 'r') as bottleneck_file2:
                    #     bottleneck_string2 = bottleneck_file2.read()
                    # try:
                    #     bottleneck_values2 = [float(x) for x in bottleneck_string2.split(',')]
                    # except ValueError:
                    #     tf.logging.warning('Invalid float found, recreating bottleneck')
                    #
                    # x.append(bottleneck_values2)
                    # y.append(205)
                    # results = sess.run([final_tensor],
                    #                    feed_dict={bottleneck_input: x})
                    # print("resultsZ", str(results))
                    # print("for:", str(y))
                    # print("():", str(x[0]))
                    #
                    # del x[15]
                    # del y[15]
                    # results = sess.run([final_tensor],
                    #                    feed_dict={bottleneck_input: x})
                    # print("resultsZ-", str(results))
                    # print("for:", str(y))
                    # print("():", str(x[0]))
                    #
                    # del x[0:14]
                    # del y[0:14]
                    # results = sess.run([final_tensor],
                    #                    feed_dict={bottleneck_input: x})
                    # print("results_2", str(results))
                    # print("for:", str(255))
                    # print("():", str(x))
                    #
                    # del x[0]
                    # results = sess.run([final_tensor],
                    #                    feed_dict={bottleneck_input: x})
                    # print("results_3", str(results))
                    # print("for:", str(255))
                    # print("():", str(x))
                    #



                    c = sess.run(MAE, feed_dict={bottleneck_input: all_test_bottlenecks,
                                                 ground_truth_input: all_test_ground_truth})
                    # validation_writer.add_summary(summary, i)
                    # c = sess.run(MAE, feed_dict={bottleneck_input: train_X, ground_truth_input: train_Y})
                    cost_y.append(c)
                    tf.logging.info("validation MAE: " + '%.5f' % c)

                    bottleneck_values = run_bottleneck_on_image(
                        sess, image_data, jpeg_data_tensor, decoded_image_tensor,
                        resized_image_tensor, bottleneck_tensor)

                    # img_result = sess.run('Mul_1:0', {
                    #     'DecodeJPGInput:0': image_data
                    # })
                    # img_result = np.reshape(img_result, (299, 299, 3))
                    # print("decoded img bytes: " + str(img_result))

                    # print("bottlen img bytes: " + str(bottleneck_values))
                    # print(img_result.size)
                    # print(bottleneck_values.size)
                    bottleneck_values = np.reshape(bottleneck_values, (1, 2048))
                    # [result, bias, pool, btln] = sess.run(['Reshape:0', 'dense_2/bias/read:0', 'pool_3/_reshape:0', 'BottleneckInput:0'], {
                    #     'DecodeJpeg:0': img_result
                    # })
                    # print("bott_ou img bytes: " + str(pool))
                    # print("bott_in img bytes: " + str(btln))
                    img_name = os.path.basename(img_dir)
                    # print("Image: " + img_name + " age: " + str(result))
                    # print("dense 2 bias:" + str(bias[0:5]))

                    # [result, bias, pool, btln] = sess.run(
                    #     ['Reshape:0', 'dense_2/bias/read:0', 'pool_3/_reshape:0', 'BottleneckInput:0'], {
                    #         'BottleneckInput:0': bottleneck_values
                    #     })
                    # print("xott_ou img bytes: " + str(pool))
                    # print("bott_in img bytes: " + str(btln))
                    # img_name = os.path.basename(img_dir)
                    # print("Image: " + img_name + " age: " + str(result))
                    # print("dense 2 bias:" + str(bias[0:5]))

                    # btln_list = []
                    # bottleneck_string = ','.join(str(x) for x in bottleneck_values)
                    # with open("trainer/tester/imgs/"+img_name+".txt", 'w') as bottleneck_file:
                    #     bottleneck_file.write(bottleneck_string)
                    # with open("trainer/tester/imgs/"+img_name+".txt", 'r') as bottleneck_file:
                    #     bottleneck_string = bottleneck_file.read()
                    # try:
                    #     bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
                    # except ValueError:
                    #     tf.logging.warning('Invalid float found, recreating bottleneck')
                    # btln_list.append(bottleneck_values)


                    #
                    # btln_list = np.reshape(btln_list, (1, 2048))
                    [result] = sess.run(
                        [final_tensor], {
                            bottleneck_input: bottleneck_values
                        })
                    # print("xxtt_ou img bytes: " + str(pool))
                    # print("bott_in img bytes: " + str(btln))
                    # img_name = os.path.basename(img_dir)
                    print("XImage: " + img_name + " age: " + str(result))
                    # print("dense 2 bias:" + str(bias[0:5]))

                    # resultsx = sess.run([final_tensor],
                    #                     feed_dict={bottleneck_input: btln_list, ground_truth_input: [204]})
                    # print("resultsx", str(resultsx))
                    # print("for:", str([204]))
                    # print("():", str(btln_list[0]))
                    #
                    # ###
                    # btln_list2 = []
                    # btln_gt = []
                    # with open("trainer/tester/imgs/_5185_9483.jpeg_inception_v3.txt", 'r') as bottleneck_file2:
                    #     bottleneck_string2 = bottleneck_file2.read()
                    # try:
                    #     bottleneck_values2 = [float(x) for x in bottleneck_string2.split(',')]
                    # except ValueError:
                    #     tf.logging.warning('Invalid float found, recreating bottleneck')
                    # for i in range(16):
                    #     btln_list2.append(bottleneck_values2)
                    #     btln_gt.append(204)
                    #
                    # resultsxx = sess.run([final_tensor],
                    #                     feed_dict={bottleneck_input: btln_list2, ground_truth_input: btln_gt})
                    # # results, _ = sess.run([final_tensor, train_step],
                    # #                       feed_dict={bottleneck_input: x, ground_truth_input: y})
                    #
                    # print("resultsxxx", str(resultsxx))
                    # print("for:", str([204]))
                    # print("():", str(btln_list2[0]))

                # Display logs per epoch step
                # tf.logging.log_every_n(tf.logging.INFO, ("MAE:", c), 20)
                # if (i) % display_step == 0:
                #     print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))  # ,# \
                #
                i += 1



                # # Store intermediate results
                # intermediate_frequency = FLAGS.intermediate_store_frequency
                #
                # if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
                #     and i > 0):
                #   # If we want to do an intermediate save, save a checkpoint of the train
                #   # graph, to restore into the eval graph.
                #   train_saver.save(sess, CHECKPOINT_NAME)
                #   intermediate_file_name = (FLAGS.intermediate_output_graphs_dir +
                #                             'intermediate_' + str(i) + '.pb')
                #   tf.logging.info('Save intermediate result to : ' +
                #                   intermediate_file_name)
                #   save_graph_to_file(graph, intermediate_file_name, model_info,
                #                      class_count)

            # train_saver.save(sess, CHECKPOINT_NAME)
            # save every epoch
            saver.save(sess, CHECKPOINT_NAME + 'model_iter', global_step=j)
            tf.logging.info("checkpoint saved in " + CHECKPOINT_NAME)

        # after all the epochs save final model for the last time

        winsound.Beep(640, 100)
        winsound.Beep(540, 500)
        winsound.Beep(440, 1000)

        saver.save(sess, CHECKPOINT_NAME + 'model_final')
        tf.logging.info("training completed. model saved in " + CHECKPOINT_NAME)
        print(time.time())
        print(datetime.datetime.now())

        ####################################


        start_time = time.time()
        i = 0
        # for (x, y) in zip(train_X, train_Y):
        #     print("elo:", x, y)
        # Run the initializer
        # sess.run(init)

        # # Fit all training data
        # for epoch in range(training_epochs):
        #     # add batch size
        #     train_X, train_Y = shuffle_data(train_X, train_Y)
        #
        #     for batch_index in batch_index_list:
        #         i+=1
        #
        #         x = train_X[batch_index:batch_index+batch_size]
        #         y = train_Y[batch_index:batch_index+batch_size]
        #         #for (x, y) in zip(train_X, train_Y):
        #         #print("elo:", x, y)
        #         # sess.run(optimizer, feed_dict={X: x, Y: y})
        #         sess.run(train_step, feed_dict={bottleneck_input: x, ground_truth_input: y})
        #         #sess.run(train_step, feed_dict={bottleneck_input: np.array(x).reshape(batch_size, bottleneck_tensor_size), ground_truth_input: np.array(y).reshape(batch_size)})
        #
        #         c = sess.run(MAE, feed_dict={bottleneck_input: train_X, ground_truth_input:train_Y})
        #         cost_y.append(c)
        #         # Display logs per epoch step
        #         if (i) % display_step == 0:
        #             print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))#,# \
        #
        # print("TIME:", time.time()-start_time)
        # print("Optimization Finished!")
        # # training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        # training_cost = sess.run(MAE, feed_dict={bottleneck_input: train_X, ground_truth_input:train_Y})
        # print("Training cost=", training_cost)#, "W=", sess.run(W), "b=", sess.run(b), '\n')
        #
        # # convert data
        # train_X = np.array(train_X)
        # train_Y = np.array(train_Y)
        # test_X = np.array(test_X)
        # test_Y = np.array(test_Y)

        # Graphic display

        # cost function
        cost_x = np.arange(len(cost_y))

        cost_epochs_x = np.arange(0, len(cost_y), int(len(cost_y) / epochs))
        cost_epochs_y = []
        for x in cost_epochs_x:
            cost_epochs_y.append(cost_y[x])
        plt.plot(cost_x, cost_y, 'b-', label='MAE')
        plt.plot(cost_epochs_x, cost_epochs_y, 'r.', label='MAE after epochs')
        plt.legend()
        plt.show()



        # # print(np.array(train_X)[:,0])
        # # print(np.array(train_Y))
        # plt.plot(train_X[:,0], train_Y, 'r.', label='Original data')
        # # plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
        # print(sess.run(final_tensor, feed_dict={bottleneck_input: train_X}))
        # plt.plot(train_X[:,0], sess.run(final_tensor, feed_dict={bottleneck_input: train_X}), 'bx', label='Fitted line')
        # plt.legend()
        # plt.show()
        #
        #
        #
        # print("Testing... (Mean square loss Comparison)")
        # testing_cost = sess.run(MAE, feed_dict={bottleneck_input: test_X, ground_truth_input: test_Y})
        # print("Testing cost=", testing_cost)
        # print("Absolute mean square loss difference:", abs(
        #     training_cost - testing_cost))
        #
        # plt.plot(test_X[:,0], test_Y, 'g.', label='Testing data')
        # plt.plot(test_X[:, 0], sess.run(final_tensor, feed_dict={bottleneck_input: test_X}), 'bx', label='Fitted line')
        # plt.legend()
        # plt.show()

except Exception as e:
    logging.error(traceback.format_exc())
    winsound.Beep(600, 200)
    winsound.Beep(600, 200)
    winsound.Beep(600, 200)
finally:
    winsound.Beep(600, 100)
    winsound.Beep(400, 300)
    winsound.Beep(300, 500)