from __future__ import print_function

import time
import traceback
import logging

from trainer.params_extractor import extract_parameters
from trainer.tf_methods import *
from trainer.graph_final_layer import append_final_layer_to_graph
import pickle
import winsound
import datetime
import argparse

rng = np.random

# CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'
# CHECKPOINT_NAME = 'trainer/trained_model3/'
# CHECKPOINT_NAME = 'trainer/trained_model14-15F_lite/'
d = 'C:/Users/Emilia/Pycharm Projects/BoneAge/'
CHECKPOINT_NAME = d + 'trainer/trained_model_FM7/'

FLAGS = None

# Parameters
learning_rate = 0.001
display_step = 50
checkpoint_every_n_batches_step = 4000

batch_size = 16
bottleneck_tensor_size = 2048

cost_y = []

is_training = 1


def extract_genders(genders):
    return np.tile(np.reshape(genders, (-1, 1)), [1, 32])


def save_curr_model(_dir, file_name, _start_time_date, global_step):
    if global_step:
        saver.save(sess, _dir + file_name, global_step=global_step)
    else:
        saver.save(sess, _dir + file_name)
    tf.logging.info("checkpoint saved in " + _dir)
    print("current time:" + str(datetime.datetime.now()))
    print("from:" + str(_start_time_date))


if __name__ == '__main__':
    name = '13'
    # image_dir_folder = 'three_classes'
    image_dir_folder = 'FM_labeled_train_validate'
    #image_dir_folder = 'X'

    epochs = 10
    # if there are already bottlenecks created, it does not look for what it should look for (if set to 0).
    # so if new datasets, set to 1 (or change bottlenecks manually)
    create_bottlenecks = 0

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
     final_tensor, MAE, train_step, gender_input) = append_final_layer_to_graph(graph, bottleneck_tensor, bottleneck_tensor_size,
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

        train_x_writer = tf.summary.FileWriter(
            FLAGS.summaries_dir + '/train_whole_set')

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

            start_time = time.time()
            start_time_date = datetime.datetime.now()
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
             all_test_ground_truth, _, _, all_test_genders) = get_random_cached_bottlenecks(
                sess, bottleneck_rnd_test, -1,
                FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                FLAGS.architecture)

            bottleneck_rnd_train = BottlenecksRandomizer('training', epoch_image_lists)

            tf.logging.info("image list labels:")
            tf.logging.info(len(list(epoch_image_lists.keys())))

            tf.logging.info("starting epoch %d" % j)

            merged = tf.summary.merge_all()
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

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


                # przypisanie
                # x = train_bottlenecks
                # y = train_ground_truth
                (x,
                 y, _, epoch_image_lists, train_genders) = get_random_cached_bottlenecks(
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

                # x = train_bottlenecks
                # y = train_ground_truth

                print("genders")
                print(train_genders)

                #train_genders = np.reshape(train_genders, (-1, 1))  #
                #train_genders_32 = extract_genders(train_genders)

                # extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                train_summary, results, _, _ = sess.run([merged, final_tensor, train_step, extra_update_ops],
                                         feed_dict={bottleneck_input: x, ground_truth_input: y,
                                                    gender_input: extract_genders(train_genders),
                                                    'dense_1/InTrainingMode:0': True})
                train_writer.add_summary(train_summary, i)
                # sess.run(train_step, feed_dict={bottleneck_input: np.array(x).reshape(batch_size, bottleneck_tensor_size), ground_truth_input: np.array(y).reshape(batch_size)})



                if (i) % display_step == 0:
                    print("results for labels from training set: (label, res, gender)")
                    print(list(zip(y, results, train_genders))[0:5])
                    # print("results", str(results))
                    # print("for:", str(y))
                    # print("():", str(x[0]))


                    validation_summary, c = sess.run([merged, MAE],
                                                     feed_dict={bottleneck_input: all_test_bottlenecks,
                                                                ground_truth_input: all_test_ground_truth,
                                                                gender_input: extract_genders(all_test_genders)})
                    validation_writer.add_summary(validation_summary, i)
                    # c = sess.run(MAE, feed_dict={bottleneck_input: train_X, ground_truth_input: train_Y})
                    # XXXXXX
                    # cost_y.append(c)
                    tf.logging.info("validation MAE: " + '%.5f' % c)
                    tf.logging.info("for batch of size: " +  str(len(all_test_ground_truth)))

                    ####### todo mae from whole training set

                    # bottleneck_rnd_train_x = BottlenecksRandomizer('training', whole_image_lists)
                    # (all_train_bottlenecks,
                    #  all_train_ground_truth, _, _, all_train_genders) = get_random_cached_bottlenecks(
                    #     sess, bottleneck_rnd_train_x, -1,
                    #     FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                    #     decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                    #     FLAGS.architecture)
                    #
                    # train_x_summary, c = sess.run([merged, MAE],
                    #                                  feed_dict={bottleneck_input: all_train_bottlenecks,
                    #                                             ground_truth_input: all_train_ground_truth,
                    #                                             gender_input: extract_genders(all_train_genders)})
                    # train_x_writer.add_summary(train_x_summary, i)
                    ########

                    # c = sess.run(MAE, feed_dict={bottleneck_input: train_X, ground_truth_input: train_Y})

                    # cost_y.append(c)
                    # tf.logging.info("train whole set MAE: " + '%.5f' % c)
                    # tf.logging.info("for batch of size: " + str(len(all_train_ground_truth)))

                    # XXXXXX

                    # slendr = [5, 2, 2, 5, 2, 5]
                    # gender = np.array([1, 0, 0, 1, 0, 1])
                    #
                    # indices = np.nonzero(gender > 0)[0]
                    #
                    # print(indices)
                    # print([slendr[x] for x in indices])

                    # divide validation set for M and F
                    M_indices = np.nonzero(np.asarray(all_test_genders) > 0)[0]
                    F_indices = np.nonzero(np.asarray(all_test_genders) < 1)[0]

                    # run test Male
                    if len(M_indices) > 0:
                        test_btln_M = [all_test_bottlenecks[x] for x in M_indices]
                        test_grnt_M = [all_test_ground_truth[x] for x in M_indices]
                        test_gend_M = [all_test_genders[x] for x in M_indices]

                        res, c = sess.run([final_tensor, MAE],
                                          feed_dict={bottleneck_input: test_btln_M,
                                                     ground_truth_input: test_grnt_M,
                                                     gender_input: extract_genders(test_gend_M)})
                        #cost_y.append(c)
                        tf.logging.info("validation M MAE: " + '%.5f' % c)
                        tf.logging.info("for batch of size: " + str(len(test_gend_M)))
                        tf.logging.info(list(zip(test_grnt_M, res, test_gend_M))[0:5])

                    # run test Female
                    if len(F_indices) > 0:
                        test_btln_F = [all_test_bottlenecks[x] for x in F_indices]
                        test_grnt_F = [all_test_ground_truth[x] for x in F_indices]
                        test_gend_F = [all_test_genders[x] for x in F_indices]

                        res, c = sess.run([final_tensor, MAE],
                                          feed_dict={bottleneck_input: test_btln_F,
                                                     ground_truth_input: test_grnt_F,
                                                     gender_input: extract_genders(test_gend_F)})
                        #cost_y.append(c)
                        tf.logging.info("validation F MAE: " + '%.5f' % c)
                        tf.logging.info("for batch of size: " + str(len(test_gend_F)))
                        tf.logging.info(list(zip(test_grnt_F, res, test_gend_F))[0:5])

                    tf.logging.info("-----")


                # Display logs per epoch step
                # tf.logging.log_every_n(tf.logging.INFO, ("MAE:", c), 20)
                # if (i) % display_step == 0:
                #     print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))  # ,# \
                #

                i += 1

                if i % checkpoint_every_n_batches_step is 0:
                    # save every x
                    save_curr_model(CHECKPOINT_NAME, 'model_iter_i', start_time_date, i)

            # save every epoch
            save_curr_model(CHECKPOINT_NAME, 'model_iter', start_time_date, j)

        # after all the epochs save final model for the last time
        winsound.Beep(640, 100)
        winsound.Beep(540, 500)
        winsound.Beep(440, 1000)

        save_curr_model(CHECKPOINT_NAME, 'model_final', start_time_date)

        ####################################


except Exception as e:
    logging.error(traceback.format_exc())
    winsound.Beep(600, 200)
    winsound.Beep(600, 200)
    winsound.Beep(600, 200)
finally:
    winsound.Beep(600, 100)
    winsound.Beep(400, 300)
    winsound.Beep(300, 500)