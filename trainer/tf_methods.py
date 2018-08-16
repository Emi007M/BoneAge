import hashlib
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from shutil import copyfile
import pickle

from trainer.bottlenecks_randomizer import BottlenecksRandomizer

FLAGS = None
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
# The location where variable checkpoints will be stored.
CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'

def initFlags(_flags):
    global FLAGS
    FLAGS = _flags


def scaleAge(value):
    "from real age <0;230> to <-1;1>"
    "(value * 2) / 230 - 1 "
    return value / 115 - 1

def unscaleAge(value):
    return (value + 1) * 115


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """Builds a list of training images from the file system.

  Analyzes the sub folders in the image directory, splits them into stable
  training, testing, and validation sets, and returns a data structure
  describing the lists of images for each label and their paths.

  Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.

  Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
  """
    if not gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        if len(file_list) < 20:
            tf.logging.warning(
                'WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning(
                'WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # We want to ignore anything after '_nohash_' in the file name when
            # deciding which set to put an image in, the data set creator has a way of
            # grouping photos that are close variations of each other. For example
            # this is used in the plant disease data set to group multiple pictures of
            # the same leaf.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should
            # go into the training, testing, or validation sets, and we want to keep
            # existing files in the same set even if more files are subsequently
            # added.
            # To do that, we need a stable way of deciding based on just the file name
            # itself, so we do a hash of that and then use that to generate a
            # probability value that we use to assign it.
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


def create_image_lists_from_prepared_dir(image_dir):
    """Builds a list of training images from the file system.
    and returns a data structure
  describing the lists of images for each label and their paths.

  Args:
    image_dir: String path to a folder containing subfolders of images.

  Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
  """
    if not gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    result = {}

    image_dir_train = image_dir + '/train_augmented/'
    image_dir_validate = image_dir + '/validate_augmented/'

    print(image_dir_train)

    # initialize all labels sets
    print("initializing label sets")
    sub_dirs1 = [x[0] for x in gfile.Walk(image_dir_train)]
    sub_dirs1 = sub_dirs1[1:]
    sub_dirs2 = [x[0] for x in gfile.Walk(image_dir_validate)]
    sub_dirs2 = sub_dirs2[1:]

    sub_dirs = sub_dirs1 + sub_dirs2
    print("start")
    for sub_dir in sub_dirs:
        dir_name = os.path.basename(sub_dir)
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

        result[label_name] = {
            'dir': dir_name,
            'training': [],
            'testing': [],
            'validation': [],
        }


    print("subdirs1:")
    print(sub_dirs1)

    # fill in image_lists with training images
    result = fill_in_specified_set_with_labels(image_dir_train, result, sub_dirs1, 'training')
    # fill in image_lists with validation images
    result = fill_in_specified_set_with_labels(image_dir_validate, result, sub_dirs2, 'validation')

    return result


def fill_in_specified_set_with_labels(image_dir, result, sub_dirs, specified_set):
    for sub_dir in sub_dirs:
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        print("ee")
        print(dir_name)
        print(image_dir)
        if dir_name == image_dir:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        if len(file_list) < 20:
            tf.logging.warning(
                'WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning(
                'WARNING: Folder {} has more than {} images. Some images will '
                'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

        images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            images.append(base_name)

        # result[label_name][specified_set] = images
        result[label_name].update({specified_set: images})

    return result


def get_image_path(image_lists, label_name, index, image_dir, category, is_for_bottleneck_path = 0):
    """"Returns a path to an image for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.

  """

    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]

    # added for pre-divided dataset into training and validation sets


    if is_for_bottleneck_path:
        sub_dir = label_lists['dir']
    else:
        if category is 'training':
            category_folder = 'train_augmented/'
        else:  # 'validation', could be also 'testing'
            category_folder = 'validate_augmented/'
        sub_dir = category_folder + label_lists['dir']

    image_dir += '/'
    sub_dir += '/'
    full_path = os.path.join(image_dir, sub_dir, base_name)
    # print("cc")
    # print(image_dir)
    # # print(category_folder)
    # print(sub_dir)
    # print(base_name)
    # print(full_path)
    # print("cx")
    return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category, architecture):
    """"Returns a path to a bottleneck file for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    category: Name string of set to pull images from - training, testing, or
    validation.
    architecture: The name of the model architecture.

  Returns:
    File system path string to an image that meets the requested parameters.
  """
    return get_image_path(image_lists, label_name, index, bottleneck_dir,
                          category, is_for_bottleneck_path=1) + '_' + architecture + '.txt'


def create_model_graph(model_info):
    """"Creates a graph from saved GraphDef file and returns a Graph object.

  Args:
    model_info: Dictionary containing information about the model architecture.

  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(FLAGS.model_dir, model_info['model_file_name'])
        if not os.path.exists(os.path.join(FLAGS.saved_model_dir)):
            os.makedirs(os.path.join(FLAGS.saved_model_dir))
        copyfile(model_path, os.path.join(FLAGS.saved_model_dir, 'ssaved_model.pb'))
        #model_path = os.path.join(FLAGS.saved_model_dir, 'saved_model.pb')
        print('Model path: ', model_path)
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
                graph_def,
                name='',
                return_elements=[
                    model_info['bottleneck_tensor_name'],
                    model_info['resized_input_tensor_name'],
                ]))
    return graph, bottleneck_tensor, resized_input_tensor

def read_model_graph_after_epoch(model_info):
    """"Creates a graph from saved GraphDef file and returns a Graph object.

  Args:
    model_info: Dictionary containing information about the model architecture.

  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(FLAGS.saved_model_dir, 'ssaved_model.pb')
        print('Model path extract: ', model_path)
        # if not os.path.exists(model_path):
        #     os.makedirs(model_path)
        #     create_model_graph(model_info)
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
                graph_def,
                name='',
                return_elements=[
                    model_info['bottleneck_tensor_name'],
                    model_info['resized_input_tensor_name'],
                ]))
    return graph, bottleneck_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.

  Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    decoded_image_tensor: Output of initial image resizing and preprocessing.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: Layer before the final softmax.

  Returns:
    Numpy array of bottleneck values.
  """
    # First decode the JPEG image, resize it, and rescale the pixel values.
    resized_input_values = sess.run(decoded_image_tensor,
                                    {image_data_tensor: image_data})
    # Then run it through the recognition network.
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def maybe_download_and_extract(data_url):
    """Download and extract model tar file.

  If the pretrained model we're using doesn't already exist, this function
  downloads it from the TensorFlow.org website and unpacks it into a directory.

  Args:
    data_url: Web location of the tar file containing the pretrained model.
  """
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        tf.logging.info('Successfully downloaded %s %d bytes.', filename,
                        statinfo.st_size)
        print('Extracting file from ', filepath)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    else:
        print('Not extracting or downloading files, model already present in disk')


def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


bottleneck_path_2_bottleneck_values = {}


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
    """Create a single bottleneck file."""
    tf.logging.info('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index,
                                image_dir, category)

    # # added for pre-divided dataset into training and validation sets
    # if category is 'training':
    #     image_path = 'train_augmented/' + image_path
    # else: #'validation', could be also 'testing'
    #     image_path = 'validate_augmented/' + image_path

    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                     str(e)))
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor, architecture):
    """Retrieves or calculates bottleneck values for an image.

  If a cached version of the bottleneck data exists on-disk, return that,
  otherwise calculate the data and save it to disk for future use.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be modulo-ed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of which set to pull images from - training, testing,
    or validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: The tensor to feed loaded jpeg data into.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The output tensor for the bottleneck values.
    architecture: The name of the model architecture.

  Returns:
    Numpy array of values produced by the bottleneck layer for the image.
  """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          bottleneck_dir, category, architecture)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        tf.logging.warning('Invalid float found, recreating bottleneck')
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Allow exceptions to propagate here, since they shouldn't happen after a
        # fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, architecture):
    """Ensures all the training, testing, and validation bottlenecks are cached.

  Because we're likely to read the same image multiple times (if there are no
  distortions applied during training) it can speed things up a lot if we
  calculate the bottleneck layer values once for each image during
  preprocessing, and then just read those cached values repeatedly during
  training. Here we go through all the images we've found, calculate those
  values, and save them off.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    image_dir: Root folder string of the subfolders containing the training
    images.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: Input tensor for jpeg data from file.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The penultimate output layer of the graph.
    architecture: The name of the model architecture.

  Returns:
    Nothing.
  """
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    print("xxx")
    print(image_lists)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            if len(label_lists[category]) == 0:
                print("no images for " + category + 'for label ' + label_name)
                continue
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                if not category_list:
                    continue
                get_or_create_bottleneck(
                    sess, image_lists, label_name, index, image_dir, category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor, architecture)

                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    tf.logging.info(
                        str(how_many_bottlenecks) + ' bottleneck files created.')




def get_random_cached_bottlenecks(sess, bottleneck_rnd: BottlenecksRandomizer, how_many,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor, architecture):
    """Retrieves bottleneck values for cached images.

  If no distortions are being applied, this function can retrieve the cached
  bottleneck values directly from disk for images. It picks a random set of
  images from the specified category.

  Args:
    sess: Current TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    how_many: If positive, a random sample of this size will be chosen.
    If negative, all bottlenecks will be retrieved.
    category: Name string of which set to pull from - training, testing, or
    validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    image_dir: Root folder string of the subfolders containing the training
    images.
    jpeg_data_tensor: The layer to feed jpeg image data into.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.
    architecture: The name of the model architecture.

  Returns:
    List of bottleneck arrays, their corresponding ground truths, and the
    relevant filenames. optionally gender list
  """
    class_count = len(bottleneck_rnd.image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    genders = []


    if how_many >= 0:
        # Retrieve a random sample of bottlenecks.
        unused_i = 0
        while unused_i < how_many:
            if len(bottleneck_rnd.image_lists.keys()) < 1:
                tf.logging.info("too little image list " + len(bottleneck_rnd.image_lists.keys()))
                return bottlenecks, ground_truths, filenames, -1

            # get random bottleneck
            label_index, label_name, image_index = bottleneck_rnd.choose_random_bottleneck()

            image_name = get_image_path(bottleneck_rnd.image_lists, label_name, image_index,
                                        image_dir, bottleneck_rnd.category)
            bottleneck = get_or_create_bottleneck(
                sess, bottleneck_rnd.image_lists, label_name, image_index, image_dir, bottleneck_rnd.category,
                bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                resized_input_tensor, bottleneck_tensor, architecture)
            bottlenecks.append(bottleneck)
            ground_truths.append(label_name) # was label_index
            filenames.append(image_name)
            genders.append(1 if os.path.basename(image_name)[0] is 'M' else 0)
            unused_i += 1

            # pop from list
            bottleneck_rnd.clean_after_choosing(label_index, label_name, image_index)

    else:
        # Retrieve all bottlenecks.
        for label_index, label_name in enumerate(bottleneck_rnd.image_lists.keys()):
            for image_index, image_name in enumerate(
                    bottleneck_rnd.image_lists[label_name][bottleneck_rnd.category]):
                image_name = get_image_path(bottleneck_rnd.image_lists, label_name, image_index,
                                            image_dir, bottleneck_rnd.category)
                bottleneck = get_or_create_bottleneck(
                    sess, bottleneck_rnd.image_lists, label_name, image_index, image_dir, bottleneck_rnd.category,
                    bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor, architecture)
                bottlenecks.append(bottleneck)
                ground_truths.append(label_name) # was label_index
                filenames.append(image_name)

                genders.append(1 if os.path.basename(image_name)[0] is 'M' else 0)
    # print("genders")
    # print(genders)
    # genders = np.asarray(genders).reshape(len(genders),1)
    # # genders = np.transpose(genders)
    # print(genders)
    return bottlenecks, ground_truths, filenames, bottleneck_rnd.image_lists, genders






def save_graph_to_file(graph, graph_file_name, model_info, bottleneck_input,
         ground_truth_input, final_tensor):
    """Saves an graph to file, creating a valid quantized one if necessary."""
    sess, _, _ = build_eval_session(model_info, bottleneck_input,
         ground_truth_input, final_tensor)
    # graph = sess.graph

    # graph_def = sess.graph.as_graph_def()
    # for node in graph_def.node:
    #     print("node: ",node)

    # output_graph_def = graph_util.convert_variables_to_constants(
    #     sess, graph.as_graph_def(), ["input/final_retrain_ops/Wx_plus_b/final_result"])
    # output_graph_def = graph_util.convert_variables_to_constants(
    #     sess, graph.as_graph_def(), [FLAGS.final_tensor_name + ":0"])
    # output_graph_def = graph_util.convert_variables_to_constants(
    #     sess, graph.as_graph_def(), ["output/BiasAdd"])
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), ["Reshape"])

    if not os.path.exists(graph_file_name):
        os.makedirs(graph_file_name)
    with gfile.FastGFile(graph_file_name + '/graph.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())



def build_eval_session(model_info, bottleneck_input,
         ground_truth_input, final_tensor):
    """Builds an restored eval session without train operations for exporting.

  Args:
    model_info: Model info dictionary from create_model_info()
    class_count: Number of classes

  Returns:
    Eval session containing the restored eval graph.
    The bottleneck input, ground truth, eval step, and prediction tensors.
  """
    # If quantized, we need to create the correct eval graph for exporting.
    # eval_graph, bottleneck_tensor, _ = create_model_graph(model_info)
    eval_graph, bottleneck_tensor, _ = read_model_graph_after_epoch(model_info)

    eval_sess = tf.Session(graph=eval_graph)
    with eval_graph.as_default():
        # Add the new layer for exporting.
        # (_, _, bottleneck_input,
        #  ground_truth_input, final_tensor) = add_final_retrain_ops(
        #     class_count, FLAGS.final_tensor_name, bottleneck_tensor,
        #     model_info['bottleneck_tensor_size'], model_info['quantize_layer'],
        #     False)

        # Now we need to restore the values from the training graph to the eval
        # graph.
        tf.train.Saver().restore(eval_sess, CHECKPOINT_NAME)

        # evaluation_step, prediction = add_evaluation_step(final_tensor,
        #                                                   ground_truth_input)
        # print("final prediction:", prediction)
        # prediction = tf.cast(prediction, tf.int64)
        # print("eval for: ", ground_truth_input, final_tensor, ground_truth_input[0],final_tensor[0])

    return (eval_sess, bottleneck_input, ground_truth_input)


def prepare_file_system():
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    if FLAGS.intermediate_store_frequency > 0:
        ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
    return


def create_model_info(architecture):
    """Given the name of a model architecture, returns information about it.

  There are different base image recognition pretrained models that can be
  retrained using transfer learning, and this function translates from the name
  of a model to the attributes that are needed to download and train with it.

  Args:
    architecture: Name of a model architecture.

  Returns:
    Dictionary of information about the model, or None if the name isn't
    recognized

  Raises:
    ValueError: If architecture name is unknown.
  """
    architecture = architecture.lower()
    is_quantized = False
    if architecture == 'inception_v3':
        # pylint: disable=line-too-long
        data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        # pylint: enable=line-too-long
        bottleneck_tensor_name = 'pool_3/_reshape:0'
        bottleneck_tensor_size = 2048
        input_width = 299
        input_height = 299
        input_depth = 3
        resized_input_tensor_name = 'Mul:0'
        model_file_name = 'classify_image_graph_def.pb'
        input_mean = 128
        input_std = 128
    elif architecture.startswith('mobilenet_'):
        parts = architecture.split('_')
        if len(parts) != 3 and len(parts) != 4:
            tf.logging.error("Couldn't understand architecture name '%s'",
                             architecture)
            return None
        version_string = parts[1]
        if (version_string != '1.0' and version_string != '0.75' and
                    version_string != '0.5' and version_string != '0.25'):
            tf.logging.error(
                """"The Mobilenet version should be '1.0', '0.75', '0.5', or '0.25',
  but found '%s' for architecture '%s'""", version_string, architecture)
            return None
        size_string = parts[2]
        if (size_string != '224' and size_string != '192' and
                    size_string != '160' and size_string != '128'):
            tf.logging.error(
                """The Mobilenet input size should be '224', '192', '160', or '128',
 but found '%s' for architecture '%s'""",
                size_string, architecture)
            return None
        if len(parts) == 3:
            is_quantized = False
        else:
            if parts[3] != 'quant':
                tf.logging.error(
                    "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
                    architecture)
                return None
            is_quantized = True

        data_url = 'http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/'
        model_name = 'mobilenet_v1_' + version_string + '_' + size_string
        if is_quantized:
            model_name += '_quant'
        data_url += model_name + '.tgz'
        bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
        resized_input_tensor_name = 'input:0'
        model_file_name = model_name + '_frozen.pb'

        bottleneck_tensor_size = 1001
        input_width = int(size_string)
        input_height = int(size_string)
        input_depth = 3
        input_mean = 127.5
        input_std = 127.5
    else:
        tf.logging.error("Couldn't understand architecture name '%s'", architecture)
        raise ValueError('Unknown architecture', architecture)

    return {
        'data_url': data_url,
        'bottleneck_tensor_name': bottleneck_tensor_name,
        'bottleneck_tensor_size': bottleneck_tensor_size,
        'input_width': input_width,
        'input_height': input_height,
        'input_depth': input_depth,
        'resized_input_tensor_name': resized_input_tensor_name,
        'model_file_name': model_file_name,
        'input_mean': input_mean,
        'input_std': input_std,
        'quantize_layer': is_quantized,
    }


def add_jpeg_decoding(input_width, input_height, input_depth, input_mean,
                      input_std):
    """Adds operations that perform JPEG decoding and resizing to the graph..

  Args:
    input_width: Desired width of the image fed into the recognizer graph.
    input_height: Desired width of the image fed into the recognizer graph.
    input_depth: Desired channels of the image fed into the recognizer graph.
    input_mean: Pixel value that should be zero in the image for the graph.
    input_std: How much to divide the pixel values by before recognition.

  Returns:
    Tensors for the node to feed JPEG data into, and the output of the
      preprocessing steps.
  """
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return jpeg_data, mul_image


def get_or_create_image_lists():

    if FLAGS.create_bottlenecks:
        # image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
        #                                  FLAGS.validation_percentage)
        image_lists = create_image_lists_from_prepared_dir(FLAGS.image_dir)

        if not os.path.exists(FLAGS.bottleneck_dir):
            os.makedirs(FLAGS.bottleneck_dir)
        with open(FLAGS.bottleneck_dir + '/image_list_division.pkl', 'wb') as output:
            pickle.dump(image_lists, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open(FLAGS.bottleneck_dir + '/image_list_division.pkl', 'rb') as input:
            image_lists = pickle.load(input)
    return image_lists





def export_model(model_info, bottleneck_input,
         ground_truth_input, final_tensor, saved_model_dir):
    """Exports model for serving.

  Args:
    model_info: The modelinfo for the current model.
    class_count: The number of classes.
    saved_model_dir: Directory in which to save exported model and variables.
  """
    # The SavedModel should hold the eval graph.
    sess, _, _, _, _ = build_eval_session(model_info, bottleneck_input,
         ground_truth_input, final_tensor)
    graph = sess.graph
    with graph.as_default():
        input_tensor = model_info['resized_input_tensor_name']
        in_image = sess.graph.get_tensor_by_name(input_tensor)
        inputs = {'image': tf.saved_model.utils.build_tensor_info(in_image)}

        # out_classes = sess.graph.get_tensor_by_name('input/final_retrain_ops/Wx_plus_b/final_result:0')
        # out_classes = sess.graph.get_tensor_by_name('/' + FLAGS.final_tensor_name + ':0')
        # out_classes = sess.graph.get_tensor_by_name('output/BiasAdd:0')
        out_classes = sess.graph.get_tensor_by_name('Reshape:0')
        outputs = {
            'prediction': tf.saved_model.utils.build_tensor_info(out_classes)
        }

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        # Save out the SavedModel.
        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    signature
            },
            legacy_init_op=legacy_init_op)
        builder.save()

