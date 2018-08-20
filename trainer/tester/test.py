import tensorflow as tf
import os.path
import numpy as np
import csv
import collections
from trainer.tester.csv_methods import getImagesAndGenders

from tensorflow.python.platform import gfile

images = ["imgs/3121.jpg", "imgs/10434.jpg", "imgs/2024.jpg", "imgs/_801_4808766.jpeg", "imgs/_5186_8315726.jpeg",
          "imgs/M_803_3067288.jpeg"]
genders = [1, 1, 1, 1, 1, 1]

#images, genders = getImagesAndGenders()

# img_dir = "imgs/3121.jpg" # age 125
# img_dir = "imgs/10434.jpg" # age 126
# img_dir = "imgs/2024.jpg" # age 126
# img_dir = "imgs/_801_4808766.jpeg" # age 126
# img_dir = "imgs/_5186_8315726.jpeg" # age 204
# img_dir = "imgs/M_803_3067288.jpeg" # age 21

# gender = 1  # 0 or 1

d = 'C:/Users/Emilia/Pycharm Projects/BoneAge/'
# model_dir = "../trained_model3/"
# model_dir = d + "../trained_modelM1epoch/"
model_dir = d + "trainer/trained_model_FM9/"
# model_name = "model_final.meta"
model_name = "model_iter_i-292000.meta"

csv_dir = "M:/Desktop/csvs/t1.csv"
is_saving_to_csv = True

max_batch_size = 50


def scaleAge(value):
    "from real age <0;230> to <-1;1>"
    "(value * 2) / 230 - 1 "
    return float(value) / 115.0 - 1.0


def unscaleAgeL(valueList):
    return [(float(value) + 1.0) * 115.0 for value in np.squeeze(valueList)]


def unscaleAgeT(value):
    return tf.scalar_mul(115.0, tf.add(value, 1.0))


def extract_genders(genders):
    return np.tile(np.reshape(genders, (-1, 1)), [1, 32])


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


def prepare_batches(_len, max_size):
    r = list(range(_len))
    l = r[0::max_size]
    l.append(_len)
    return l


def gender_to_string(_i):
    if _i is 1:
        return "M"
    return "F"


def img_dir_get_name(_dir):
    return os.path.splitext(os.path.basename(_dir))[0]


def init_csv_file(_csv_dir):
    csv_data = []
    csv_data.append(["id", "boneage"])
    write_to_csv(_csv_dir, csv_data, 'w')


def write_to_csv(csv_dir, csv_data, mode='a'):
    writer = csv.writer(open(csv_dir, mode, newline=''))

    for row in csv_data:
        writer.writerow(row)

#####


tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph(model_dir + model_name)
graph = tf.get_default_graph()

if is_saving_to_csv:
    init_csv_file(csv_dir)

with tf.Session() as sess:
    imported_meta.restore(sess, tf.train.latest_checkpoint(model_dir))

    batches_indices = prepare_batches(len(images), max_batch_size)
    print(str(len(images)) + " images divided into " + str(len(batches_indices) - 1) + " batches:")


    for l_i, i in enumerate(batches_indices):
        if l_i is len(batches_indices) - 1:
            continue

        csv_data = []

        start = i
        end = batches_indices[l_i + 1]
        image_data = []
        for single_dir in images[start:end]:
            image_data.append(gfile.FastGFile(single_dir, 'rb').read())

        bottleneck_values = []
        for img in image_data:
            bottleneck_values.append(run_bottleneck_on_image(
                sess, img, 'DecodeJPGInput:0', 'Mul_1:0',
                'Mul:0', 'pool_3/_reshape:0'))

        result = sess.run(['Reshape:0'], {
            'BottleneckInput:0': bottleneck_values,
            "GenderInput:0": extract_genders(genders[start:end])
        })

        for img_dir, gender, age in zip(images[start:end], genders[start:end], unscaleAgeL(result)):
            print("Image: " + img_dir_get_name(img_dir) + "(" + gender_to_string(gender) + ")" + " age: " + str(
                int(round(age))))

            if is_saving_to_csv:
                csv_data.append([img_dir_get_name(img_dir), str(int(round(age)))])

        print("finished batch "+ str(l_i+1) +" from "+str(len(batches_indices) - 1))

        if is_saving_to_csv:
            write_to_csv(csv_dir, csv_data)
