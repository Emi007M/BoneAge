import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import os.path
import numpy as np

from tensorflow.python.platform import gfile

img_dir = "imgs/3121.jpg" # age 125 zwraca  2.33032274  61.44803619
# img_dir = "imgs/10434.jpg" # age 126 zwraca  2.33032894 61.44826126
#img_dir = "imgs/2024.jpg" # age 126 zwraca              61.44814682
#img_dir = "imgs/_801_4808766.jpeg" # age 126 zwraca     61.44825745
#img_dir = "imgs/_5186_8315726.jpeg" # age 204 zwraca    61.44829178



img_name = os.path.basename(img_dir)
bottleneck = None
gender = 1  # 0 or 1


model_dir = "../trained_model2/"



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




def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])

  #   features['x'] = (features['x'] - min_x) / (max_x - min_x)
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)
  result2 = sess.run(resized)

  # print("xx", result)
  # print("xx2", result2)

  res = tf.squeeze(result, 0)

  return result[0]

def get_gender(i):
    if i is 1:
        return "M"
    return "F"



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



bottleneck_tensor_name = 'pool_3/_reshape:0'
bottleneck_tensor_size = 2048
input_width = 299
input_height = 299
input_depth = 3
resized_input_tensor_name = 'Mul:0'
model_file_name = 'classify_image_graph_def.pb'
input_mean = 128
input_std = 128

# jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
#         input_width, input_height,
#         input_depth, input_mean,
#         input_std)


jpeg_data_tensor = 'DecodeJPGInput:0'
decoded_image_tensor = 'Mul_1:0'

tensor = read_tensor_from_image_file(img_dir)
gt = np.array(0.0).reshape(1,)

image_data = gfile.FastGFile(img_dir, 'rb').read()

#####


tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph(model_dir+"model_final.meta")
graph = tf.get_default_graph()

with tf.Session() as sess:
    imported_meta.restore(sess, tf.train.latest_checkpoint(model_dir))

    # graph_def = graph.as_graph_def()
    # for node in graph_def.node:
    #     print("name: ", node.name)

    # final_tensor = graph.get_tensor_by_name('Reshape:0')
    final_tensor = graph.get_operation_by_name('Reshape').outputs[0]

    input_operation = graph.get_operation_by_name("DecodeJPGInput")
    input_operation2 = graph.get_operation_by_name("DecodeJpeg")
    bottleneck_input = graph.get_tensor_by_name('BottleneckInput:0')

    # result = sess.run([final_tensor], {
    #     # ground_truth_input: gt,
    #     input_operation.outputs[0]: tensor
    #     # 'DecodeJpeg:0': tensor
    #
    # })

    bottleneck_values = run_bottleneck_on_image(
        sess, image_data, jpeg_data_tensor, decoded_image_tensor,
        'Mul:0', 'pool_3/_reshape:0')

    # img_result = sess.run('Mul_1:0', {
    #     'DecodeJPGInput:0': image_data
    # })
    # img_result = np.reshape(img_result, (299, 299, 3))

    # img_result = sess.run('Mul:0', {
    #     'DecodeJPGInput:0': image_data
    # })

    # print("decoded img bytes: " + str(img_result[0][0][0]))

    # result, bias = sess.run(['Reshape:0', 'dense_2/bias/read:0'], {
    #     'DecodeJpeg:0': np.reshape(img_result, (299, 299, 3))
    # })

    bottleneck_values = np.reshape(bottleneck_values, (1, 2048))

    result, bias = sess.run(['Reshape:0', 'dense_2/bias/read:0'], {
        'BottleneckInput:0': bottleneck_values
    })


    print("Image: " + img_name + "(" + get_gender(gender) + ")" + " age: " + str(result))
    print("dense 2 bias:" + str(bias[0:5]))


