
import tensorflow as tf
import numpy as np
import os


def batch_norm(x, scope, is_training, epsilon=0.001, decay=0.99):
    """
    Returns a batch normalization layer that automatically switch between train and test phases based on the
    tensor is_training

    Args:
        x: input tensor
        scope: scope name
        is_training: boolean tensor or variable
        epsilon: epsilon parameter - see batch_norm_layer
        decay: epsilon parameter - see batch_norm_layer

    Returns:
        The correct batch normalization layer based on the value of is_training
    """
    assert isinstance(is_training, (ops.Tensor, variables.Variable)) and is_training.dtype == tf.bool

    return tf.cond(
        is_training,
        lambda: batch_norm_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=None),
        lambda: batch_norm_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),
    )


def batch_norm_layer(x, scope, is_training, epsilon=0.001, decay=0.99, reuse=None):
    """
    Performs a batch normalization layer

    Args:
        x: input tensor
        scope: scope name
        is_training: python boolean value
        epsilon: the variance epsilon - a small float number to avoid dividing by 0
        decay: the moving average decay

    Returns:
        The ops of a batch normalization layer
    """
    with tf.variable_scope(scope, reuse=reuse):
        shape = x.get_shape().as_list()
        # gamma: a trainable scale factor
        gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
        # beta: a trainable shift value
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
        moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
        if is_training:
            # tf.nn.moments == Calculate the mean and the variance of the tensor x
            avg, var = tf.nn.moments(x, range(len(shape)-1))
            update_moving_avg = moving_averages.assign_moving_average(moving_avg, avg, decay)
            update_moving_var = moving_averages.assign_moving_average(moving_var, var, decay)
            control_inputs = [update_moving_avg, update_moving_var]
        else:
            avg = moving_avg
            var = moving_var
            control_inputs = []
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

    return output

def append_final_layer_to_graph(graph, bottleneck_tensor, bottleneck_tensor_size, learning_rate):

    with graph.as_default():

        gender_input = tf.placeholder(tf.float32, [None, 32], 'GenderInput')
        # g = tf.transpose(gender_input)
        # print(tf.shape(g)[0:1])
        # #gender_i = tf.tile(g, [32, 1])

        #gender_i = tf.tile(g, tf.reshape(32, [1]))
        # gender_i = tf.tile(g, tf.shape(g)[0:1])
        # print(tf.shape(gender_i))
        # g = tf.transpose(gender_i)
        print(tf.shape(gender_input))

        dense_gender = tf.layers.dense(inputs=gender_input, units=32, activation=tf.nn.relu, name="dense_gender",
                                bias_initializer=tf.constant_initializer(np.full_like((1, 32), 0.1, dtype=np.double))) #changed from ones

        tf.summary.histogram('after_dense_g', dense_gender)
        dg_weights = tf.get_default_graph().get_tensor_by_name(os.path.split(dense_gender.name)[0] + '/kernel:0')
        tf.summary.histogram('dense_g_weights', dg_weights)
        dg_biases = tf.get_default_graph().get_tensor_by_name(os.path.split(dense_gender.name)[0] + '/bias:0')
        tf.summary.histogram('dense_g_biases', dg_biases)
        # dg_activations = tf.get_default_graph().get_tensor_by_name(os.path.split(dense_gender.name)[0] + '/activations:0')
        # tf.summary.histogram('dense_g_activations', dg_activations)



        #was np.ones((1, 32)
        # bottleneck_input = tf.placeholder(tf.float32, [None, bottleneck_tensor_size], 'BottleneckInput')
        bottleneck_input = tf.placeholder_with_default(bottleneck_tensor, [None, bottleneck_tensor_size], #got back
                                                       'BottleneckInput')
        ground_truth_input = tf.placeholder(tf.float32, [None], 'GroundTruthInput')

        # with tf.name_scope('dense_1'):
        #     w2_initial = np.random.normal(size=(bottleneck_tensor_size, 1000)).astype(np.float32)
        #     epsilon = 1e-3
        #
        #     w2_BN = tf.get_variable("w2_BN", initializer=w2_initial)
        #
        #     #todo
        #     # input będzie tf.concat([dense_gender, bottleneck_input], 0)
        #     z2_BN = tf.matmul(bottleneck_input, w2_BN)
        #     batch_mean2, batch_var2 = tf.nn.moments(z2_BN, [0])
        #     scale2 = tf.get_variable("scale2", initializer=tf.ones([1000]))
        #     beta2 = tf.get_variable("beta2", initializer=tf.zeros([1000]))
        #     BN2 = tf.nn.batch_normalization(z2_BN, batch_mean2, batch_var2, beta2, scale2, epsilon)
        #     dense1 = tf.nn.relu(BN2, name="dense_1_activation_f")



        with tf.name_scope('dense_1'):
            in_training_mode = tf.placeholder_with_default(False, shape=None, name='InTrainingMode')

            # byl input > bottleneck_input <, zmienione na concat
            merged_input = tf.concat([dense_gender, bottleneck_input], 1)
            batch_normed = tf.layers.batch_normalization(
                inputs=merged_input,
                axis=-1,
                momentum=0.999,
                epsilon=1e-3,
                center=True,
                scale=True,
                training=in_training_mode
            )


            # hidden = tf.layers.dense(inputs=bottleneck_input, units=1000, activation=tf.nn.relu,
            #                 bias_initializer=tf.constant_initializer(np.ones((1, 1000))))
            # # hidden = tf.keras.layers.Dense(n_units,
            # #                                activation=None)(X)  # no activation function, yet
            # batch_normed = tf.keras.layers.BatchNormalization()(hidden, training=in_training_mode)
            # # dense1 = tf.keras.activations \
            #     .relu(batch_normed)  # ReLu is typically done after batch normalization
            dense1 = tf.nn.relu(batch_normed, name="dense_1_activation_f")

            tf.summary.histogram('dense_1_batch_normed', batch_normed)
            tf.summary.histogram('after_dense_1_relu', dense1)

        dense2 = tf.layers.dense(inputs=dense1, units=1000, activation=tf.nn.relu, name="dense_2",
                                 bias_initializer=tf.constant_initializer(np.full_like((1, 1000), 0.1, dtype=np.double))) #changed from ones np.zeros((1, 1000))
        
        tf.summary.histogram('after_dense_2', dense2)
        d2_weights = tf.get_default_graph().get_tensor_by_name(os.path.split(dense2.name)[0] + '/kernel:0')
        tf.summary.histogram('dense_2_weights', d2_weights)
        d2_biases = tf.get_default_graph().get_tensor_by_name(os.path.split(dense2.name)[0] + '/bias:0')
        tf.summary.histogram('dense_2_biases', d2_biases)
        # d2_activations = tf.get_default_graph().get_tensor_by_name(os.path.split(dense2.name)[0] + '/activation:0')
        # tf.summary.histogram('dense_2_activations', d2_activations)


        dense3 = tf.layers.dense(
            inputs=dense2,
            units=1,
            activation=None,
            name="output",
            bias_initializer=tf.constant_initializer(0.1) #changed from one
        )

        tf.summary.histogram('after_dense_3', dense3)
        d3_weights = tf.get_default_graph().get_tensor_by_name(os.path.split(dense3.name)[0] + '/kernel:0')
        tf.summary.histogram('dense_3_weights', d3_weights)
        d3_biases = tf.get_default_graph().get_tensor_by_name(os.path.split(dense3.name)[0] + '/bias:0')
        tf.summary.histogram('dense_3_bias', d3_biases)
        # d3_activations = tf.get_default_graph().get_tensor_by_name(os.path.split(dense3.name)[0] + '/activation:0')
        # tf.summary.histogram('dense_3_activations', d3_activations)

        final_tensor = tf.reshape(dense3, [-1])

        with tf.name_scope('MAE'):
            losses = tf.squared_difference(tf.cast(final_tensor, tf.float32), tf.cast(ground_truth_input, tf.float32))
            absolute_losses = tf.sqrt(losses, name="MAE")
            MAE = tf.reduce_mean(absolute_losses)

            tf.summary.scalar('MAE', MAE)

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate, name="ADAM_optimizer")
            train_step = optimizer.minimize(MAE, name="train_step")

        return bottleneck_input, ground_truth_input, final_tensor, MAE, train_step, gender_input
