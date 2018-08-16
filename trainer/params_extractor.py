
from datetime import datetime
from time import gmtime, strftime

def extract_parameters(parser, params):
    (name, image_dir_folder, epochs, create_bottlenecks) = params

    d = 'C:/Users/Emilia/Pycharm Projects/BoneAge/'

    # parser.add_argument(
    #     '--image_dir',
    #     type=str,
    #     default='training_dataset/' + image_dir_folder,
    #     help='Path to folders of labeled images.'
    # )
    # parser.add_argument(
    #     '--bottleneck_dir',
    #     type=str,
    #     default='model/bottleneck/' + image_dir_folder,
    #     help='Path to cache bottleneck layer values as files.'
    # )

    parser.add_argument(
        '--image_dir',
        type=str,
        default= d + 'training_dataset/'+ image_dir_folder,
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--bottleneck_dir',
        type=str,
        default= d + 'bottleneck/' + image_dir_folder,
        help='Path to cache bottleneck layer values as files.'
    )

    parser.add_argument(
        '--create_bottlenecks',
        type=str,
        default=create_bottlenecks,
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--output_graph',
        type=str,
        default= d +'model/output_graph/' + name + '/output_graph' + '-' + strftime("%m-%d %H.%M.%S", gmtime()),
        help='Where to save the trained graph.'
    )
    parser.add_argument(
        '--output_graph_fin',
        type=str,
        default='model',
        help='Where to save the trained graph.'
    )
    parser.add_argument(
        '--intermediate_output_graphs_dir',
        type=str,
        default= d +'model/intermediate_graph/' + name + '/',
        help='Where to save the intermediate graphs.'
    )
    parser.add_argument(
        '--intermediate_store_frequency',
        type=int,
        default=0,
        help="""\
         How many steps to store intermediate graph. If "0" then will not
         store.\
      """
    )
    parser.add_argument(
        '--output_labels',
        type=str,
        default= d +'model/output_labels.txt',
        help='Where to save the trained graph\'s labels.'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default= d +'model/models/retrain_logs/' + name + '-' + strftime("%Y-%m-%d %H.%M.%S", gmtime()),
        help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
        '--how_many_epochs',
        type=int,
        default= epochs,
        help='How many training steps to run before ending.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=0,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=15,
        help='What percentage of images to use as a validation set.'
    )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=50,
        help='How often to evaluate the training results.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=16,
        help='How many images to train on at a time.'
    )
    parser.add_argument(
        '--validation_batch_size',
        type=int,
        default=16,
        help=""
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default= d +'model/imagenet',
        help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
    )
    parser.add_argument(
        '--final_tensor_name',
        type=str,
        default='output_layer',
        help="""\
      The name of the output classification layer in the retrained graph.\
      """
    )
    parser.add_argument(
        '--architecture',
        type=str,
        default='inception_v3',
        help=""
    )
    parser.add_argument(
        '--saved_model_dir',
        type=str,
        default= d +'model/saved_models/' + name + '-' + strftime("%Y-%m-%d %H.%M.%S", gmtime()) + '/',
        help='Where to save the exported graph.')
    return parser.parse_known_args()
