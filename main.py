# -- Public Imports
import argparse
import tensorflow as tf

# -- Private Imports
import test_QE2E as tqf
import QE2E as qf
from parameters import config

# -- Global Variables
tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx('float64')

# -- Functions


def main(args):
    """
    Main function to either train or test the model based on the provided arguments.
    """
    config.update(args)

    if args.train_mode:
        qf.train(config)
    else:
        tqf.test(config)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train or test the model.')
    parser.add_argument('--train', dest='train_mode', action='store_true', help='Set this flag to train the model. Default is to test.')
    parser.add_argument('--test', dest='train_mode', action='store_false', help='Set this flag to test the model. This is the default mode.')
    parser.add_argument('--num_qubits', type=int, default=4, help='Number of qubits (channel_use). Default is 4.')
    parser.add_argument('--bit_num', type=int, default=4, help='Number of bits (bit_num). Default is 4.')
    parser.add_argument('--save', dest='save', action='store_true', help='Set this flag to save the models. Default is not to save.')
    parser.add_argument('--onehot', dest='use_onehot', action='store_true', help='Set this flag to use onehot encoding. Default is not to use onehot.')
    parser.add_argument('--retrain', dest='retrain', action='store_true', help='Set this flag to retrain the model. Default is to test..')
    parser.add_argument('--channel_type', dest='channel_type', type=str, choices=['awgn', 'rayleigh'], default='rayleigh', help="Specify the channel type.")
    
    parser.set_defaults(train_mode=True, save=True, use_onehot=True, channel_type='rayleigh')

    # Parse arguments
    args = parser.parse_args()

    # Call main with the appropriate arguments
    main(args)
